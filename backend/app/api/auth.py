from __future__ import annotations

import datetime as dt
import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.csrf import csrf_cookie_options, new_csrf_token
from app.core.rate_limit import (
    SlidingWindowRateLimiter,
    enforce_limit,
    make_identifier_key,
    _client_key,
)
from app.core.security import (
    create_access_token,
    generate_otp,
    hash_otp,
    hash_password,
    validate_password_strength,
    verify_otp,
    verify_password,
)
from app.db.models import User
from app.db.session import get_db
from app.schemas.schemas import (
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    ResendOtpRequest,
    Token,
    UserOut,
    VerifyOtpRequest,
)
from app.services.email_verification import EmailDeliveryError, send_verification_email

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()

register_limiter = SlidingWindowRateLimiter(settings.REGISTER_RATE_LIMIT_PER_MINUTE)
verify_limiter = SlidingWindowRateLimiter(settings.VERIFY_OTP_RATE_LIMIT_PER_MINUTE)
resend_limiter = SlidingWindowRateLimiter(settings.RESEND_OTP_RATE_LIMIT_PER_HOUR, window_seconds=3600)


def _set_session_cookie(response: Response, user_id: int) -> str:
    token = create_access_token(subject=str(user_id))
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        httponly=True,
        samesite=settings.AUTH_COOKIE_SAMESITE,
    )
    return token


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _set_otp(user: User) -> tuple[str, dt.datetime]:
    otp = generate_otp()
    now = dt.datetime.now(dt.timezone.utc)
    expires_at = now + dt.timedelta(minutes=settings.OTP_EXPIRY_MINUTES)
    user.otp_hash = hash_otp(otp)
    user.otp_expires_at = expires_at
    user.otp_sent_at = now
    user.otp_attempts = 0
    return otp, expires_at


def _otp_cooldown_remaining(user: User) -> int:
    if user.otp_sent_at is None:
        return 0
    sent_at = user.otp_sent_at
    if sent_at.tzinfo is None:
        sent_at = sent_at.replace(tzinfo=dt.timezone.utc)
    elapsed = (dt.datetime.now(dt.timezone.utc) - sent_at).total_seconds()
    return max(0, settings.OTP_COOLDOWN_SECONDS - int(elapsed))


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    enforce_limit(register_limiter, _client_key(request))

    username = payload.username.strip()
    email = _normalize_email(payload.email)

    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")

    missing_rules = validate_password_strength(payload.password)
    if missing_rules:
        raise HTTPException(
            status_code=400,
            detail="Password requirements not met: " + ", ".join(missing_rules) + ".",
        )

    db.info["visionbridge_auth_operation"] = "register"
    db.info["visionbridge_auth_username"] = username
    db.info["visionbridge_auth_email"] = email

    existing_username = db.query(User).filter(User.username == username).first()
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already taken")

    existing_email = db.query(User).filter(User.email == email).first()
    if existing_email:
        if not existing_email.is_verified:
            raise HTTPException(
                status_code=409,
                detail="An account with this email is waiting for verification. Check your email or use resend verification.",
            )
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(payload.password),
        is_verified=False,
    )
    otp, _ = _set_otp(user)
    db.add(user)

    try:
        db.flush()
        send_verification_email(email, otp)
        db.commit()
        db.refresh(user)
    except EmailDeliveryError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        db.rollback()
        raise HTTPException(status_code=503, detail="We could not create the account right now. Please try again.") from None

    return RegisterResponse(
        message="Account created. Check your email for the 6-digit verification code.",
        email=email,
        verification_required=True,
        resend_after_seconds=settings.OTP_COOLDOWN_SECONDS,
    )


@router.post("/verify-otp", response_model=Token)
def verify_email(
    payload: VerifyOtpRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    email = _normalize_email(payload.email)
    identifier_key = make_identifier_key(request, email)
    enforce_limit(verify_limiter, identifier_key)

    db.info["visionbridge_auth_operation"] = "verify_otp"
    db.info["visionbridge_auth_email"] = email

    user = db.query(User).filter(User.email == email).first()
    generic_error = "The verification code is invalid or has expired. Request a new code and try again."

    if user is None or user.is_verified:
        raise HTTPException(status_code=400, detail=generic_error)

    if user.otp_attempts >= settings.OTP_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many incorrect verification attempts. Request a new code and try again later.",
        )

    expires_at = user.otp_expires_at
    if expires_at is None:
        raise HTTPException(status_code=400, detail=generic_error)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=dt.timezone.utc)

    if dt.datetime.now(dt.timezone.utc) >= expires_at:
        raise HTTPException(status_code=400, detail="This verification code has expired. Request a new code.")

    user.otp_attempts += 1
    if not user.otp_hash or not verify_otp(payload.otp, user.otp_hash):
        db.commit()
        if user.otp_attempts >= settings.OTP_MAX_ATTEMPTS:
            raise HTTPException(
                status_code=429,
                detail="Too many incorrect verification attempts. Request a new code and try again later.",
            )
        remaining = settings.OTP_MAX_ATTEMPTS - user.otp_attempts
        raise HTTPException(
            status_code=400,
            detail=f"That code is not correct. You have {remaining} attempt" + ("." if remaining == 1 else "s left."),
        )

    user.is_verified = True
    user.otp_hash = None
    user.otp_expires_at = None
    user.otp_sent_at = None
    user.otp_attempts = 0
    db.commit()

    _set_session_cookie(response, user.id)
    return Token(access_token=create_access_token(subject=str(user.id)))


@router.post("/resend-otp")
def resend_otp(
    payload: ResendOtpRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    email = _normalize_email(payload.email)
    enforce_limit(resend_limiter, _client_key(request))
    enforce_limit(resend_limiter, make_identifier_key(request, email))

    db.info["visionbridge_auth_operation"] = "resend_otp"
    db.info["visionbridge_auth_email"] = email

    user = db.query(User).filter(User.email == email).first()
    generic = {
        "message": "If that account can receive verification email, a new code has been sent.",
        "resend_after_seconds": settings.OTP_COOLDOWN_SECONDS,
    }
    if user is None or user.is_verified:
        return generic

    remaining = _otp_cooldown_remaining(user)
    if remaining:
        raise HTTPException(
            status_code=429,
            detail=f"Please wait {remaining} seconds before requesting another code.",
            headers={"Retry-After": str(remaining)},
        )

    otp, _ = _set_otp(user)
    try:
        send_verification_email(email, otp)
        db.commit()
    except EmailDeliveryError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        db.rollback()
        raise HTTPException(status_code=503, detail="We could not resend the verification email. Please try again.") from None

    return generic


@router.post("/login", response_model=Token)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    identifier = payload.identifier.strip()
    db.info["visionbridge_auth_operation"] = "login"
    db.info["visionbridge_auth_identifier"] = identifier

    user = db.query(User).filter(
        (User.username == identifier) | (User.email == identifier.lower())
    ).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username/email or password")

    if not user.is_verified:
        raise HTTPException(
            status_code=403,
            detail="Please verify your email before signing in.",
        )

    token = _set_session_cookie(response, user.id)
    return Token(access_token=token)


@router.get("/csrf")
def csrf(response: Response) -> dict[str, str]:
    token = new_csrf_token()
    response.set_cookie(value=token, **csrf_cookie_options())
    return {"csrf_token": token}


@router.post("/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(
        key=settings.AUTH_COOKIE_NAME,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        samesite=settings.AUTH_COOKIE_SAMESITE,
    )
    response.delete_cookie(
        key=settings.CSRF_COOKIE_NAME,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        samesite=settings.AUTH_COOKIE_SAMESITE,
    )
    return {"logged_out": True}
