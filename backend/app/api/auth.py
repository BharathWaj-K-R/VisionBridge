from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.csrf import csrf_cookie_options, new_csrf_token
from app.core.rate_limit import SlidingWindowRateLimiter, enforce_limit, _client_key
from app.core.security import (
    create_access_token,
    hash_password,
    validate_password_strength,
    verify_password,
)
from app.db.models import User
from app.db.session import get_db
from app.schemas.schemas import LoginRequest, RegisterRequest, Token

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()

register_limiter = SlidingWindowRateLimiter(settings.REGISTER_RATE_LIMIT_PER_MINUTE)


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


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
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

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(payload.password),
    )

    try:
        db.add(user)
        db.flush()
        db.commit()
        db.refresh(user)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="We could not create the account right now. Please try again.",
        ) from None

    token = _set_session_cookie(response, user.id)
    return Token(access_token=token)


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
