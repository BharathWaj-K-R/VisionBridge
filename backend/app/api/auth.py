from __future__ import annotations

import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.csrf import csrf_cookie_options, new_csrf_token
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User
from app.db.session import get_db
from app.schemas.schemas import LoginRequest, RegisterRequest, Token, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()
GOOGLE_STATE_COOKIE = "__Host-visionbridge_google_state"


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


def _google_configured() -> bool:
    return bool(
        settings.GOOGLE_CLIENT_ID
        and settings.GOOGLE_CLIENT_SECRET
        and settings.GOOGLE_REDIRECT_URI
    )


@router.post("/register", response_model=UserOut)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    username = payload.username.strip()
    email = payload.email.strip().lower()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    identifier = payload.identifier.strip()
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
    response.delete_cookie(
        key=GOOGLE_STATE_COOKIE,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        samesite="lax",
    )
    return {"logged_out": True}


@router.get("/google")
def google_login():
    if not _google_configured():
        raise HTTPException(
            status_code=503,
            detail="Google sign-in is not configured on the server",
        )

    state = secrets.token_urlsafe(32)
    response = RedirectResponse(
        "https://accounts.google.com/o/oauth2/v2/auth?"
        + urlencode(
            {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
                "access_type": "online",
                "prompt": "select_account",
            }
        )
    )
    response.set_cookie(
        key=GOOGLE_STATE_COOKIE,
        value=state,
        max_age=600,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    if error:
        return RedirectResponse(settings.FRONTEND_URL + "/login?oauth_error=" + error)
    if not code or not state:
        return RedirectResponse(settings.FRONTEND_URL + "/login?oauth_error=missing_response")

    expected_state = request.cookies.get(GOOGLE_STATE_COOKIE)
    if not expected_state or not secrets.compare_digest(expected_state, state):
        return RedirectResponse(settings.FRONTEND_URL + "/login?oauth_error=invalid_state")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]

            user_response = await client.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": "Bearer " + access_token},
            )
            user_response.raise_for_status()
            profile = user_response.json()
    except (httpx.HTTPError, KeyError, ValueError):
        return RedirectResponse(settings.FRONTEND_URL + "/login?oauth_error=google_exchange_failed")

    google_sub = str(profile.get("sub") or "").strip()
    email = str(profile.get("email") or "").strip().lower()
    if not google_sub or not email or profile.get("email_verified") is not True:
        return RedirectResponse(settings.FRONTEND_URL + "/login?oauth_error=email_not_verified")

    user = db.query(User).filter(User.google_sub == google_sub).first()
    if user is None:
        user = db.query(User).filter(User.email == email).first()
    if user is None:
        base = "".join(ch for ch in email.split("@")[0] if ch.isalnum() or ch in "._-").strip("._-") or "googleuser"
        username = base[:110]
        suffix = 1
        while db.query(User).filter(User.username == username).first():
            username = f"{base[:105]}-{suffix}"
            suffix += 1
        user = User(
            username=username,
            email=email,
            google_sub=google_sub,
            hashed_password=hash_password(secrets.token_urlsafe(32)),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    if user.google_sub is None:
        user.google_sub = google_sub
        db.commit()
    response = RedirectResponse(settings.FRONTEND_URL + "/login?oauth_success=1")
    _set_session_cookie(response, user.id)
    response.delete_cookie(
        key=GOOGLE_STATE_COOKIE,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        samesite="lax",
    )
    return response
