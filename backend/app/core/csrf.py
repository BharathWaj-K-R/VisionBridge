"""CSRF helpers for cookie-backed browser sessions.

The session JWT is HttpOnly, so browser JavaScript cannot read it. A separate
random CSRF token is issued in a readable cookie and must be echoed in the
X-CSRF-Token request header for browser state-changing requests.
"""
import secrets

from fastapi import HTTPException, Request, status

from app.core.config import get_settings

settings = get_settings()
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
BROWSER_AUTH_PATHS = {
    "/api/v1/auth/login",
    "/api/v1/auth/register",
}


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_cookie_options() -> dict[str, object]:
    return {
        "key": settings.CSRF_COOKIE_NAME,
        "path": "/",
        "secure": settings.AUTH_COOKIE_SECURE,
        "httponly": False,
        "samesite": settings.AUTH_COOKIE_SAMESITE,
    }


def validate_csrf(request: Request) -> None:
    if request.method.upper() in SAFE_METHODS:
        return
    if request.url.path == "/api/v1/auth/csrf":
        return

    session_cookie = request.cookies.get(settings.AUTH_COOKIE_NAME)
    csrf_cookie = request.cookies.get(settings.CSRF_COOKIE_NAME)
    csrf_header = request.headers.get("X-CSRF-Token")

    if request.url.path in BROWSER_AUTH_PATHS:
        origin = request.headers.get("Origin")
        if origin and origin not in settings.ALLOWED_ORIGINS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Origin not allowed",
            )
        if csrf_cookie is not None:
            if not csrf_header or not secrets.compare_digest(csrf_cookie, csrf_header):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="CSRF validation failed",
                )
        return

    # Bearer-only legacy API clients do not have an automatically-sent
    # session cookie and therefore are not subject to browser CSRF.
    if session_cookie is None:
        return

    if not csrf_cookie or not csrf_header or not secrets.compare_digest(csrf_cookie, csrf_header):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed",
        )
