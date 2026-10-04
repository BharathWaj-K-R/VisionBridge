from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import decode_access_token
from app.db.models import User
from app.db.session import get_db

_bearer = HTTPBearer(auto_error=False)
settings = get_settings()


def _user_from_credentials(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
    db: Session,
) -> User | None:
    # An explicit bearer token identifies the API caller and must take precedence
    # over any ambient browser session cookie. Cookie auth remains the fallback for
    # normal browser requests that do not send an Authorization header.
    token = None
    if credentials is not None and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    if token is None:
        token = request.cookies.get(settings.AUTH_COOKIE_NAME)
    if token is None:
        return None
    subject = decode_access_token(token)
    if subject is None:
        return None
    try:
        user_id = int(subject)
    except (TypeError, ValueError):
        return None
    db.info["visionbridge_user_id"] = user_id
    db.info["visionbridge_auth_operation"] = "authenticated"
    db.info.pop("visionbridge_auth_identifier", None)
    db.info.pop("visionbridge_auth_username", None)
    db.info.pop("visionbridge_auth_email", None)
    return db.get(User, user_id)


def get_optional_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User | None:
    """Return the authenticated user when a bearer token is supplied.

    Anonymous base-model translation remains supported for the public demo,
    but invalid/malformed credentials are not treated as an authenticated user.
    """
    return _user_from_credentials(request, credentials, db)


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    user = _user_from_credentials(request, credentials, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
