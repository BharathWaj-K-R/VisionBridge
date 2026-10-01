"""
Password hashing + JWT helpers.

Authentication continues to use bcrypt for passwords and HS256 JWTs for the
existing HttpOnly-cookie session. OTPs use an HMAC-SHA256 server-secret hash
because a six-digit code is too small to protect with an unkeyed digest if the
database were ever exposed.
"""
import datetime as dt
import hashlib
import hmac
import secrets

from jose import jwt
from passlib.context import CryptContext

from app.core.config import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


PASSWORD_RULES = (
    ("at least 8 characters", lambda password: len(password) >= 8),
    ("at least 1 uppercase letter", lambda password: any(char.isupper() for char in password)),
    ("at least 1 lowercase letter", lambda password: any(char.islower() for char in password)),
    ("at least 1 number", lambda password: any(char.isdigit() for char in password)),
    ("at least 1 special character", lambda password: any(not char.isalnum() for char in password)),
)


def validate_password_strength(password: str) -> list[str]:
    """Return the human-readable password requirements that are still missing."""
    return [message for message, rule in PASSWORD_RULES if not rule(password)]


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(otp: str) -> str:
    return hmac.new(
        settings.SECRET_KEY.encode("utf-8"),
        otp.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_otp(otp: str, otp_hash: str) -> bool:
    expected = hash_otp(otp)
    return hmac.compare_digest(expected, otp_hash)


def create_access_token(subject: str) -> str:
    expire = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("sub")
    except Exception:
        return None
