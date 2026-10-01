import datetime as dt
import uuid

from fastapi.testclient import TestClient

from app.api import auth as auth_api
from app.core.config import get_settings
from app.core.security import hash_password
from app.db.models import User
from app.db.session import SessionLocal
from app.main import app


def _reset_auth_limiters() -> None:
    auth_api.register_limiter.reset()
    auth_api.verify_limiter.reset()
    auth_api.resend_limiter.reset()


def test_registration_requires_all_strong_password_rules():
    _reset_auth_limiters()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "weak-" + uuid.uuid4().hex[:8],
                "email": uuid.uuid4().hex[:8] + "@example.com",
                "password": "password",
            },
        )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "uppercase" in detail
    assert "number" in detail
    assert "special" in detail


def test_registration_creates_unverified_user_and_sends_hashed_otp(monkeypatch):
    _reset_auth_limiters()
    captured: dict[str, str] = {}

    def fake_send(email: str, otp: str) -> None:
        captured["email"] = email
        captured["otp"] = otp

    monkeypatch.setattr(auth_api, "send_verification_email", fake_send)

    username = "otp-" + uuid.uuid4().hex[:8]
    email = uuid.uuid4().hex[:8] + "@example.com"

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": "StrongPass1!",
            },
        )

    settings = get_settings()
    assert response.status_code == 201
    assert response.cookies.get(settings.AUTH_COOKIE_NAME) is None
    assert response.json()["verification_required"] is True
    assert response.json()["email"] == email

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).one()
        assert user.is_verified is False
        assert user.otp_hash
        assert user.otp_hash != captured["otp"]
        assert user.otp_expires_at is not None
        assert user.otp_expires_at > dt.datetime.now(dt.timezone.utc)
        assert user.otp_attempts == 0
    finally:
        db.close()


def test_wrong_otp_increments_attempts_and_correct_otp_logs_user_in(monkeypatch):
    _reset_auth_limiters()
    captured: dict[str, str] = {}

    def fake_send(email: str, otp: str) -> None:
        captured["otp"] = otp

    monkeypatch.setattr(auth_api, "send_verification_email", fake_send)

    email = uuid.uuid4().hex[:8] + "@example.com"
    username = "verify-" + uuid.uuid4().hex[:8]

    with TestClient(app) as client:
        registration = client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": "StrongPass1!",
            },
        )
        assert registration.status_code == 201

        wrong = client.post(
            "/api/v1/auth/verify-otp",
            json={"email": email, "otp": "000000"},
        )
        assert wrong.status_code == 400

        verified = client.post(
            "/api/v1/auth/verify-otp",
            json={"email": email, "otp": captured["otp"]},
        )
        assert verified.status_code == 200
        assert verified.cookies.get(get_settings().AUTH_COOKIE_NAME)

        me = client.get("/api/v1/users/me")
        assert me.status_code == 200
        assert me.json()["is_verified"] is True

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).one()
        assert user.is_verified is True
        assert user.otp_hash is None
        assert user.otp_expires_at is None
        assert user.otp_attempts == 0
    finally:
        db.close()


def test_expired_otp_is_rejected(monkeypatch):
    _reset_auth_limiters()
    monkeypatch.setattr(auth_api, "send_verification_email", lambda _email, _otp: None)

    email = uuid.uuid4().hex[:8] + "@example.com"
    username = "expired-" + uuid.uuid4().hex[:8]

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": "StrongPass1!",
            },
        )
        assert response.status_code == 201

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).one()
        user.otp_expires_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        db.commit()
    finally:
        db.close()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/verify-otp",
            json={"email": email, "otp": "123456"},
        )
        assert response.status_code == 400
        assert "expired" in response.json()["detail"].lower()


def test_login_rejects_unverified_user(monkeypatch):
    _reset_auth_limiters()
    monkeypatch.setattr(auth_api, "send_verification_email", lambda _email, _otp: None)

    email = uuid.uuid4().hex[:8] + "@example.com"
    username = "blocked-" + uuid.uuid4().hex[:8]

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": "StrongPass1!",
            },
        )
        assert response.status_code == 201

        login = client.post(
            "/api/v1/auth/login",
            json={"identifier": username, "password": "StrongPass1!"},
        )

    assert login.status_code == 403
    assert "verify your email" in login.json()["detail"].lower()


def test_resend_obeys_cooldown(monkeypatch):
    _reset_auth_limiters()
    sent: list[str] = []
    monkeypatch.setattr(auth_api, "send_verification_email", lambda _email, otp: sent.append(otp))

    email = uuid.uuid4().hex[:8] + "@example.com"
    username = "resend-" + uuid.uuid4().hex[:8]

    with TestClient(app) as client:
        registration = client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": "StrongPass1!",
            },
        )
        assert registration.status_code == 201

        response = client.post("/api/v1/auth/resend-otp", json={"email": email})

    assert response.status_code == 429
    assert len(sent) == 1
