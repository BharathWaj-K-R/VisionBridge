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


def test_registration_creates_active_user_and_sets_session_cookie():
    _reset_auth_limiters()
    username = "register-" + uuid.uuid4().hex[:8]
    email = f"{username}@example.com"
    settings = get_settings()

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
        assert response.json()["access_token"]
        assert response.cookies.get(settings.AUTH_COOKIE_NAME)

        me = client.get("/api/v1/users/me")
        assert me.status_code == 200
        assert me.json()["username"] == username
        assert me.json()["email"] == email

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).one()
        assert user.username == username
        assert user.hashed_password
    finally:
        db.close()


def test_existing_user_can_log_in_normally():
    _reset_auth_limiters()
    username = "existing-" + uuid.uuid4().hex[:8]
    password = "StrongPass1!"
    db = SessionLocal()
    user = User(
        username=username,
        email=f"{username}@example.com",
        hashed_password=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/login",
            json={"identifier": username, "password": password},
        )
        assert response.status_code == 200
        assert response.cookies.get(get_settings().AUTH_COOKIE_NAME)


def test_registration_rejects_duplicate_username_or_email():
    _reset_auth_limiters()
    username = "duplicate-" + uuid.uuid4().hex[:8]
    email = f"{username}@example.com"

    with TestClient(app) as client:
        first = client.post(
            "/api/v1/auth/register",
            json={"username": username, "email": email, "password": "StrongPass1!"},
        )
        assert first.status_code == 201

        duplicate_username = client.post(
            "/api/v1/auth/register",
            json={"username": username, "email": f"other-{email}", "password": "StrongPass1!"},
        )
        assert duplicate_username.status_code == 400

        duplicate_email = client.post(
            "/api/v1/auth/register",
            json={"username": f"other-{username}", "email": email, "password": "StrongPass1!"},
        )
        assert duplicate_email.status_code == 400
