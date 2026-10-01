import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token
from app.db.models import User
from app.db.session import SessionLocal


def _token_and_user():
    db = SessionLocal()
    username = f"profile-test-{uuid.uuid4().hex[:8]}"
    user = User(username=username, email=f"{username}@example.com", hashed_password="test", is_verified=True)
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(str(user.id))
    db.close()
    return token, user.id


def _config():
    return {
        "avatar": {
            "skinTone": "#B97A56",
            "hair": "short",
            "hairColor": "#171717",
            "shirtColor": "#2053A6",
            "bodyShape": "average",
            "apparel": "tee",
            "highContrast": False,
        },
        "quickAccess": ["Hello", None, None, None, None, None, None, None, None, None],
        "favorites": ["Hello"],
        "signingSpeed": 1,
        "ttsVoice": None,
    }


def test_profile_crud_usage_and_delete_cleanup():
    token, user_id = _token_and_user()
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/communication/profiles",
            headers=headers,
            json={"name": "Home", "config": _config()},
        )
        assert created.status_code == 201, created.text
        profile_id = created.json()["id"]

        duplicate = client.post(
            "/api/v1/communication/profiles",
            headers=headers,
            json={"name": "home", "config": _config()},
        )
        assert duplicate.status_code == 409

        usage = client.post(
            "/api/v1/communication/usage",
            headers=headers,
            json={"profileId": profile_id, "phrase": "Hello"},
        )
        assert usage.status_code == 200

        most_used = client.get(
            f"/api/v1/communication/most-used?profileId={profile_id}",
            headers=headers,
        )
        assert most_used.status_code == 200
        assert most_used.json()[0]["phrase"] == "Hello"
        assert most_used.json()[0]["usage_count"] == 1

        second = client.post(
            "/api/v1/communication/profiles",
            headers=headers,
            json={"name": "Work", "config": _config()},
        )
        assert second.status_code == 201

        deleted = client.delete(
            f"/api/v1/communication/profiles/{profile_id}",
            headers=headers,
        )
        assert deleted.status_code == 200

        listed = client.get("/api/v1/communication/profiles", headers=headers)
        assert listed.status_code == 200
        assert len(listed.json()) == 1
        assert listed.json()[0]["name"] == "Work"

        deleted_last = client.delete(
            f"/api/v1/communication/profiles/{second.json()['id']}",
            headers=headers,
        )
        assert deleted_last.status_code == 409

    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        assert user is not None
    finally:
        db.close()


def test_profile_update_preserves_user_scope():
    token_one, user_one = _token_and_user()
    token_two, _user_two = _token_and_user()
    headers_one = {"Authorization": f"Bearer {token_one}"}
    headers_two = {"Authorization": f"Bearer {token_two}"}

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/communication/profiles",
            headers=headers_one,
            json={"name": "Work", "config": _config()},
        )
        assert created.status_code == 201
        profile_id = created.json()["id"]

        forbidden_update = client.put(
            f"/api/v1/communication/profiles/{profile_id}",
            headers=headers_two,
            json={"name": "Hijack", "config": _config()},
        )
        assert forbidden_update.status_code == 404

    db = SessionLocal()
    try:
        profile_count = db.execute(
            __import__("sqlalchemy").text(
                "select count(*) from personalization_profiles where user_id=:uid"
            ),
            {"uid": user_one},
        ).scalar_one()
        assert profile_count == 1
    finally:
        db.close()
