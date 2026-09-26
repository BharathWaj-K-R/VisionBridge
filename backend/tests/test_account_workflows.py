from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_private_account_endpoints_require_authentication():
    with TestClient(app) as client:
        assert client.get("/api/v1/dashboard").status_code == 401
        assert client.get("/api/v1/history").status_code == 401
        assert client.get("/api/v1/history/export.csv").status_code == 401
        assert client.get("/api/v1/users/me").status_code == 401
        assert client.get("/api/v1/letter/model").status_code == 401


def test_registration_rejects_invalid_credentials():
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/register", json={"username": "bad user", "password": "short"})
        assert response.status_code == 422


def test_unknown_adapter_delete_is_not_successful():
    with TestClient(app) as client:
        response = client.delete("/api/v1/users/me/adapters/999999")
        assert response.status_code == 401


def test_adapter_delete_preserves_history_and_clears_adapter_reference():
    import uuid
    from app.db.models import SignerAdapter, TranslationLog, User
    from app.db.session import SessionLocal
    from app.core.config import get_settings
    from app.core.security import hash_password

    username = f"delete-user-{uuid.uuid4().hex[:8]}"
    db = SessionLocal()
    user = User(username=username, hashed_password=hash_password("correct horse battery staple"))
    db.add(user)
    db.commit()
    db.refresh(user)

    adapter = SignerAdapter(
        owner_id=user.id,
        weights_path=str(Path(get_settings().ADAPTER_WEIGHTS_DIR) / "letter_adapter_test.json"),
        calibration_seconds=3,
        param_count=126,
    )
    db.add(adapter)
    db.commit()
    db.refresh(adapter)

    log = TranslationLog(
        user_id=user.id,
        adapter_id=adapter.id,
        predicted_text="A",
        confidence=0.9,
        latency_ms=1.0,
        used_adapter=1,
    )
    db.add(log)
    db.commit()
    log_id = log.id
    adapter_id = adapter.id
    user_id = user.id
    db.close()

    client = TestClient(app)
    client.__enter__()
    try:
        from app.core.security import create_access_token
        token = create_access_token(str(user_id))
        response = client.delete(
            f"/api/v1/users/me/adapters/{adapter_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text
    finally:
        client.__exit__(None, None, None)

    db = SessionLocal()
    try:
        assert db.get(SignerAdapter, adapter_id) is None
        preserved = db.get(TranslationLog, log_id)
        assert preserved is not None
        assert preserved.predicted_text == "A"
        assert preserved.adapter_id is None
    finally:
        db.close()


def test_calibration_commit_failure_removes_new_adapter_file(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from app.api import letter as letter_api
    from app.schemas.schemas import LetterCalibrationRequest

    class User:
        id = 7

    class BrokenDB:
        def __init__(self):
            self.added = None
            self.rolled_back = False

        def add(self, row):
            self.added = row

        def commit(self):
            raise RuntimeError("simulated database failure")

        def rollback(self):
            self.rolled_back = True

        def refresh(self, row):
            raise AssertionError("refresh must not run after a failed commit")

    weights_path = tmp_path / "adapters" / "letter_adapter_failure.json"
    weights_path.parent.mkdir()

    def fake_save(_payload):
        weights_path.write_text("adapter", encoding="utf-8")
        return str(weights_path)

    monkeypatch.setattr(
        letter_api,
        "get_letter_base_model",
        lambda: object(),
    )
    monkeypatch.setattr(
        letter_api,
        "fit_prototype_adapter",
        lambda *_args: {"payload": {}, "letters": ["A", "B"], "shots": {"A": 3, "B": 3}, "param_count": 252},
    )
    monkeypatch.setattr(letter_api, "save_prototype_adapter", fake_save)

    vector_a = [1.0] + [0.0] * 125
    vector_b = [0.0, 1.0] + [0.0] * 124
    payload = LetterCalibrationRequest(
        user_id=7,
        samples=[
            {"letter": "A", "hand_keypoints": vector_a},
            {"letter": "A", "hand_keypoints": vector_a},
            {"letter": "A", "hand_keypoints": vector_a},
            {"letter": "B", "hand_keypoints": vector_b},
            {"letter": "B", "hand_keypoints": vector_b},
            {"letter": "B", "hand_keypoints": vector_b},
        ],
    )
    db = BrokenDB()

    try:
        letter_api.calibrate_letters(payload, db=db, current_user=User())
    except HTTPException as exc:
        assert exc.status_code == 500
        assert "persisted safely" in exc.detail
    else:
        raise AssertionError("expected calibration persistence failure")

    assert db.rolled_back is True
    assert not weights_path.exists()


def test_adapter_delete_restores_staged_file_when_history_update_fails(monkeypatch, tmp_path):
    from app.api import users as users_api
    from app.db.models import SignerAdapter, TranslationLog

    class User:
        id = 7

    original = tmp_path / "letter_adapter_failure.json"
    original.write_text("adapter", encoding="utf-8")

    class BrokenQuery:
        def filter(self, *_args, **_kwargs):
            return self

        def first(self):
            return type("Adapter", (), {
                "id": 11,
                "owner_id": 7,
                "weights_path": str(original),
            })()

        def update(self, *_args, **_kwargs):
            raise RuntimeError("simulated history update failure")

    class BrokenDB:
        def __init__(self):
            self.rolled_back = False

        def query(self, *_args):
            return BrokenQuery()

        def rollback(self):
            self.rolled_back = True

        def delete(self, *_args):
            raise AssertionError("adapter delete must not run after history update failure")

        def commit(self):
            raise AssertionError("commit must not run after history update failure")

    monkeypatch.setattr(users_api, "stage_adapter_delete", lambda path: original.with_name(".staged.deleting"))
    monkeypatch.setattr(users_api, "restore_adapter_delete", lambda tombstone, path: Path(path).write_text("adapter", encoding="utf-8"))
    monkeypatch.setattr(users_api, "finalize_adapter_delete", lambda _tombstone: None)

    db = BrokenDB()
    try:
        users_api.delete_my_adapter(11, db=db, current_user=User())
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 500
    else:
        raise AssertionError("expected adapter deletion failure")

    assert db.rolled_back is True
    assert original.exists()
