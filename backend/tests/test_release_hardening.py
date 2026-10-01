import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.security import hash_password
from app.api import auth as auth_api
from app.db.models import TranslationLog, User
from app.db.session import Base, SessionLocal, engine
from app.main import app
from app.models.letter_model import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    INPUT_DIM,
    LETTER_LABELS,
    VisionBridgeLetterBaseModel,
    load_checkpoint,
    save_checkpoint,
)
from app.services.letter_fewshot import (
    fit_prototype_adapter,
    validate_prototype_adapter_payload,
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client


def _checkpoint(path):
    model = VisionBridgeLetterBaseModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        labels=LETTER_LABELS,
        dropout=0.10,
    )
    save_checkpoint(model, path)
    return model


def _sample(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=INPUT_DIM).astype(np.float32).tolist()


def _csrf(client):
    response = client.get("/api/v1/auth/csrf")
    assert response.status_code == 200
    return response.json()["csrf_token"]


def test_model_loading_and_contract(tmp_path):
    path = tmp_path / "letter_base_model.pt"
    _checkpoint(path)
    model = load_checkpoint(path)
    assert model.input_dim == 126
    assert model.hidden_dim == 128
    assert model.embedding_dim == 64
    assert model.num_classes == 26
    assert tuple(model.labels) == LETTER_LABELS


def test_base_model_prediction_returns_valid_letter(tmp_path):
    path = tmp_path / "letter_base_model.pt"
    model = _checkpoint(path)
    x = np.asarray(_sample(1), dtype=np.float32)
    import torch

    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.from_numpy(x).unsqueeze(0)), dim=1)[0]
    index = int(probabilities.argmax())
    assert model.labels[index] in LETTER_LABELS
    assert 0.0 <= float(probabilities[index]) <= 1.0


def test_adapter_compatibility_rejects_wrong_model_hash(tmp_path, monkeypatch):
    path = tmp_path / "letter_base_model.pt"
    _checkpoint(path)

    from app.services import letter_fewshot

    monkeypatch.setattr(letter_fewshot.settings, "LETTER_BASE_MODEL_PATH", str(path))
    model = load_checkpoint(path)
    samples = []
    for letter, offset in (("A", 0), ("B", 10)):
        for shot in range(3):
            samples.append((letter, _sample(offset + shot)))

    fitted = fit_prototype_adapter(model, samples)
    validate_prototype_adapter_payload(fitted["payload"], path)

    broken = json.loads(json.dumps(fitted["payload"]))
    broken["base_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recalibration"):
        validate_prototype_adapter_payload(broken, path)


def test_protected_route_requires_authentication(client):
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401


def test_login_and_history_are_user_scoped(client, monkeypatch):
    csrf = _csrf(client)
    otp_by_email = {}
    monkeypatch.setattr(
        auth_api,
        "send_verification_email",
        lambda email, otp: otp_by_email.__setitem__(email, otp),
    )
    first = client.post(
        "/api/v1/auth/register",
        headers={"X-CSRF-Token": csrf},
        json={"username": "release-user-a", "email": "release-a@example.com", "password": "StrongPass123!"},
    )
    assert first.status_code == 200
    verify_first = client.post(
        "/api/v1/auth/verify-otp",
        headers={"X-CSRF-Token": csrf},
        json={"email": "release-a@example.com", "otp": otp_by_email["release-a@example.com"]},
    )
    assert verify_first.status_code == 200

    second = client.post(
        "/api/v1/auth/register",
        headers={"X-CSRF-Token": csrf},
        json={"username": "release-user-b", "email": "release-b@example.com", "password": "StrongPass123!"},
    )
    assert second.status_code == 200
    verify_second = client.post(
        "/api/v1/auth/verify-otp",
        headers={"X-CSRF-Token": csrf},
        json={"email": "release-b@example.com", "otp": otp_by_email["release-b@example.com"]},
    )
    assert verify_second.status_code == 200

    db = SessionLocal()
    try:
        users = {
            row.username: row
            for row in db.query(User).filter(User.username.in_(["release-user-a", "release-user-b"])).all()
        }
        db.add_all([
            TranslationLog(user_id=users["release-user-a"].id, predicted_text="A", confidence=0.9, latency_ms=1.0),
            TranslationLog(user_id=users["release-user-b"].id, predicted_text="B", confidence=0.9, latency_ms=1.0),
        ])
        db.commit()
    finally:
        db.close()

    login = client.post(
        "/api/v1/auth/login",
        headers={"X-CSRF-Token": csrf},
        json={"identifier": "release-user-a", "password": "StrongPass123!"},
    )
    assert login.status_code == 200

    me = client.get("/api/v1/users/me")
    assert me.status_code == 200
    assert me.json()["username"] == "release-user-a"

    history = client.get("/api/v1/history?range=all")
    assert history.status_code == 200
    assert {item["predicted_text"] for item in history.json()["items"]} == {"A"}
