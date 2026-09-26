from fastapi.testclient import TestClient

import pytest

from app.main import app

client = TestClient(app)


def test_health_is_process_liveness():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["project"] == "VisionBridge"
    assert "model" not in body


def test_ready_reports_model_state():
    response = client.get("/api/v1/ready")
    assert response.status_code in {200, 503}
    body = response.json()
    assert body["project"] == "VisionBridge"
    assert body["status"] in {"ok", "degraded"}
    assert "model" in body
    if response.status_code == 200:
        assert body["model"]["status"] == "ready"
    else:
        assert body["status"] == "degraded"


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_security_headers_are_present():
    response = client.get("/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Permissions-Policy"] == "camera=(self), microphone=(), geolocation=()"


def test_evaluator_requires_all_a_z_classes(tmp_path):
    import numpy as np
    from app.models.letter_model import VisionBridgeLetterBaseModel, save_checkpoint
    from app.training.evaluate_letter_base import evaluate

    checkpoint = tmp_path / "base.pt"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_checkpoint(VisionBridgeLetterBaseModel(), checkpoint)

    x = np.zeros((26, 126), dtype=np.float32)
    y = np.zeros(26, dtype=np.int64)
    np.savez_compressed(data_dir / "test.npz", x=x, y=y)

    with pytest.raises(ValueError, match="missing required A-Z classes"):
        evaluate(checkpoint, data_dir, "test", 26)


def test_production_rejects_ephemeral_sqlite():
    from app.core.config import Settings

    settings = Settings()
    settings.ENV = "production"
    settings.SECRET_KEY = "production-secret"
    settings.DATABASE_URL = "sqlite:///./visionbridge.db"

    with pytest.raises(RuntimeError, match="durable database"):
        settings.validate_for_runtime()


def test_production_rejects_local_cors_origin():
    from app.core.config import Settings

    settings = Settings()
    settings.ENV = "production"
    settings.SECRET_KEY = "production-secret"
    settings.DATABASE_URL = "postgresql://example"
    settings.ALLOWED_ORIGINS = ["https://visionbridge-2c7h.onrender.com", "http://localhost:5173"]

    with pytest.raises(RuntimeError, match="localhost origins"):
        settings.validate_for_runtime()
