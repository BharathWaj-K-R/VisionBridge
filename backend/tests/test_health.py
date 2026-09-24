from fastapi.testclient import TestClient

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
    assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"
