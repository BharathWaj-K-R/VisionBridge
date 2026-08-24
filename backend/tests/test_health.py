from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["project"] == "VisionBridge"
    assert response.json()["model"]["status"] == "ready"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
