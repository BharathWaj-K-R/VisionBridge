from fastapi.testclient import TestClient

from app.main import app


def test_private_account_endpoints_require_authentication():
    with TestClient(app) as client:
        assert client.get("/api/v1/dashboard").status_code == 401
        assert client.get("/api/v1/history").status_code == 401
        assert client.get("/api/v1/users/me").status_code == 401
        assert client.get("/api/v1/evaluation").status_code == 401


def test_registration_rejects_invalid_credentials():
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/register", json={"username": "bad user", "password": "short"})
        assert response.status_code == 422


def test_unknown_adapter_delete_is_not_successful():
    with TestClient(app) as client:
        response = client.delete("/api/v1/users/me/adapters/999999")
        assert response.status_code == 401
