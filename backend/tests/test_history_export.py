"""End-to-end coverage for GET /api/v1/history/export.csv, which had no
test coverage at all before this — and, until recently, no frontend UI
to reach it either (the React rewrite dropped the export button that
existed in the previous vanilla-JS frontend; restored in App.tsx)."""
import csv
import io
import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
client.__enter__()


def _register_and_login(username: str, password: str = "correct horse battery staple") -> str:
    register_resp = client.post("/api/v1/auth/register", json={"username": username, "password": password})
    assert register_resp.status_code == 200, register_resp.text
    login_resp = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert login_resp.status_code == 200, login_resp.text
    return login_resp.json()["access_token"]


def test_export_csv_contains_the_users_own_prediction():
    from app.db.models import TranslationLog
    from app.db.session import SessionLocal

    username = f"export-user-{uuid.uuid4().hex[:8]}"
    token = _register_and_login(username)
    user_resp = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})
    user_id = user_resp.json()["id"]

    db = SessionLocal()
    db.add(
        TranslationLog(
            user_id=user_id,
            predicted_text="A",
            confidence=0.75,
            latency_ms=0.8,
            used_adapter=1,
        )
    )
    db.commit()
    db.close()

    export_resp = client.get("/api/v1/history/export.csv", headers={"Authorization": f"Bearer {token}"})
    assert export_resp.status_code == 200, export_resp.text

    rows = list(csv.reader(io.StringIO(export_resp.text)))
    assert rows[0] == ["id", "created_at", "predicted_text", "confidence", "latency_ms", "used_adapter"]
    assert any(row[2] == "A" for row in rows[1:])


def test_export_csv_does_not_leak_another_users_predictions():
    from app.db.models import TranslationLog
    from app.db.session import SessionLocal

    token_one = _register_and_login(f"export-user-a-{uuid.uuid4().hex[:8]}")
    token_two = _register_and_login(f"export-user-b-{uuid.uuid4().hex[:8]}")

    user_one = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token_one}"}).json()["id"]

    db = SessionLocal()
    db.add(
        TranslationLog(
            user_id=user_one,
            predicted_text="SECRET",
            confidence=0.5,
            latency_ms=1.0,
            used_adapter=1,
        )
    )
    db.commit()
    db.close()

    export_resp = client.get("/api/v1/history/export.csv", headers={"Authorization": f"Bearer {token_two}"})
    assert export_resp.status_code == 200, export_resp.text
    assert "SECRET" not in export_resp.text


def test_history_date_range_filter_includes_new_letter_predictions():
    from app.db.models import TranslationLog
    from app.db.session import SessionLocal

    token = _register_and_login(f"range-user-{uuid.uuid4().hex[:8]}")
    user_id = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"}).json()["id"]

    db = SessionLocal()
    db.add(
        TranslationLog(
            user_id=user_id,
            predicted_text="Z",
            confidence=0.5,
            latency_ms=1.2,
            used_adapter=1,
        )
    )
    db.commit()
    db.close()

    for range_value in ("7d", "30d", "90d", "all"):
        response = client.get(
            f"/api/v1/history?range={range_value}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text
        assert any(item["predicted_text"] == "Z" for item in response.json()["items"])
