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


def test_export_csv_contains_the_users_own_translation(monkeypatch):
    def fake_inference(*_args, **_kwargs):
        return {"predicted_text": "hello world", "confidence": 0.75, "latency_ms": 12.0, "used_adapter": False}

    monkeypatch.setattr("app.api.translate.run_inference", fake_inference)

    username = f"export-user-{uuid.uuid4().hex[:8]}"
    token = _register_and_login(username)
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "user_id": None, "adapter_id": None,
        "pose_keypoints": [[0.0] * 132],
        "face_keypoints": [[0.0] * 1404],
        "left_hand_keypoints": [[0.0] * 63],
        "right_hand_keypoints": [[0.0] * 63],
    }
    translate_resp = client.post("/api/v1/translate", headers=headers, json=payload)
    assert translate_resp.status_code == 200, translate_resp.text

    export_resp = client.get("/api/v1/history/export.csv", headers=headers)
    assert export_resp.status_code == 200, export_resp.text
    assert "text/csv" in export_resp.headers["content-type"]

    rows = list(csv.reader(io.StringIO(export_resp.text)))
    assert rows[0] == ["id", "created_at", "predicted_text", "confidence", "latency_ms", "used_adapter"]
    assert any(row[2] == "hello world" for row in rows[1:]), rows


def test_export_csv_does_not_leak_another_users_translations(monkeypatch):
    def fake_inference(*_args, **_kwargs):
        return {"predicted_text": "user one only", "confidence": 0.5, "latency_ms": 5.0, "used_adapter": False}

    monkeypatch.setattr("app.api.translate.run_inference", fake_inference)

    user_one = f"export-user-a-{uuid.uuid4().hex[:8]}"
    user_two = f"export-user-b-{uuid.uuid4().hex[:8]}"
    token_one = _register_and_login(user_one)
    token_two = _register_and_login(user_two)

    payload = {
        "user_id": None, "adapter_id": None,
        "pose_keypoints": [[0.0] * 132],
        "face_keypoints": [[0.0] * 1404],
        "left_hand_keypoints": [[0.0] * 63],
        "right_hand_keypoints": [[0.0] * 63],
    }
    translate_resp = client.post(
        "/api/v1/translate", headers={"Authorization": f"Bearer {token_one}"}, json=payload
    )
    assert translate_resp.status_code == 200, translate_resp.text

    export_resp = client.get("/api/v1/history/export.csv", headers={"Authorization": f"Bearer {token_two}"})
    assert export_resp.status_code == 200, export_resp.text
    assert "user one only" not in export_resp.text


def test_history_date_range_filter_includes_just_created_rows(monkeypatch):
    """Regression test for the timezone-aware datetime fix in
    db/models.py (created_at defaults) and history.py's _start_for_range():
    both sides of the range comparison must stay consistent, or a row
    created seconds ago could be silently excluded from range=7d/30d/90d
    (or the reverse — everything silently matching regardless of range)."""

    def fake_inference(*_args, **_kwargs):
        return {"predicted_text": "just now", "confidence": 0.5, "latency_ms": 5.0, "used_adapter": False}

    monkeypatch.setattr("app.api.translate.run_inference", fake_inference)

    username = f"range-user-{uuid.uuid4().hex[:8]}"
    token = _register_and_login(username)
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "user_id": None, "adapter_id": None,
        "pose_keypoints": [[0.0] * 132],
        "face_keypoints": [[0.0] * 1404],
        "left_hand_keypoints": [[0.0] * 63],
        "right_hand_keypoints": [[0.0] * 63],
    }
    translate_resp = client.post("/api/v1/translate", headers=headers, json=payload)
    assert translate_resp.status_code == 200, translate_resp.text

    for range_value in ("7d", "30d", "90d", "all"):
        resp = client.get(f"/api/v1/history?range={range_value}", headers=headers)
        assert resp.status_code == 200, resp.text
        items = resp.json()["items"]
        assert any(item["predicted_text"] == "just now" for item in items), (
            f"a row created seconds ago should always appear under range={range_value}"
        )
