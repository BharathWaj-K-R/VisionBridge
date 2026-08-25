"""Regression tests for live translation, shape validation, and adapter auth."""
import os
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.models.base_model import FACE_INPUT_DIM, POSE_INPUT_DIM, load_frozen_base_model
from app.services import inference_service
from app.services.inference_service import ModelUnavailableError

client = TestClient(app)
client.__enter__()

BACKEND_DIR = Path(__file__).resolve().parents[1]
REAL_WEIGHTS_PATH = BACKEND_DIR / "app" / "models" / "weights" / "base_model.pt"
REAL_VOCAB_PATH = REAL_WEIGHTS_PATH.with_suffix(".vocab.json")

requires_real_weights = pytest.mark.skipif(
    not REAL_WEIGHTS_PATH.exists() or not REAL_VOCAB_PATH.exists(),
    reason="real trained base_model.pt / base_model.vocab.json not present",
)


def _realistic_keypoints(n_frames: int) -> tuple[list[list[float]], list[list[float]]]:
    torch.manual_seed(0)
    pose = (torch.rand(n_frames, POSE_INPUT_DIM) * 0.8 + 0.1).tolist()
    face = (torch.rand(n_frames, FACE_INPUT_DIM) * 0.8 + 0.1).tolist()
    return pose, face


@requires_real_weights
def test_decode_logits_uses_real_vocab(monkeypatch):
    id_to_token = inference_service._load_vocab(str(REAL_WEIGHTS_PATH))
    assert len(id_to_token) == 49
    token_to_id = {tok: i for i, tok in id_to_token.items()}
    monkeypatch.setattr(inference_service, "_id_to_token", id_to_token)

    target_text = "hi there"
    ids = []
    previous = None
    for ch in target_text:
        token_id = token_to_id[ch]
        if token_id == previous:
            ids.append(0)
        ids.append(token_id)
        previous = token_id

    logits = torch.full((1, len(ids), len(id_to_token)), -10.0)
    for t, token_id in enumerate(ids):
        logits[0, t, token_id] = 10.0

    text, confidence = inference_service.decode_logits(logits)
    assert text == target_text
    assert 0.0 <= confidence <= 1.0


def test_decode_logits_rejects_vocab_mismatch():
    inference_service._id_to_token = {0: "<blank>", 1: "a"}
    with pytest.raises(ValueError, match="Decoder vocabulary/logit mismatch"):
        inference_service.decode_logits(torch.zeros(1, 2, 3))


def test_translate_endpoint_rejects_mismatched_frame_counts():
    pose, face = _realistic_keypoints(10)
    resp = client.post("/api/v1/translate", json={
        "user_id": None, "adapter_id": None,
        "pose_keypoints": pose, "face_keypoints": face[:8],
    })
    assert resp.status_code == 422
    assert "frame count mismatch" in resp.json()["detail"]


def test_translate_endpoint_rejects_wrong_face_dim():
    pose, _ = _realistic_keypoints(5)
    bad_face = (torch.rand(5, 478 * 3) * 0.8 + 0.1).tolist()
    resp = client.post("/api/v1/translate", json={
        "user_id": None, "adapter_id": None,
        "pose_keypoints": pose, "face_keypoints": bad_face,
    })
    assert resp.status_code == 422
    assert "1404" in resp.json()["detail"]


def test_translate_endpoint_rejects_empty_payload():
    resp = client.post("/api/v1/translate", json={
        "user_id": None, "adapter_id": None,
        "pose_keypoints": [], "face_keypoints": [],
    })
    assert resp.status_code == 422


def test_translate_endpoint_rejects_non_finite_keypoints():
    pose, face = _realistic_keypoints(1)
    pose[0][0] = float("nan")
    resp = client.post("/api/v1/translate", json={
        "user_id": None, "adapter_id": None,
        "pose_keypoints": pose, "face_keypoints": face,
    })
    assert resp.status_code == 422
    assert "non-finite" in resp.json()["detail"]


def test_translate_endpoint_requires_auth_for_adapter_access():
    pose, face = _realistic_keypoints(1)
    resp = client.post("/api/v1/translate", json={
        "user_id": None, "adapter_id": 1,
        "pose_keypoints": pose, "face_keypoints": face,
    })
    assert resp.status_code == 401


def test_calibration_endpoint_requires_authentication():
    pose = [[0.0] * POSE_INPUT_DIM]
    face = [[0.0] * FACE_INPUT_DIM]
    resp = client.post("/api/v1/calibration", json={
        "user_id": 1,
        "calibration_seconds": 1,
        "pose_keypoints": pose,
        "face_keypoints": face,
        "target_labels": [1],
    })
    assert resp.status_code == 401


def test_translate_endpoint_returns_503_when_the_model_is_unavailable(monkeypatch):
    pose, face = _realistic_keypoints(1)

    def unavailable_model(*_args, **_kwargs):
        raise ModelUnavailableError("checkpoint unavailable")

    monkeypatch.setattr("app.api.translate.run_inference", unavailable_model)
    resp = client.post("/api/v1/translate", json={
        "user_id": None, "adapter_id": None,
        "pose_keypoints": pose, "face_keypoints": face,
    })
    assert resp.status_code == 503


@requires_real_weights
def test_translate_endpoint_end_to_end_with_real_model_and_realistic_keypoints():
    pose, face = _realistic_keypoints(40)
    prev_cwd = os.getcwd()
    try:
        os.chdir(BACKEND_DIR)
        resp = client.post("/api/v1/translate", json={
            "user_id": None, "adapter_id": None,
            "pose_keypoints": pose, "face_keypoints": face,
        })
    finally:
        os.chdir(prev_cwd)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["predicted_text"], str) and body["predicted_text"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0
    assert body["used_adapter"] is False
    model = load_frozen_base_model(str(REAL_WEIGHTS_PATH))
    assert model.output_head.out_features == 49
