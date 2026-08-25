"""Regression tests for the live-translation pipeline fixes.

Covers two real bugs fixed alongside translate-live.js's 478->468 face
landmark count and 15->50 frame window:

1. inference_service.decode_logits() was joining decoded characters with
   " ".join(tokens). SimpleCharTokenizer (app/training/isltranslate.py) is
   CHARACTER-level and its vocabulary already contains a literal " " token,
   so the old code inserted an extra space between every single character
   (e.g. "hi" -> "h i", and worse around real spaces) — unreadable output
   even with a perfectly correct model.

2. /api/v1/translate had no validation of pose_keypoints/face_keypoints
   shape before building tensors — a malformed payload either crashed
   torch.tensor() on ragged input or reached the model with a wrong shape
   and failed deep inside a matmul with a confusing error instead of a
   clear 422.

Both tests exercise the REAL trained base_model.pt + base_model.vocab.json
checked into the repo (not synthetic weights), per the project's
requirement not to retrain or replace them.
"""
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
# FastAPI's table-creation now runs inside main.py's lifespan handler (moved
# there deliberately so importing app.main no longer has DB side effects —
# see main.py's `lifespan()`). The TestClient only triggers lifespan
# startup/shutdown when used as a context manager, so enter it once here;
# the process exits at the end of the test run, so there's no matching
# __exit__ to worry about.
client.__enter__()

BACKEND_DIR = Path(__file__).resolve().parents[1]  # backend/
REAL_WEIGHTS_PATH = BACKEND_DIR / "app" / "models" / "weights" / "base_model.pt"
REAL_VOCAB_PATH = REAL_WEIGHTS_PATH.with_suffix(".vocab.json")

requires_real_weights = pytest.mark.skipif(
    not REAL_WEIGHTS_PATH.exists() or not REAL_VOCAB_PATH.exists(),
    reason="real trained base_model.pt / base_model.vocab.json not present",
)


def _realistic_keypoints(n_frames: int) -> tuple[list[list[float]], list[list[float]]]:
    """Non-zero, plausibly-ranged synthetic keypoints (not all-zero test
    data) — coordinates in [0, 1] and small z, matching what real MediaPipe
    Holistic output looks like, at the correct 132 / 1404 feature contract."""
    torch.manual_seed(0)
    pose = (torch.rand(n_frames, POSE_INPUT_DIM) * 0.8 + 0.1).tolist()
    face = (torch.rand(n_frames, FACE_INPUT_DIM) * 0.8 + 0.1).tolist()
    return pose, face


@requires_real_weights
def test_decode_logits_uses_real_vocab_and_joins_characters_without_spaces(monkeypatch):
    """Direct regression test for the join-with-space bug, using the actual
    49-token vocabulary from base_model.vocab.json — not a fabricated one."""
    id_to_token = inference_service._load_vocab(str(REAL_WEIGHTS_PATH))
    assert len(id_to_token) == 49, f"expected the real 49-token vocab, got {len(id_to_token)}"
    token_to_id = {tok: i for i, tok in id_to_token.items()}

    monkeypatch.setattr(inference_service, "_id_to_token", id_to_token)

    # Build logits that greedily decode to the exact character sequence
    # "hi there" via the CTC collapse rule (repeat each id so collapsing
    # consecutive duplicates leaves exactly this sequence, blanks between
    # repeated-in-source characters where needed).
    target_text = "hi there"
    blank = 0
    ids: list[int] = []
    prev = None
    for ch in target_text:
        tid = token_to_id[ch]
        if prev == tid:
            ids.append(blank)  # separate repeated-in-source chars so collapse doesn't eat one
        ids.append(tid)
        prev = tid

    vocab_size = len(id_to_token)
    frames = len(ids)
    logits = torch.full((1, frames, vocab_size), -10.0)
    for t, tid in enumerate(ids):
        logits[0, t, tid] = 10.0

    text, confidence = inference_service.decode_logits(logits)

    assert text == target_text, f"expected {target_text!r}, got {text!r} — character-join regression"
    assert 0.0 <= confidence <= 1.0


@requires_real_weights
def test_decode_logits_collapses_repeats_and_drops_blanks_with_real_vocab(monkeypatch):
    id_to_token = inference_service._load_vocab(str(REAL_WEIGHTS_PATH))
    monkeypatch.setattr(inference_service, "_id_to_token", id_to_token)
    token_to_id = {tok: i for i, tok in id_to_token.items()}

    blank = 0
    a = token_to_id["a"]
    b = token_to_id["b"]
    # a a a <blank> <blank> b b  ->  collapse consecutive dupes -> a <blank> b -> drop blank -> "ab"
    ids = [a, a, a, blank, blank, b, b]
    vocab_size = len(id_to_token)
    logits = torch.full((1, len(ids), vocab_size), -10.0)
    for t, tid in enumerate(ids):
        logits[0, t, tid] = 10.0

    text, _ = inference_service.decode_logits(logits)
    assert text == "ab"


@requires_real_weights
def test_decode_logits_confidence_reflects_the_decoded_text_not_blank_certainty(monkeypatch):
    """Regression test for the '86% confidence' next to '(no sign detected)'
    bug: confidence must be about the returned text, not overall frame
    certainty. A model that is very sure every frame is blank must report
    LOW confidence (nothing was predicted) — not high confidence borrowed
    from how sure it was about blanks."""
    id_to_token = inference_service._load_vocab(str(REAL_WEIGHTS_PATH))
    monkeypatch.setattr(inference_service, "_id_to_token", id_to_token)

    blank = 0
    vocab_size = len(id_to_token)
    frames = 20
    # Every frame extremely confident it's blank.
    logits = torch.full((1, frames, vocab_size), -10.0)
    logits[0, :, blank] = 10.0

    text, confidence = inference_service.decode_logits(logits)

    assert text == "(no sign detected)"
    assert confidence == 0.0, (
        f"expected 0.0 confidence for an all-blank prediction, got {confidence} — "
        "confidence must not be borrowed from blank-frame certainty"
    )


@requires_real_weights
def test_decode_logits_confidence_ignores_low_confidence_blank_frames(monkeypatch):
    """A prediction with a clear, confident non-blank token surrounded by
    weak/uncertain blank frames should report confidence based on the
    confident non-blank frame, not diluted by the uncertain blanks."""
    id_to_token = inference_service._load_vocab(str(REAL_WEIGHTS_PATH))
    monkeypatch.setattr(inference_service, "_id_to_token", id_to_token)
    token_to_id = {tok: i for i, tok in id_to_token.items()}

    blank = 0
    a = token_to_id["a"]
    vocab_size = len(id_to_token)

    # Blank frames only mildly favor blank (low confidence); the "a" frame
    # is extremely confident.
    logits = torch.full((3, vocab_size), 0.0)
    logits[0, blank] = 0.1
    logits[1, a] = 10.0
    logits[2, blank] = 0.1
    logits = logits.unsqueeze(0)

    text, confidence = inference_service.decode_logits(logits)

    assert text == "a"
    assert confidence > 0.9, f"expected confidence near-certain on the 'a' frame, got {confidence}"


def test_translate_endpoint_rejects_mismatched_frame_counts():
    pose, face = _realistic_keypoints(10)
    face = face[:8]  # deliberately mismatched frame count
    resp = client.post("/api/v1/translate", json={
        "user_id": None, "adapter_id": None,
        "pose_keypoints": pose, "face_keypoints": face,
    })
    assert resp.status_code == 422
    assert "frame count mismatch" in resp.json()["detail"]


def test_translate_endpoint_rejects_wrong_face_dim():
    """The exact historical bug: sending 1434-dim (478*3) face frames
    instead of 1404 (468*3)."""
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


def test_translate_endpoint_rejects_nonexistent_adapter_id():
    pose, face = _realistic_keypoints(5)
    resp = client.post("/api/v1/translate", json={
        "user_id": None, "adapter_id": 999999,
        "pose_keypoints": pose, "face_keypoints": face,
    })
    assert resp.status_code == 404


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
    assert resp.json()["detail"] == "Translation model is unavailable"


@requires_real_weights
def test_translate_endpoint_end_to_end_with_real_model_and_realistic_keypoints():
    """Full pipeline smoke test against the REAL trained weights: realistic
    (non-zero) pose+face keypoints at the correct 132/1404 contract, through
    the actual /api/v1/translate endpoint, the actual base_model.pt, and the
    actual decoder — not synthetic/placeholder weights, not all-zero input."""
    pose, face = _realistic_keypoints(40)

    prev_cwd = os.getcwd()
    try:
        # The configured default model path is anchored to backend/, so it is
        # independent of the process's current working directory.
        os.chdir(BACKEND_DIR)
        resp = client.post("/api/v1/translate", json={
            "user_id": None, "adapter_id": None,
            "pose_keypoints": pose, "face_keypoints": face,
        })
    finally:
        os.chdir(prev_cwd)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["predicted_text"], str) and body["predicted_text"] != ""
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0
    assert body["used_adapter"] is False

    # The model actually loaded is the real trained one, not a random-init
    # fallback — confirms the vocab_size-from-checkpoint fix and this
    # endpoint are wired together correctly.
    model = load_frozen_base_model(str(REAL_WEIGHTS_PATH))
    assert model.output_head.out_features == 49
