from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from app.models.letter_model import (
    LANDMARK_RUNTIME,
    MODEL_VERSION,
    PREPROCESSING_VERSION,
    VisionBridgeLetterBaseModel,
    load_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = BACKEND_ROOT / "app" / "models" / "weights" / "letter_base_model.pt"
EXPECTED_SHA256 = "f903872e96f269c5fbee6ea897f0027b91e162f2fc45b4c9a115e8ba8e38c917"
EXPECTED_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v3_checkpoint_and_contract_are_frozen() -> None:
    assert CHECKPOINT.stat().st_size == 111_429
    assert _sha256(CHECKPOINT) == EXPECTED_SHA256

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert payload["model_version"] == MODEL_VERSION == "visionbridge-letter-base-v3"
    assert payload["preprocessing_version"] == PREPROCESSING_VERSION == "two-hand-wrist-scale-v1"
    assert payload["landmark_runtime"] == LANDMARK_RUNTIME == "mediapipe-hand-landmarker-0.10.35"
    assert payload["input_dim"] == 126
    assert payload["hidden_dim"] == 128
    assert payload["embedding_dim"] == 64
    assert payload["num_classes"] == 26
    assert payload["labels"] == EXPECTED_LABELS

    model = load_checkpoint(CHECKPOINT)
    assert isinstance(model, VisionBridgeLetterBaseModel)
    assert sum(parameter.numel() for parameter in model.parameters()) == 26_582

    with torch.inference_mode():
        inputs = torch.zeros((2, 126), dtype=torch.float32)
        assert model.embed(inputs).shape == (2, 64)
        assert model(inputs).shape == (2, 26)


def test_frozen_reports_are_copies_bound_to_the_checkpoint() -> None:
    evaluation_source = REPO_ROOT / "visionbridge_letter_evaluation.json"
    evidence_source = REPO_ROOT / "visionbridge_v3_release_evidence.json"
    baseline_dir = REPO_ROOT / "docs" / "baseline" / "v3"
    evaluation_copy = baseline_dir / "visionbridge_letter_evaluation.json"
    evidence_copy = baseline_dir / "visionbridge_v3_release_evidence.json"

    report = json.loads(evaluation_source.read_text(encoding="utf-8"))
    assert report["checkpoint_sha256"] == EXPECTED_SHA256
    assert evaluation_copy.read_bytes() == evaluation_source.read_bytes()
    assert evidence_copy.read_bytes() == evidence_source.read_bytes()

    copied_evidence = json.loads(evidence_copy.read_text(encoding="utf-8"))
    assert copied_evidence["checkpoint"]["sha256"] == EXPECTED_SHA256
    assert copied_evidence["checkpoint"]["bytes"] == 111_429


def test_v3_golden_landmarks_match_frozen_logits_and_embeddings() -> None:
    fixture_path = BACKEND_ROOT / "tests" / "fixtures" / "v3_golden_landmarks.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["checkpoint_sha256"] == EXPECTED_SHA256
    assert fixture["model_version"] == MODEL_VERSION
    assert fixture["preprocessing_version"] == PREPROCESSING_VERSION
    assert fixture["landmark_runtime"] == LANDMARK_RUNTIME
    assert fixture["input_dim"] == 126
    assert fixture["embedding_dim"] == 64
    assert fixture["num_classes"] == 26
    assert fixture["labels"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    frames = fixture["frames"]
    assert len(frames) >= 32
    assert {frame["kind"] for frame in frames} >= {"left_only", "right_only", "both", "near_empty"}

    model = load_checkpoint(CHECKPOINT)
    model.eval()
    with torch.inference_mode():
        for frame in frames:
            raw_values = np.asarray(frame["input"], dtype=np.float32)
            assert raw_values.shape == (126,), frame["id"]
            if frame["kind"] == "left_only":
                assert np.count_nonzero(raw_values[63:]) == 0, frame["id"]
            elif frame["kind"] == "right_only":
                assert np.count_nonzero(raw_values[:63]) == 0, frame["id"]
            elif frame["kind"] == "near_empty":
                assert np.count_nonzero(raw_values) <= 2, frame["id"]

            raw = torch.from_numpy(raw_values).unsqueeze(0)
            actual_embedding = model.embed(raw)[0].cpu().numpy()
            actual_logits = model(raw)[0].cpu().numpy()
            expected_embedding = np.asarray(frame["embedding"], dtype=np.float32)
            expected_logits = np.asarray(frame["logits"], dtype=np.float32)

            assert expected_embedding.shape == (64,), frame["id"]
            assert expected_logits.shape == (26,), frame["id"]
            np.testing.assert_allclose(
                actual_embedding,
                expected_embedding,
                rtol=1e-5,
                atol=fixture["output_atol"],
                err_msg=f"V3 embedding drift for {frame['id']}",
            )
            np.testing.assert_allclose(
                actual_logits,
                expected_logits,
                rtol=1e-5,
                atol=fixture["output_atol"],
                err_msg=f"V3 logits drift for {frame['id']}",
            )
