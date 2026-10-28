from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.models.letter_model import (
    MODEL_VERSION,
    PREPROCESSING_VERSION,
    VisionBridgeLetterBaseModel,
    VisionBridgeLetterBaseV4Model,
    load_checkpoint,
    load_model_checkpoint,
    load_v4_checkpoint,
    save_checkpoint,
    save_v4_checkpoint,
)
from app.services.letter_fewshot import (
    normalize_hand_pair,
    normalize_hand_pair_v2,
    validate_prototype_adapter_payload,
    validate_prototype_adapter_payload_v4,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = BACKEND_ROOT / "tests" / "fixtures" / "v4_feature_parity.json"


def test_python_v4_features_match_shared_golden_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["contract"] == "two-hand-geometry-v2"
    assert fixture["input_dim"] == 126
    assert fixture["output_dim"] == 141
    assert len(fixture["frames"]) >= 4

    for frame in fixture["frames"]:
        actual = normalize_hand_pair_v2(
            frame["raw"],
            frame["left_present"],
            frame["right_present"],
        )
        expected = np.asarray(frame["expected"], dtype=np.float32)
        assert actual.dtype == np.float32
        assert actual.shape == (141,)
        np.testing.assert_allclose(actual, expected, atol=fixture["atol"], rtol=0)

        if not frame["left_present"]:
            assert np.count_nonzero(actual[:64]) == 0
        if not frame["right_present"]:
            assert np.count_nonzero(actual[64:128]) == 0
        if not frame["left_present"] or not frame["right_present"]:
            assert np.count_nonzero(actual[128:]) == 0


def test_v4_features_reject_invalid_raw_input_and_presence_flags() -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_hand_pair_v2([0.0] * 125 + [float("nan")], True, True)
    with pytest.raises(ValueError, match="booleans"):
        normalize_hand_pair_v2([0.0] * 126, 1, False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="126 raw hand coordinates"):
        normalize_hand_pair_v2([0.0] * 125, True, False)


def test_v4_features_leave_v1_normalization_available() -> None:
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["frames"][0]["raw"]
    v1 = normalize_hand_pair(raw)
    v2 = normalize_hand_pair_v2(raw, True, True)
    assert v1.shape == (126,)
    assert v2.shape == (141,)
    assert np.allclose(v1[:63], v2[:63], atol=1e-6)
    assert np.allclose(v1[63:], v2[64:127], atol=1e-6)


@pytest.mark.parametrize(
    ("variant", "input_dim"),
    [
        ("full_v4", 141),
        ("v4_features_old_mlp", 141),
        ("v1_features_shared_encoder", 126),
    ],
)
def test_v4_variants_save_load_and_produce_expected_shapes(
    tmp_path: Path,
    variant: str,
    input_dim: int,
) -> None:
    model = VisionBridgeLetterBaseV4Model(variant=variant)
    inputs = torch.zeros((3, input_dim), dtype=torch.float32)
    assert model.embed(inputs).shape == (3, 64)
    assert model(inputs).shape == (3, 26)
    expected_parameters = {
        "full_v4": 25_156,
        "v4_features_old_mlp": 28_532,
        "v1_features_shared_encoder": 25_156,
    }
    assert sum(p.numel() for p in model.parameters()) == expected_parameters[variant]

    checkpoint = tmp_path / f"{variant}.pt"
    save_v4_checkpoint(model, checkpoint)
    loaded = load_v4_checkpoint(checkpoint, expected_variant=variant)
    assert loaded.architecture_variant == variant
    assert loaded.input_dim == input_dim
    assert loaded.embed(inputs).shape == (3, 64)
    assert isinstance(load_model_checkpoint(checkpoint), VisionBridgeLetterBaseV4Model)

    with pytest.raises(ValueError, match="Unsupported letter base-model version"):
        load_checkpoint(checkpoint)


def test_v3_and_v4_loaders_reject_each_others_checkpoints(tmp_path: Path) -> None:
    v3_path = tmp_path / "v3.pt"
    v4_path = tmp_path / "v4.pt"
    save_checkpoint(VisionBridgeLetterBaseModel(), v3_path)
    save_v4_checkpoint(VisionBridgeLetterBaseV4Model(), v4_path)

    assert isinstance(load_checkpoint(v3_path), VisionBridgeLetterBaseModel)
    with pytest.raises(ValueError, match="Unsupported V4 letter base-model version"):
        load_v4_checkpoint(v3_path)
    with pytest.raises(ValueError, match="Unsupported letter base-model version"):
        load_checkpoint(v4_path)


def test_v4_loader_rejects_preprocessing_variant_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v4_bad_metadata.pt"
    save_v4_checkpoint(VisionBridgeLetterBaseV4Model(variant="full_v4"), checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["preprocessing_version"] = "two-hand-wrist-scale-v1"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="preprocessing metadata"):
        load_v4_checkpoint(checkpoint)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_adapter_binding_rejects_cross_version_and_hash_mismatch(tmp_path: Path) -> None:
    v3_path = tmp_path / "v3.pt"
    v4_path = tmp_path / "v4.pt"
    save_checkpoint(VisionBridgeLetterBaseModel(), v3_path)
    v4_model = VisionBridgeLetterBaseV4Model()
    save_v4_checkpoint(v4_model, v4_path)
    prototype = [0.0] * 64
    good_v4_payload = {
        "version": 5,
        "method": "dynamic-base-embedding-prototype",
        "base_model_version": "visionbridge-letter-base-v4",
        "base_model_sha256": _sha256(v4_path),
        "architecture_variant": "full_v4",
        "preprocessing_version": "two-hand-geometry-v2",
        "landmark_runtime": "mediapipe-hand-landmarker-0.10.35",
        "feature_dim": 141,
        "embedding_dim": 64,
        "base_model_labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "prototypes": {"A": prototype, "B": prototype},
    }
    assert validate_prototype_adapter_payload_v4(good_v4_payload, v4_path) == good_v4_payload

    stale_hash = {**good_v4_payload, "base_model_sha256": "0" * 64}
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        validate_prototype_adapter_payload_v4(stale_hash, v4_path)

    v3_payload = {
        "version": 4,
        "method": "dynamic-base-embedding-prototype",
        "base_model_version": MODEL_VERSION,
        "base_model_sha256": _sha256(v3_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "landmark_runtime": "mediapipe-hand-landmarker-0.10.35",
        "feature_dim": 126,
        "embedding_dim": 64,
        "base_model_labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "prototypes": {"A": prototype, "B": prototype},
    }
    with pytest.raises(ValueError, match="V3 adapters require recalibration"):
        validate_prototype_adapter_payload_v4(v3_payload, v4_path)
    with pytest.raises(ValueError, match="Invalid VisionBridge letter adapter"):
        validate_prototype_adapter_payload(good_v4_payload, v3_path)
