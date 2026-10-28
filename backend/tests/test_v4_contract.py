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
from app.training.letter_base import train_model
from scripts.run_v4_comparison import verify_aligned_splits
import scripts.prepare_v4_feature_dataset as v4_preparer
import scripts.run_v4_comparison as comparison

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


def test_v4_loader_rejects_missing_architecture_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v4_missing_metadata.pt"
    save_v4_checkpoint(VisionBridgeLetterBaseV4Model(), checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload.pop("hand_embedding_dim")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="embedding metadata is missing"):
        load_v4_checkpoint(checkpoint)


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



def test_v4_training_runs_without_applying_v1_only_augmentation(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    labels = np.arange(26, dtype=np.int64)
    root = tmp_path / "data"
    root.mkdir()
    (root / "labels.json").write_text(
        json.dumps({"labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}),
        encoding="utf-8",
    )
    for split in ("train", "val"):
        features = rng.normal(0.0, 0.05, size=(26, 141)).astype(np.float32)
        features[:, 63] = 1.0
        features[:, 127] = 1.0
        np.savez_compressed(root / f"{split}.npz", x=features, y=labels)

    report = train_model(
        root=root,
        output_path=tmp_path / "v4-one-epoch.pt",
        epochs=1,
        batch_size=26,
        learning_rate=1e-3,
        weight_decay=1e-4,
        target_class_accuracy=1.0,
        seed=42,
        hidden_dim=128,
        embedding_dim=64,
        dropout=0.10,
        loss_name="cross_entropy",
        landmark_jitter_std=0.0,
        rotation_degrees=0.0,
        validation_only=True,
        model_version="v4",
        v4_variant="full_v4",
    )
    assert report["model_version"] == "visionbridge-letter-base-v4"
    assert report["input_dim"] == 141
    assert report["parameter_count"] == 25_156
    assert (tmp_path / "v4-one-epoch.pt").is_file()


def test_comparison_runner_verifies_split_identity_and_label_order(tmp_path: Path) -> None:
    v1_root = tmp_path / "v1"
    v4_root = tmp_path / "v4"
    v1_root.mkdir()
    v4_root.mkdir()

    rows = []
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    for i, label in enumerate("ABC"):
        rows.append({
            "sample_id": f"sample-{i}",
            "source_path": f"images/{label.lower()}.png",
            "image_sha256": f"{i + 1:064x}",
            "label": label,
            "landmark_status": "ok",
        })

    for split in ("train", "val", "test"):
        x1 = np.zeros((3, 126), dtype=np.float32)
        x4 = np.zeros((3, 141), dtype=np.float32)
        np.savez_compressed(v1_root / f"{split}.npz", x=x1, y=labels)
        np.savez_compressed(v4_root / f"{split}.npz", x=x4, y=labels)
        for root in (v1_root, v4_root):
            with (root / f"{split}_manifest.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

    result = verify_aligned_splits(v1_root, v4_root)
    assert result["train"]["samples"] == 3
    assert result["val"]["samples"] == 3
    assert result["test"]["samples"] == 3

    wrong_rows = [dict(row) for row in rows]
    wrong_rows[1]["sample_id"] = "different-sample"
    with (v4_root / "test_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in wrong_rows:
            handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="sample identities/order differ"):
        verify_aligned_splits(v1_root, v4_root)



def test_fixed_epoch_budget_uses_all_epochs_even_after_target_is_met(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.training.letter_base as trainer

    rng = np.random.default_rng(21)
    root = tmp_path / "data"
    root.mkdir()
    (root / "labels.json").write_text(
        json.dumps({"labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}),
        encoding="utf-8",
    )
    for split in ("train", "val"):
        features = rng.normal(0.0, 0.05, size=(26, 141)).astype(np.float32)
        features[:, 63] = 1.0
        features[:, 127] = 1.0
        np.savez_compressed(root / f"{split}.npz", x=features, y=np.arange(26, dtype=np.int64))

    all_letters = {letter: 1.0 for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    counts = {letter: 1 for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    monkeypatch.setattr(trainer, "class_accuracy", lambda model, loader, device: (1.0, all_letters, counts))
    report = trainer.train_model(
        root=root,
        output_path=tmp_path / "budget.pt",
        epochs=3,
        batch_size=26,
        learning_rate=1e-3,
        weight_decay=1e-4,
        target_class_accuracy=1.0,
        seed=42,
        hidden_dim=128,
        embedding_dim=64,
        dropout=0.10,
        loss_name="cross_entropy",
        validation_only=True,
        model_version="v4",
        v4_variant="full_v4",
        fixed_epoch_budget=True,
    )
    assert report["reached_target"] is True
    assert report["epochs_completed"] == 3
    assert report["best_epoch"] == 1
    assert report["training"]["fixed_epoch_budget"] is True



def test_v4_dataset_preparation_requires_v1_contract_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v1_root = tmp_path / "v1"
    v1_root.mkdir()
    (v1_root / "labels.json").write_text(
        json.dumps({"labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "preprocessing_version": "wrong"}),
        encoding="utf-8",
    )
    called = False

    def must_not_create_landmarker(path: Path):
        nonlocal called
        called = True
        raise AssertionError("landmarker should not be initialized for incompatible V1 data")

    monkeypatch.setattr(v4_preparer, "create_hand_landmarker", must_not_create_landmarker)
    with pytest.raises(ValueError, match="validated V1 data directory"):
        v4_preparer.prepare_v4_dataset(
            input_root=tmp_path / "raw",
            v1_data_dir=v1_root,
            output_dir=tmp_path / "v4",
        )
    assert called is False


def test_v4_dataset_preparation_rebuilds_all_splits_and_preserves_row_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["frames"][0]["expected"]
    raw_root = tmp_path / "raw"
    v1_root = tmp_path / "v1"
    output_root = tmp_path / "v4"
    raw_root.mkdir()
    v1_root.mkdir()
    v1_metadata = {
        "labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "preprocessing_version": "two-hand-wrist-scale-v1",
        "landmarker": {"runtime": "mediapipe-hand-landmarker-0.10.35"},
    }
    (v1_root / "labels.json").write_text(json.dumps(v1_metadata), encoding="utf-8")
    stale = output_root / "test.npz"
    output_root.mkdir()
    np.savez_compressed(stale, x=np.zeros((1, 141), dtype=np.float32), y=np.asarray([0]))

    class FakeLandmarker:
        closed = False

        def close(self) -> None:
            self.closed = True

    fake_landmarker = FakeLandmarker()
    monkeypatch.setattr(v4_preparer, "create_hand_landmarker", lambda path: fake_landmarker)
    monkeypatch.setattr(
        v4_preparer,
        "_extract_v4_vector",
        lambda image_path, landmarker: (np.asarray(fixture, dtype=np.float32), None, None),
    )

    rows_by_split = {}
    for split in ("train", "val", "test"):
        relpath = f"{split}/A.png"
        image = raw_root / relpath
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes((split + "-image").encode("utf-8"))
        import hashlib
        row = {
            "sample_id": f"{split}-sample-1",
            "source_path": relpath,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "label": "A",
            "letter": "A",
            "landmark_status": "ok",
            "signer_id": None,
            "recording_id": None,
            "split": split,
        }
        rows_by_split[split] = row
        (v1_root / f"{split}_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        np.savez_compressed(
            v1_root / f"{split}.npz",
            x=np.zeros((1, 126), dtype=np.float32),
            y=np.asarray([0], dtype=np.int64),
        )

    report = v4_preparer.prepare_v4_dataset(
        input_root=raw_root,
        v1_data_dir=v1_root,
        output_dir=output_root,
        hand_model_path=tmp_path / "fake.task",
    )
    assert fake_landmarker.closed is True
    assert report["input_dim"] == 141
    for split, row in rows_by_split.items():
        with np.load(output_root / f"{split}.npz", allow_pickle=False) as data:
            assert data["x"].shape == (1, 141)
            assert data["y"].tolist() == [0]
        output_row = json.loads((output_root / f"{split}_manifest.jsonl").read_text(encoding="utf-8"))
        assert output_row["sample_id"] == row["sample_id"]
        assert output_row["image_sha256"] == row["image_sha256"]



def test_comparison_runner_freezes_selection_then_evaluates_test_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_root = tmp_path / "v1"
    v4_root = tmp_path / "v4"
    output_root = tmp_path / "run"
    v1_root.mkdir()
    v4_root.mkdir()

    labels = np.arange(26, dtype=np.int64)
    manifest_rows = []
    for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        manifest_rows.append({
            "sample_id": f"sample-{index}",
            "source_path": f"images/{letter.lower()}.png",
            "image_sha256": f"{index + 1:064x}",
            "label": letter,
            "letter": letter,
            "landmark_status": "ok",
        })
    for split in ("train", "val", "test"):
        np.savez_compressed(v1_root / f"{split}.npz", x=np.zeros((26, 126), dtype=np.float32), y=labels)
        np.savez_compressed(v4_root / f"{split}.npz", x=np.zeros((26, 141), dtype=np.float32), y=labels)
        for root in (v1_root, v4_root):
            (root / f"{split}_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows),
                encoding="utf-8",
            )

    train_calls = []
    test_calls = []
    variant_accuracy = {
        "v3_mlp": (0.50, 0.50, 0.30),
        "v4_features_old_mlp": (0.60, 0.60, 0.40),
        "v1_features_shared_encoder": (0.65, 0.65, 0.45),
        "full_v4": (0.80, 0.80, 0.60),
    }

    def fake_train_model(**kwargs):
        train_calls.append(kwargs)
        output = kwargs["output_path"]
        if kwargs["model_version"] == "v3":
            model = VisionBridgeLetterBaseModel()
            save_checkpoint(model, output)
            variant_name = "v3_mlp"
        else:
            model = VisionBridgeLetterBaseV4Model(variant=kwargs["v4_variant"])
            save_v4_checkpoint(model, output)
            variant_name = kwargs["v4_variant"]
        overall, macro, worst = variant_accuracy[variant_name]
        return {
            "best_epoch": 2,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "validation_accuracy": overall,
            "validation_macro_accuracy": macro,
            "worst_validation_accuracy": worst,
            "validation_per_letter": {letter: worst for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        }

    def fake_latency(checkpoint, data_dir, batch_size):
        return {
            "device": "cpu",
            "batch_size": min(batch_size, 26),
            "warmup_runs": 5,
            "measured_runs": 20,
            "mean_batch_ms": 0.2,
            "mean_sample_ms": 0.2 / min(batch_size, 26),
        }

    def fake_evaluate(*, checkpoint, data_dir, split, batch_size):
        test_calls.append({"checkpoint": checkpoint, "data_dir": data_dir, "split": split})
        frozen_manifest = output_root / "selection_frozen.json"
        assert frozen_manifest.is_file(), "selection must be frozen before touching test"
        frozen = json.loads(frozen_manifest.read_text(encoding="utf-8"))
        assert frozen["selected_variant"] == "full_v4"
        assert frozen["checkpoint_sha256"] == comparison.sha256_file(checkpoint)
        assert frozen["test_evaluation_id"]
        assert split == "test"
        assert data_dir == v4_root
        return {
            "checkpoint_sha256": comparison.sha256_file(checkpoint),
            "overall_accuracy": 0.77,
            "macro_accuracy": 0.75,
            "samples": 26,
            "worst_letter": "W",
            "worst_letter_accuracy": 0.50,
            "device": "cpu",
        }

    monkeypatch.setattr(comparison, "train_model", fake_train_model)
    monkeypatch.setattr(comparison, "measure_python_latency", fake_latency)
    monkeypatch.setattr(comparison, "evaluate", fake_evaluate)
    report = comparison.run_comparison(
        v1_dir=v1_root,
        v4_dir=v4_root,
        output_dir=output_root,
        seeds=(7, 11, 19),
        epochs=3,
        batch_size=26,
        learning_rate=1e-3,
        weight_decay=1e-4,
        markdown_output=tmp_path / "comparison.md",
    )
    assert len(train_calls) == 12
    assert all(call["validation_only"] is True for call in train_calls)
    assert all(call["fixed_epoch_budget"] is True for call in train_calls)
    assert {call["seed"] for call in train_calls} == {7, 11, 19}
    assert len(test_calls) == 1
    assert report["selection"]["variant"] == "full_v4"
    assert report["protocol"]["test_accuracy_evaluation_calls"] == 1
    assert report["final_test"]["evaluation_run"]["test_accuracy_evaluation_calls"] == 1
    assert (output_root / "selection_frozen.json").is_file()
    assert (output_root / "v4_comparison.json").is_file()
