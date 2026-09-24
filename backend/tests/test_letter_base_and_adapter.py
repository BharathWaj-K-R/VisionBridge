import numpy as np
import pytest
import torch

from app.models.letter_model import LANDMARK_RUNTIME, PREPROCESSING_VERSION, VisionBridgeLetterBaseModel, build_browser_payload, build_checkpoint, save_checkpoint, load_checkpoint
from app.schemas.schemas import LetterCalibrationRequest
from scripts.migrate_v3_checkpoint import migrate_checkpoint
from app.services import letter_fewshot
from scripts import prepare_letter_dataset


def _pair(seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(0, 0.02, (21, 3)).astype(np.float32)
    points[0] = 0
    return points.reshape(-1).tolist() + points.reshape(-1).tolist()


def test_checkpoint_records_preprocessing_contract():
    model = VisionBridgeLetterBaseModel()
    payload = build_checkpoint(model)
    assert payload["preprocessing_version"] == PREPROCESSING_VERSION
    assert payload["landmark_runtime"] == LANDMARK_RUNTIME


def test_train_validation_split_keeps_exact_duplicates_together():
    features = np.arange(26 * 4 * 126, dtype=np.float32).reshape(26 * 4, 126)
    labels = np.repeat(np.arange(26), 4)
    samples = []
    for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        samples.extend(
            [
                {"source_path": f"{letter}/one-a.jpg", "source_split": "training", "letter": letter, "image_sha256": f"{index}-a"},
                {"source_path": f"{letter}/one-b.jpg", "source_split": "training", "letter": letter, "image_sha256": f"{index}-a"},
                {"source_path": f"{letter}/two-a.jpg", "source_split": "validation", "letter": letter, "image_sha256": f"{index}-b"},
                {"source_path": f"{letter}/two-b.jpg", "source_split": "validation", "letter": letter, "image_sha256": f"{index}-b"},
            ]
        )

    train_x, train_y, train_samples, val_x, val_y, val_samples = (
        prepare_letter_dataset.make_train_validation_split(
            features,
            labels,
            samples,
            validation_ratio=0.5,
            seed=42,
        )
    )

    train_hashes = {item["image_sha256"] for item in train_samples}
    val_hashes = {item["image_sha256"] for item in val_samples}

    assert train_x.shape[1] == 126
    assert val_x.shape[1] == 126
    assert set(train_y.tolist()) == set(range(26))
    assert set(val_y.tolist()) == set(range(26))
    assert train_hashes.isdisjoint(val_hashes)


def test_base_model_contract():
    model = VisionBridgeLetterBaseModel()
    x = torch.randn(4, 126)
    logits = model(x)
    emb = model.embed(x)
    assert logits.shape == (4, 26)
    assert emb.shape == (4, 64)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(emb).all()


def test_base_model_scales_from_config():
    model = VisionBridgeLetterBaseModel(
        input_dim=126,
        hidden_dim=256,
        embedding_dim=96,
        labels=("A", "B", "C", "D", "E", "F"),
    )
    x = torch.randn(3, 126)
    assert model(x).shape == (3, 6)
    assert model.embed(x).shape == (3, 96)


def test_checkpoint_round_trip_preserves_dynamic_width_configuration(tmp_path):
    path = tmp_path / "base.pt"
    model = VisionBridgeLetterBaseModel(
        hidden_dim=192,
        embedding_dim=80,
        labels=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        dropout=0.2,
    )
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)

    assert loaded.input_dim == 126
    assert loaded.hidden_dim == 192
    assert loaded.embedding_dim == 80
    assert loaded.labels == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert loaded.output_head.out_features == 26
    assert loaded.dropout == pytest.approx(0.2)


def test_few_shot_adapter_tracks_current_base_version(tmp_path, monkeypatch):
    model = VisionBridgeLetterBaseModel(embedding_dim=80)
    base = tmp_path / "base.pt"
    save_checkpoint(model, base)
    monkeypatch.setattr(letter_fewshot.settings, "LETTER_BASE_MODEL_PATH", str(base))
    monkeypatch.setattr(letter_fewshot.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path / "adapters"))

    fitted = letter_fewshot.fit_prototype_adapter(
        model,
        [("A", _pair(1)), ("A", _pair(1)), ("A", _pair(1)), ("B", _pair(2)), ("B", _pair(2)), ("B", _pair(2))],
    )
    adapter_path = letter_fewshot.save_prototype_adapter(fitted["payload"])
    loaded = letter_fewshot.load_prototype_adapter(adapter_path, base)
    pred, conf, scores = letter_fewshot.predict_letter(model, loaded, _pair(1))

    assert loaded["method"] == "dynamic-base-embedding-prototype"
    assert loaded["preprocessing_version"] == PREPROCESSING_VERSION
    assert loaded["landmark_runtime"] == LANDMARK_RUNTIME
    assert loaded["embedding_dim"] == 80
    assert loaded["base_model_version"] != ""
    assert pred == "A"
    assert 0 < conf <= 1
    assert scores[0][0] == "A"


def test_adapter_rejects_after_base_change(tmp_path, monkeypatch):
    model = VisionBridgeLetterBaseModel(embedding_dim=80)
    base = tmp_path / "base.pt"
    changed = tmp_path / "changed.pt"
    save_checkpoint(model, base)
    save_checkpoint(
        VisionBridgeLetterBaseModel(hidden_dim=256, embedding_dim=96),
        changed,
    )

    monkeypatch.setattr(
        letter_fewshot.settings,
        "LETTER_BASE_MODEL_PATH",
        str(base),
    )
    monkeypatch.setattr(
        letter_fewshot.settings,
        "ADAPTER_WEIGHTS_DIR",
        str(tmp_path / "adapters"),
    )

    fitted = letter_fewshot.fit_prototype_adapter(
        model,
        [("A", _pair(1)), ("A", _pair(2)), ("A", _pair(3)), ("B", _pair(4)), ("B", _pair(5)), ("B", _pair(6))],
    )
    path = letter_fewshot.save_prototype_adapter(fitted["payload"])

    with pytest.raises(ValueError, match="requires recalibration"):
        letter_fewshot.load_prototype_adapter(path, changed)

def test_browser_payload_matches_checkpoint(tmp_path):
    path = tmp_path / "base.pt"
    model = VisionBridgeLetterBaseModel(hidden_dim=160, embedding_dim=72, labels=("A", "B", "C"))
    save_checkpoint(model, path)

    payload = build_browser_payload(model, "sha")
    assert payload["hidden_dim"] == 160
    assert payload["embedding_dim"] == 72
    assert payload["num_classes"] == 3
    assert len(payload["layers"]["hidden"]["weight"]) == 160
    assert len(payload["layers"]["head"]["weight"]) == 3


def test_degenerate_input_rejected():
    with pytest.raises(ValueError, match="No visible"):
        letter_fewshot.normalize_hand_pair([0.0] * 126)


def test_adapter_rejects_incompatible_preprocessing_metadata(tmp_path, monkeypatch):
    model = VisionBridgeLetterBaseModel()
    base = tmp_path / "base.pt"
    save_checkpoint(model, base)

    monkeypatch.setattr(
        letter_fewshot.settings,
        "LETTER_BASE_MODEL_PATH",
        str(base),
    )
    monkeypatch.setattr(
        letter_fewshot.settings,
        "ADAPTER_WEIGHTS_DIR",
        str(tmp_path / "adapters"),
    )

    fitted = letter_fewshot.fit_prototype_adapter(
        model,
        [("A", _pair(1)), ("A", _pair(2)), ("A", _pair(3)), ("B", _pair(4)), ("B", _pair(5)), ("B", _pair(6))],
    )
    fitted["payload"]["preprocessing_version"] = "wrong-contract"
    path = letter_fewshot.save_prototype_adapter(fitted["payload"])

    with pytest.raises(ValueError, match="preprocessing"):
        letter_fewshot.load_prototype_adapter(path, base)


@pytest.mark.parametrize("field", ["preprocessing_version", "landmark_runtime"])
def test_checkpoint_rejects_missing_runtime_contract_metadata(tmp_path, field):
    path = tmp_path / "base.pt"
    save_checkpoint(VisionBridgeLetterBaseModel(), path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.pop(field)
    torch.save(payload, path)

    with pytest.raises(ValueError, match="Unsupported letter base-model"):
        load_checkpoint(path)


@pytest.mark.parametrize("field", ["preprocessing_version", "landmark_runtime"])
def test_adapter_rejects_missing_runtime_contract_metadata(tmp_path, monkeypatch, field):
    model = VisionBridgeLetterBaseModel()
    base = tmp_path / "base.pt"
    save_checkpoint(model, base)

    monkeypatch.setattr(
        letter_fewshot.settings,
        "LETTER_BASE_MODEL_PATH",
        str(base),
    )
    monkeypatch.setattr(
        letter_fewshot.settings,
        "ADAPTER_WEIGHTS_DIR",
        str(tmp_path / "adapters"),
    )

    fitted = letter_fewshot.fit_prototype_adapter(
        model,
        [("A", _pair(1)), ("A", _pair(2)), ("A", _pair(3)), ("B", _pair(4)), ("B", _pair(5)), ("B", _pair(6))],
    )
    fitted["payload"].pop(field)
    path = letter_fewshot.save_prototype_adapter(fitted["payload"])

    with pytest.raises(ValueError, match="landmark runtime|preprocessing"):
        letter_fewshot.load_prototype_adapter(path, base)


def test_adapter_does_not_persist_raw_calibration_landmarks(tmp_path, monkeypatch):
    model = VisionBridgeLetterBaseModel()
    base = tmp_path / "base.pt"
    save_checkpoint(model, base)

    monkeypatch.setattr(
        letter_fewshot.settings,
        "LETTER_BASE_MODEL_PATH",
        str(base),
    )
    monkeypatch.setattr(
        letter_fewshot.settings,
        "ADAPTER_WEIGHTS_DIR",
        str(tmp_path / "adapters"),
    )

    fitted = letter_fewshot.fit_prototype_adapter(
        model,
        [("A", _pair(1)), ("A", _pair(2)), ("A", _pair(3)), ("B", _pair(4)), ("B", _pair(5)), ("B", _pair(6))],
    )

    assert "calibration_samples" not in fitted["payload"]


def test_calibration_request_requires_three_examples_per_letter():
    samples = [
        {"letter": "A", "hand_keypoints": _pair(1)},
        {"letter": "A", "hand_keypoints": _pair(2)},
        {"letter": "B", "hand_keypoints": _pair(3)},
        {"letter": "B", "hand_keypoints": _pair(4)},
        {"letter": "B", "hand_keypoints": _pair(5)},
    ]

    with pytest.raises(ValueError, match="at least 3 examples"):
        LetterCalibrationRequest(user_id=1, samples=samples)

    valid_samples = samples + [{"letter": "A", "hand_keypoints": _pair(6)}]
    request = LetterCalibrationRequest(user_id=1, samples=valid_samples)

    assert len(request.samples) == 6


@pytest.mark.parametrize(
    "mutator,message",
    [
        (
            lambda payload: payload.__setitem__("input_dim", 64),
            "input dimension",
        ),
        (
            lambda payload: payload.__setitem__("labels", ["A", "B"]),
            "label vocabulary",
        ),
        (
            lambda payload: payload["state_dict"]["encoder.1.weight"].__setitem__(
                0, torch.full_like(payload["state_dict"]["encoder.1.weight"][0], float("nan"))
            ),
            "state_dict",
        ),
    ],
)
def test_checkpoint_rejects_incompatible_active_contract(tmp_path, mutator, message):
    path = tmp_path / "base.pt"
    save_checkpoint(VisionBridgeLetterBaseModel(), path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mutator(payload)
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)


def test_train_validation_split_rejects_class_without_train_side(tmp_path):
    features = np.arange(26 * 3 * 126, dtype=np.float32).reshape(26 * 3, 126)
    labels = np.repeat(np.arange(26), 3)
    samples = []
    for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        samples.extend(
            [
                {"source_path": f"{letter}/a.jpg", "source_split": "training", "letter": letter, "image_sha256": f"{index}-shared"},
                {"source_path": f"{letter}/b.jpg", "source_split": "training", "letter": letter, "image_sha256": f"{index}-shared"},
                {"source_path": f"{letter}/c.jpg", "source_split": "training", "letter": letter, "image_sha256": f"{index}-unique"},
            ]
        )

    with pytest.raises(ValueError, match="both train and validation"):
        prepare_letter_dataset.make_train_validation_split(
            features,
            labels,
            samples,
            validation_ratio=0.9,
            seed=42,
        )


def test_legacy_v3_checkpoint_migrates_without_weight_changes(tmp_path):
    source = tmp_path / "legacy.pt"
    destination = tmp_path / "migrated.pt"
    model = VisionBridgeLetterBaseModel()
    save_checkpoint(model, source)

    legacy = torch.load(source, map_location="cpu", weights_only=True)
    legacy.pop("preprocessing_version")
    legacy.pop("landmark_runtime")
    torch.save(legacy, source)

    migrate_checkpoint(source, destination)
    loaded = load_checkpoint(destination)
    migrated = torch.load(destination, map_location="cpu", weights_only=True)

    assert loaded.input_dim == 126
    assert loaded.hidden_dim == 128
    assert loaded.embedding_dim == 64
    assert loaded.labels == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert migrated["preprocessing_version"] == PREPROCESSING_VERSION
    assert migrated["landmark_runtime"] == LANDMARK_RUNTIME

    for key, value in legacy["state_dict"].items():
        assert torch.equal(value, migrated["state_dict"][key])


def test_train_validation_split_rejects_same_image_under_multiple_labels():
    features = np.zeros((52, 126), dtype=np.float32)
    labels = np.repeat(np.arange(26), 2)
    samples = []
    for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        samples.extend(
            [
                {"source_path": f"{letter}/a.jpg", "source_split": "training", "letter": letter, "image_sha256": f"hash-{index}"},
                {"source_path": f"{letter}/b.jpg", "source_split": "training", "letter": letter, "image_sha256": f"unique-{index}"},
            ]
        )

    samples[-1]["image_sha256"] = "hash-0"

    with pytest.raises(ValueError, match="multiple labels"):
        prepare_letter_dataset.make_train_validation_split(
            features,
            labels,
            samples,
            validation_ratio=0.2,
            seed=42,
        )


def test_calibration_request_rejects_more_than_130_samples():
    samples = [{"letter": "A", "hand_keypoints": _pair(index)} for index in range(130)]
    samples[1]["letter"] = "B"
    samples[2]["letter"] = "B"
    samples[3]["letter"] = "B"

    with pytest.raises(ValueError):
        LetterCalibrationRequest(user_id=1, samples=samples + [
            {"letter": "C", "hand_keypoints": _pair(999)}
        ])


def test_fit_prototype_adapter_requires_three_examples_per_letter(tmp_path, monkeypatch):
    model = VisionBridgeLetterBaseModel()
    base = tmp_path / "base.pt"
    save_checkpoint(model, base)

    monkeypatch.setattr(letter_fewshot.settings, "LETTER_BASE_MODEL_PATH", str(base))
    monkeypatch.setattr(letter_fewshot.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path / "adapters"))

    with pytest.raises(ValueError, match="at least 3 examples"):
        letter_fewshot.fit_prototype_adapter(
            model,
            [("A", _pair(1)), ("A", _pair(2)), ("A", _pair(3)),
             ("B", _pair(4)), ("B", _pair(5))],
        )
