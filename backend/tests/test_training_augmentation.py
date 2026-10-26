import pytest
import torch

from app.training.letter_base import (
    augment_landmark_batch,
    build_focus_weights,
    classification_loss,
)


def _sample_features(batch_size: int = 2) -> torch.Tensor:
    features = torch.zeros((batch_size, 126), dtype=torch.float32)
    # A non-degenerate left-hand skeleton with a wrist anchor at the origin.
    points = torch.arange(63, dtype=torch.float32).reshape(21, 3) / 63.0
    points[0] = 0.0
    features[0, :63] = points.reshape(-1)
    return features


def test_landmark_augmentation_preserves_missing_hand_and_wrist_anchor():
    torch.manual_seed(17)
    features = _sample_features()
    original = features.clone()

    augmented = augment_landmark_batch(
        features,
        jitter_std=0.004,
        rotation_degrees=5.0,
    )

    assert augmented.shape == (2, 126)
    assert torch.isfinite(augmented).all()
    assert torch.equal(features, original), "augmentation must not mutate the dataset tensor"
    assert torch.equal(augmented[0, :3], torch.zeros(3)), "wrist origin must remain fixed"
    assert torch.equal(augmented[0, 63:], torch.zeros(63)), "missing right hand must remain zero"
    assert torch.equal(augmented[1], torch.zeros(126)), "a sample without hands must remain zero"
    assert not torch.equal(augmented[0, :63], original[0, :63])


def test_zero_landmark_augmentation_is_an_identity_operation():
    features = _sample_features()
    result = augment_landmark_batch(features, jitter_std=0.0, rotation_degrees=0.0)
    assert result is features


def test_focus_weights_emphasize_only_selected_classes():
    labels = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    weights = build_focus_weights(labels, ("W", "M"), 1.5)

    assert weights[labels.index("W")].item() == pytest.approx(1.5)
    assert weights[labels.index("M")].item() == pytest.approx(1.5)
    assert weights[labels.index("A")].item() == pytest.approx(1.0)


def test_focal_loss_is_finite_and_differentiable():
    labels = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    weights = build_focus_weights(labels, ("W", "M"), 1.35)
    logits = torch.randn(8, 26, requires_grad=True)
    targets = torch.tensor([0, 12, 22, 22, 3, 12, 4, 22], dtype=torch.long)

    loss = classification_loss(
        logits,
        targets,
        loss_name="focal",
        focal_gamma=1.5,
        class_weights=weights,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_invalid_rotation_range_is_rejected():
    with pytest.raises(ValueError, match="rotation_degrees"):
        augment_landmark_batch(_sample_features(), rotation_degrees=20.0)

def test_validation_only_training_does_not_require_test_split(tmp_path):
    import json

    import numpy as np

    from app.training.letter_base import train_model

    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    (tmp_path / "labels.json").write_text(
        json.dumps({"labels": labels}),
        encoding="utf-8",
    )

    rng = np.random.default_rng(73)
    train_x = rng.normal(0.0, 0.25, size=(52, 126)).astype(np.float32)
    train_y = np.repeat(np.arange(26, dtype=np.int64), 2)
    val_x = rng.normal(0.0, 0.25, size=(26, 126)).astype(np.float32)
    val_y = np.arange(26, dtype=np.int64)
    np.savez_compressed(tmp_path / "train.npz", x=train_x, y=train_y)
    np.savez_compressed(tmp_path / "val.npz", x=val_x, y=val_y)

    # Deliberately do not create test.npz. Candidate selection must use validation only.
    result = train_model(
        root=tmp_path,
        output_path=tmp_path / "candidate.pt",
        epochs=1,
        batch_size=26,
        learning_rate=1e-3,
        weight_decay=1e-4,
        target_class_accuracy=1.0,
        seed=73,
        hidden_dim=16,
        embedding_dim=8,
        dropout=0.0,
        validation_only=True,
    )

    assert result["test_accuracy"] is None
    assert result["test_per_letter"] is None
    assert result["test_counts"] is None
    assert len(result["validation_per_letter"]) == 26
    assert (tmp_path / "candidate.pt").is_file()

def test_training_cli_writes_metrics_json(tmp_path, monkeypatch):
    import json
    import sys

    import numpy as np

    from app.training.letter_base import main as training_main

    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    (tmp_path / "labels.json").write_text(
        json.dumps({"labels": labels}),
        encoding="utf-8",
    )
    rng = np.random.default_rng(91)
    train_x = rng.normal(0.0, 0.25, size=(52, 126)).astype(np.float32)
    train_y = np.repeat(np.arange(26, dtype=np.int64), 2)
    val_x = rng.normal(0.0, 0.25, size=(26, 126)).astype(np.float32)
    val_y = np.arange(26, dtype=np.int64)
    np.savez_compressed(tmp_path / "train.npz", x=train_x, y=train_y)
    np.savez_compressed(tmp_path / "val.npz", x=val_x, y=val_y)

    checkpoint = tmp_path / "candidate.pt"
    report_path = tmp_path / "candidate-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "letter_base.py",
            "--data-dir", str(tmp_path),
            "--output", str(checkpoint),
            "--epochs", "1",
            "--batch-size", "26",
            "--hidden-dim", "16",
            "--embedding-dim", "8",
            "--dropout", "0",
            "--validation-only",
            "--output-json", str(report_path),
        ],
    )

    training_main()

    assert checkpoint.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["test_accuracy"] is None
    assert report["test_per_letter"] is None
    assert report["validation_macro_accuracy"] >= 0.0
    assert len(report["validation_per_letter"]) == 26

