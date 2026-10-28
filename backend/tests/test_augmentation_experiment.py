from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_augmentation_experiment as experiment
from app.services.letter_fewshot import normalize_hand_pair


def _raw_hand(center_x: float = 0.3) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = [center_x, 0.65, 0.0]
    points[1] = [center_x - 0.02, 0.63, 0.001]
    points[2] = [center_x - 0.03, 0.61, 0.002]
    points[3] = [center_x - 0.04, 0.59, 0.003]
    points[4] = [center_x - 0.05, 0.57, 0.004]
    for base, offset in ((5, -0.03), (9, -0.01), (13, 0.01), (17, 0.03)):
        points[base] = [center_x + offset, 0.62, 0.0]
        points[base + 1] = [center_x + offset, 0.59, 0.001]
        points[base + 2] = [center_x + offset, 0.56, 0.002]
        points[base + 3] = [center_x + offset, 0.53, 0.003]
    return points


def _make_data_dirs(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data-v1"
    raw_dir = tmp_path / "raw"
    data_dir.mkdir()
    raw_dir.mkdir()
    (data_dir / "labels.json").write_text(json.dumps({"labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}), encoding="utf-8")
    (raw_dir / "labels.json").write_text(
        json.dumps({"raw_landmark_contract": "raw-mediapipe-xyz-before-normalization-v1"}),
        encoding="utf-8",
    )
    rows = []
    raw = []
    y = np.arange(26, dtype=np.int64)
    for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        row = {
            "sample_id": f"sample-{index}",
            "source_path": f"images/{letter.lower()}.png",
            "image_sha256": f"{index + 1:064x}",
            "label": letter,
            "letter": letter,
            "landmark_status": "ok",
        }
        rows.append(row)
        raw.append(np.concatenate((_raw_hand(0.3 + index * 0.0001).reshape(-1), np.zeros(63, dtype=np.float32))))
    raw_x = np.stack(raw).astype(np.float32)
    normalized_x = np.stack([normalize_hand_pair(item.tolist()) for item in raw_x]).astype(np.float32)
    for split in ("train", "val"):
        np.savez_compressed(data_dir / f"{split}.npz", x=normalized_x, y=y)
        with (data_dir / f"{split}_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        np.savez_compressed(
            raw_dir / f"{split}_raw.npz",
            x=raw_x,
            left_present=np.ones(26, dtype=np.bool_),
            right_present=np.zeros(26, dtype=np.bool_),
            y=y,
        )
        with (raw_dir / f"{split}_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return data_dir, raw_dir


def test_augmentation_runner_uses_validation_only_and_does_not_require_test_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, raw_dir = _make_data_dirs(tmp_path)
    phase2 = tmp_path / "phase2.json"
    phase2.write_text(json.dumps({
        "status": "completed",
        "selection": {"variant": "v3_mlp"},
        "candidates": {"v3_mlp": {"feature_contract": "two-hand-wrist-scale-v1"}},
    }), encoding="utf-8")
    output = tmp_path / "augmentation-run"
    train_calls = []
    eval_calls = []

    def fake_train_model(**kwargs):
        train_calls.append(kwargs)
        checkpoint = kwargs["output_path"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(checkpoint.name.encode("utf-8"))
        return {
            "best_epoch": 2,
            "epochs_completed": 3,
            "parameter_count": 123,
        }

    def fake_evaluate(*, checkpoint, data_dir, split, batch_size):
        eval_calls.append(split)
        assert split == "val"
        assert not (data_dir / "test.npz").exists()
        candidate = "candidate" in checkpoint.name
        acc = 0.75 if candidate else 0.70
        cm = [[0 for _ in range(26)] for _ in range(26)]
        cm[22][12] = 2 if candidate else 5
        cm[22][13] = 3 if candidate else 6
        cm[22][14] = 4 if candidate else 7
        return {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "overall_accuracy": acc,
            "macro_accuracy": acc,
            "worst_letter": "W",
            "worst_letter_accuracy": acc - 0.1,
            "per_letter_accuracy": {letter: acc for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
            "confusion_matrix": cm,
            "latency": {"mean_batch_ms": 0.4, "mean_sample_ms": 0.02},
            "samples": 26,
            "labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            "device": "cpu",
        }

    monkeypatch.setattr(experiment, "train_model", fake_train_model)
    monkeypatch.setattr(experiment, "evaluate", fake_evaluate)
    report = experiment.run_augmentation_experiment(
        phase2_report=phase2,
        data_dir_v1=data_dir,
        data_dir_v4=tmp_path / "not-used-v4",
        raw_dir=raw_dir,
        output_dir=output,
        augmentation="gaussian_jitter",
        seeds=(7, 11, 19),
        epochs=3,
        batch_size=26,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    assert len(train_calls) == 6
    assert len(eval_calls) == 6
    assert set(eval_calls) == {"val"}
    assert all(call["validation_only"] is True for call in train_calls)
    assert all(call["fixed_epoch_budget"] is True for call in train_calls)
    assert all(call["seed"] in {7, 11, 19} for call in train_calls)
    assert report["protocol"]["test_accuracy_evaluation_calls"] == 0
    assert report["candidate"]["keep_candidate"] is True
    assert report["candidate"]["w_confusion_count_delta"]["W_to_M"]["mean"] == -3.0
    assert not (data_dir / "test.npz").exists()
    assert (output / "augmentation_result.json").is_file()


def test_combined_runner_requires_all_four_standalone_results_before_marker(
    tmp_path: Path,
) -> None:
    phase2 = tmp_path / "phase2.json"
    phase2.write_text(json.dumps({
        "status": "completed",
        "selection": {"variant": "v3_mlp"},
        "candidates": {"v3_mlp": {"feature_contract": "two-hand-wrist-scale-v1"}},
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="not found"):
        experiment.run_augmentation_experiment(
            phase2_report=phase2,
            data_dir_v1=tmp_path / "v1",
            data_dir_v4=tmp_path / "v4",
            raw_dir=tmp_path / "raw",
            output_dir=tmp_path / "combined",
            augmentation="combined",
            seeds=(7, 11, 19),
            epochs=3,
            batch_size=26,
            learning_rate=1e-3,
            weight_decay=1e-4,
            standalone_results=(tmp_path / "missing.json",),
            experiment_root=tmp_path / "experiment-root",
        )
    assert not (tmp_path / "experiment-root" / "combined_augmentation_attempted.json").exists()
