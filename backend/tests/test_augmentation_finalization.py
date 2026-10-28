from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.finalize_augmentation_test as finalizer
from scripts.aggregate_augmentation_results import AUGMENTATIONS


def _write_report(path: Path, augmentation: str, phase2_hash: str, *, keep: bool = False, baseline_checkpoint: Path, candidate_checkpoint: Path) -> None:
    def run(role, checkpoint, overall, macro, worst):
        return {
            "role": role,
            "seed": 42,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "validation": {
                "overall_accuracy": overall,
                "macro_accuracy": macro,
                "worst_letter_accuracy": worst,
                "per_letter_accuracy": {letter: worst for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
                "confusion_matrix": [[0] * 26 for _ in range(26)],
            },
        }

    baseline_runs = [
        run("baseline", baseline_checkpoint, 0.80, 0.79, 0.60),
        run("baseline", baseline_checkpoint, 0.81, 0.80, 0.61),
        run("baseline", baseline_checkpoint, 0.82, 0.81, 0.62),
    ]
    candidate_runs = [
        run("candidate", candidate_checkpoint, 0.82, 0.81, 0.62),
        run("candidate", candidate_checkpoint, 0.83, 0.82, 0.63),
        run("candidate", candidate_checkpoint, 0.84, 0.83, 0.64),
    ]
    report = {
        "status": "validation_completed",
        "augmentation": augmentation,
        "phase2_report_sha256": phase2_hash,
        "phase2_winner": "v3_mlp",
        "feature_contract": "two-hand-wrist-scale-v1",
        "model_version": "v3",
        "seeds": [42, 123, 314],
        "protocol": {"test_split_opened": False, "test_accuracy_evaluation_calls": 0},
        "aligned_train_val_manifests": {
            "train": {"samples": 26, "source_manifest_sha256": "b" * 64, "raw_manifest_sha256": "b" * 64},
            "val": {"samples": 26, "source_manifest_sha256": "c" * 64, "raw_manifest_sha256": "c" * 64},
        },
        "baseline": {"runs": baseline_runs},
        "candidate": {
            "keep_candidate": keep,
            "mean_macro_gain": 0.02,
            "summary": {
                "macro_accuracy": {"mean": 0.82, "std": 0.01},
                "worst_class_accuracy": {"mean": 0.63, "std": 0.01},
                "overall_accuracy": {"mean": 0.83, "std": 0.01},
            },
            "runs": candidate_runs,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_finalizer_freezes_selection_before_one_test_call_and_rejects_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    phase2_path = tmp_path / "phase2.json"
    phase2_path.write_text(json.dumps({"status": "completed", "selection": {"variant": "v3_mlp"}}), encoding="utf-8")
    phase2_hash = hashlib.sha256(phase2_path.read_bytes()).hexdigest()
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    baseline.write_bytes(b"baseline checkpoint")
    candidate.write_bytes(b"candidate checkpoint")
    result_paths = []
    for aug in AUGMENTATIONS:
        path = tmp_path / f"{aug}.json"
        _write_report(path, aug, phase2_hash, keep=False, baseline_checkpoint=baseline, candidate_checkpoint=candidate)
        result_paths.append(path)

    data_v1 = tmp_path / "v1"
    data_v4 = tmp_path / "v4"
    data_v1.mkdir()
    data_v4.mkdir()
    np.savez_compressed(data_v1 / "test.npz", x=np.zeros((26, 126), dtype=np.float32), y=np.arange(26))
    np.savez_compressed(data_v4 / "test.npz", x=np.zeros((26, 141), dtype=np.float32), y=np.arange(26))
    output_dir = tmp_path / "final"
    calls = []

    def fake_evaluate(*, checkpoint, data_dir, split, batch_size):
        calls.append((checkpoint, data_dir, split))
        assert split == "test"
        assert (output_dir / "selection_frozen.json").is_file()
        assert (output_dir / "test_evaluation_attempted.json").is_file()
        selected = json.loads((output_dir / "selection_frozen.json").read_text(encoding="utf-8"))
        assert selected["selected_augmentation"] == "none"
        return {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "overall_accuracy": 0.91,
            "macro_accuracy": 0.90,
            "samples": 26,
        }

    monkeypatch.setattr(finalizer, "evaluate", fake_evaluate)
    result = finalizer.finalize_test_evaluation(
        phase2_report=phase2_path,
        standalone_results=result_paths,
        output_dir=output_dir,
        data_dir_v1=data_v1,
        data_dir_v4=data_v4,
        batch_size=8,
    )
    assert len(calls) == 1
    assert result["overall_accuracy"] == 0.91
    assert (output_dir / "final_test_evaluation.json").is_file()
    with pytest.raises(FileExistsError, match="already frozen/attempted"):
        finalizer.finalize_test_evaluation(
            phase2_report=phase2_path,
            standalone_results=result_paths,
            output_dir=output_dir,
            data_dir_v1=data_v1,
            data_dir_v4=data_v4,
            batch_size=8,
        )
    assert len(calls) == 1


def test_finalizer_requires_combined_validation_if_multiple_augmentations_kept(tmp_path: Path) -> None:
    phase2_path = tmp_path / "phase2.json"
    phase2_path.write_text(json.dumps({"status": "completed", "selection": {"variant": "v3_mlp"}}), encoding="utf-8")
    phase2_hash = hashlib.sha256(phase2_path.read_bytes()).hexdigest()
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    paths = []
    for index, augmentation in enumerate(AUGMENTATIONS):
        path = tmp_path / f"{augmentation}.json"
        _write_report(path, augmentation, phase2_hash, keep=index < 2, baseline_checkpoint=baseline, candidate_checkpoint=candidate)
        paths.append(path)
    with pytest.raises(ValueError, match="single gated combined validation"):
        finalizer.choose_final_candidate(
            standalone_paths=paths,
            phase2_report_path=phase2_path,
            combined_result_path=None,
        )
