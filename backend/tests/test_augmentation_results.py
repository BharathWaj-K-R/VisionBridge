from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_augmentation_results import load_standalone_reports, render_results_table, update_markdown


def _report(augmentation: str, *, keep: bool = True) -> dict:
    per_letter = {letter: {"mean": 0.01, "std": 0.005} for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    summary = {
        "macro_accuracy": {"mean": 0.91, "std": 0.01},
        "overall_accuracy": {"mean": 0.92, "std": 0.01},
        "worst_class_accuracy": {"mean": 0.70, "std": 0.03},
        "mean_batch_latency_ms": {"mean": 0.4, "std": 0.02},
        "mean_sample_latency_ms": {"mean": 0.02, "std": 0.001},
    }
    return {
        "status": "validation_completed",
        "augmentation": augmentation,
        "phase2_report_sha256": "a" * 64,
        "phase2_winner": "full_v4",
        "seeds": [42, 123, 314],
        "protocol": {"test_split_opened": False, "test_accuracy_evaluation_calls": 0},
        "aligned_train_val_manifests": {
            "train": {"samples": 26, "source_manifest_sha256": "b" * 64, "raw_manifest_sha256": "b" * 64},
            "val": {"samples": 26, "source_manifest_sha256": "c" * 64, "raw_manifest_sha256": "c" * 64},
        },
        "candidate": {
            "keep_candidate": keep,
            "summary": summary,
            "mean_macro_gain": 0.02,
            "required_gain_threshold": 0.01,
            "rejection": {"rejection_rate": 0.0},
            "w_confusion_count_delta": {
                "W_to_M": {"mean": -1.0, "std": 0.5},
                "W_to_N": {"mean": -2.0, "std": 0.5},
                "W_to_O": {"mean": 0.0, "std": 0.0},
            },
            "per_letter_accuracy_delta": per_letter,
        },
    }


def test_aggregation_renders_only_supplied_reports_and_marks_gaps() -> None:
    reports = {"gaussian_jitter": _report("gaussian_jitter"), "depth_scale": _report("depth_scale", keep=False)}
    rendered = render_results_table(reports)
    assert "PARTIAL (2/4 standalone runs completed)" in rendered
    assert "| gaussian_jitter | 91.000% ± 1.000% | +2.000 pp |" in rendered
    assert "| in_plane_rotation | NOT RUN | NOT RUN |" in rendered
    assert "W→M count" in rendered
    assert "-1.000 ± 0.500 counts" in rendered
    assert "-2.000 ± 0.500 counts" in rendered
    assert "0.000 ± 0.000 counts" in rendered
    assert "100.000 pp" not in rendered
    assert "W_to_M" not in rendered


def test_aggregation_rejects_report_that_opened_test_split(tmp_path: Path) -> None:
    report = _report("gaussian_jitter")
    report["protocol"]["test_split_opened"] = True
    path = tmp_path / "augmentation_result.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="test data may have been opened"):
        load_standalone_reports([path])


def test_aggregation_rejects_mismatched_splits_and_duplicate_augmentation(tmp_path: Path) -> None:
    one, two = _report("gaussian_jitter"), _report("depth_scale")
    two["aligned_train_val_manifests"]["train"]["source_manifest_sha256"] = "z" * 64
    p1, p2 = tmp_path / "one.json", tmp_path / "two.json"
    p1.write_text(json.dumps(one), encoding="utf-8")
    p2.write_text(json.dumps(two), encoding="utf-8")
    with pytest.raises(ValueError, match="do not share"):
        load_standalone_reports([p1, p2])
    p2.write_text(json.dumps(_report("gaussian_jitter")), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate standalone"):
        load_standalone_reports([p1, p2])


def test_aggregation_updates_marked_results_section(tmp_path: Path) -> None:
    doc = tmp_path / "report.md"
    doc.write_text("before\n<!-- AUTO_RESULTS_START -->\nold\n<!-- AUTO_RESULTS_END -->\nafter\n", encoding="utf-8")
    update_markdown(doc, {"gaussian_jitter": _report("gaussian_jitter")})
    result = doc.read_text(encoding="utf-8")
    assert "before" in result and "after" in result
    assert "gaussian_jitter" in result
    assert "in_plane_rotation | NOT RUN" in result
