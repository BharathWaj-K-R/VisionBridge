from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_temporal_sequences import diagnose_manifests


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_temporal_diagnostic_finds_explicit_ordered_labelled_frames(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "train_manifest.jsonl",
        [
            {"recording_id": "R1", "label": "W", "frame_index": 12, "sample_id": "s3"},
            {"recording_id": "R1", "label": "W", "frame_index": 10, "sample_id": "s1"},
            {"recording_id": "R1", "label": "W", "frame_index": 11, "sample_id": "s2"},
            {"recording_id": "R2", "label": "A", "frame_index": 0},
        ],
    )
    report = diagnose_manifests([manifest])
    assert report["status"] == "ordered_sequences_found"
    assert report["temporal_model_training_allowed"] is True
    assert report["ordered_sequence_group_count"] == 1
    assert report["ordered_sequences"][0]["first_frame_index"] == 10
    assert report["ordered_sequences"][0]["last_frame_index"] == 12


def test_temporal_diagnostic_rejects_still_image_manifests_without_frame_index(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "test_manifest.jsonl",
        [
            {"recording_id": "R1", "label": "W", "sample_id": "s1"},
            {"recording_id": "R1", "label": "W", "sample_id": "s2"},
        ],
    )
    report = diagnose_manifests([manifest])
    assert report["status"] == "not_testable"
    assert report["temporal_model_training_allowed"] is False
    assert report["rows_missing_frame_index"] == 2
    assert report["ordered_sequence_group_count"] == 0


def test_temporal_diagnostic_flags_duplicate_frame_indexes(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "val_manifest.jsonl",
        [
            {"recording_id": "R1", "label": "M", "frame_index": 3},
            {"recording_id": "R1", "label": "M", "frame_index": 3},
        ],
    )
    report = diagnose_manifests([manifest])
    assert report["status"] == "not_testable"
    assert report["duplicate_frame_index_group_count"] == 1
    assert report["ordered_sequence_group_count"] == 0


def test_temporal_diagnostic_reports_no_manifests_as_not_testable() -> None:
    report = diagnose_manifests([])
    assert report["status"] == "not_testable"
    assert report["scanned_files"] == 0
    assert report["temporal_model_training_allowed"] is False
