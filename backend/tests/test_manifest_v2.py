from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.prepare_letter_dataset import (
    assign_signer_splits,
    build_extraction_failure_report,
    enforce_no_cross_split_near_duplicates,
    find_near_duplicate_landmarks,
)
from scripts.validate_manifest import ManifestValidationError, validate_manifest_records


CAPTURE = {
    "device": "webcam-model-x",
    "lighting": "indoor-diffuse",
    "background": "plain",
    "distance_cm_bucket": "50-100",
    "hand_dominance": "right",
    "resolution": "1280x720",
    "fps": 30,
}


def manifest_row(
    sample_id: str,
    signer_id: str = "S001",
    recording_id: str = "R001",
    label: str = "A",
    split: str = "unassigned",
    **updates,
) -> dict:
    row = {
        "signer_id": signer_id,
        "recording_id": recording_id,
        "sample_id": sample_id,
        "label": label,
        "source_path": f"frames/{sample_id}.jpg",
        "image_sha256": f"{len(sample_id):064x}",
        "split": split,
        "capture_metadata": copy.deepcopy(CAPTURE),
        "consent_id": "CONSENT-REF-001",
        "landmark_status": "ok",
        "failure_reason": None,
    }
    row.update(updates)
    return row


def test_manifest_validator_accepts_repeated_signer_and_recording_ids() -> None:
    rows = [
        manifest_row("sample-1"),
        manifest_row("sample-2"),
    ]
    assert len(validate_manifest_records(rows)) == 2


def test_manifest_validator_rejects_duplicate_sample_ids() -> None:
    with pytest.raises(ManifestValidationError, match="duplicate sample_id"):
        validate_manifest_records([
            manifest_row("sample-1"),
            manifest_row("sample-1", recording_id="R002"),
        ])


def test_manifest_validator_rejects_signer_in_multiple_splits() -> None:
    with pytest.raises(ManifestValidationError, match="appears in more than one split"):
        validate_manifest_records([
            manifest_row("sample-1", split="train"),
            manifest_row("sample-2", split="test"),
        ])


def test_manifest_validator_rejects_absent_consent_and_invalid_label() -> None:
    absent_consent = manifest_row("sample-1", consent_id="")
    invalid_label = manifest_row("sample-2", label="AA")
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest_records([absent_consent, invalid_label])
    assert "consent_id must be a non-empty string" in str(exc.value)
    assert "label must be exactly one uppercase letter A-Z" in str(exc.value)


def test_manifest_validator_rejects_incomplete_metadata_and_failed_status_without_reason() -> None:
    row = manifest_row("sample-1", landmark_status="no_hand", failure_reason=None)
    row["capture_metadata"].pop("lighting")
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest_records([row])
    assert "capture_metadata missing field(s): lighting" in str(exc.value)
    assert "failure_reason is required" in str(exc.value)


def test_signer_split_is_deterministic_disjoint_and_stratified_when_possible() -> None:
    rows = []
    for signer_number in range(1, 10):
        signer = f"S{signer_number:03d}"
        for label in "ABCD":
            rows.append(manifest_row(
                sample_id=f"{signer}-{label}",
                signer_id=signer,
                recording_id=f"R-{signer}",
                label=label,
            ))

    first = assign_signer_splits(rows, seed=314)
    second = assign_signer_splits(rows, seed=314)
    assert [row["split"] for row in first] == [row["split"] for row in second]

    signer_split: dict[str, set[str]] = {}
    for row in first:
        signer_split.setdefault(row["signer_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in signer_split.values())
    assert {row["split"] for row in first} == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert {row["label"] for row in first if row["split"] == split} == set("ABCD")


def test_near_duplicate_detector_reports_recording_pairs_and_cross_split_leakage() -> None:
    features = np.zeros((4, 126), dtype=np.float32)
    features[1, 0] = 0.05
    features[2, 1] = 0.01
    features[3, :] = 1.0
    rows = [
        {"sample_id": "a", "source_path": "a.jpg", "image_sha256": "a" * 64,
         "prepared_split": "train", "signer_id": "S001", "recording_id": "R001", "label": "A"},
        {"sample_id": "b", "source_path": "b.jpg", "image_sha256": "b" * 64,
         "prepared_split": "train", "signer_id": "S001", "recording_id": "R001", "label": "A"},
        {"sample_id": "c", "source_path": "c.jpg", "image_sha256": "c" * 64,
         "prepared_split": "test", "signer_id": "S002", "recording_id": "R002", "label": "A"},
        {"sample_id": "d", "source_path": "d.jpg", "image_sha256": "d" * 64,
         "prepared_split": "val", "signer_id": "S003", "recording_id": "R003", "label": "B"},
    ]
    report = find_near_duplicate_landmarks(features, rows, threshold=0.01)
    assert report["within_recording_pair_count"] >= 1
    assert report["cross_split_pair_count"] >= 1
    with pytest.raises(RuntimeError, match="cross-split landmark near-duplicate"):
        enforce_no_cross_split_near_duplicates(report)


def test_extraction_failure_report_counts_letter_signer_and_conditions() -> None:
    rows = [
        manifest_row("sample-1", signer_id="S001", label="A", landmark_status="ok"),
        manifest_row(
            "sample-2",
            signer_id="S002",
            label="B",
            landmark_status="low_conf",
            failure_reason="handedness_confidence_below_threshold",
            capture_metadata={**CAPTURE, "lighting": "low-light"},
        ),
    ]
    report = build_extraction_failure_report(rows)
    assert report["total_attempted"] == 2
    assert report["total_skipped"] == 1
    assert report["failure_rate"] == 0.5
    assert report["per_letter"]["B"]["skipped"] == 1
    assert report["per_signer"]["S002"]["skipped"] == 1
    assert report["per_condition"]["lighting"]["low-light"]["skipped"] == 1
