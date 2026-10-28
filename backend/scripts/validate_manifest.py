"""Validate VisionBridge manifest schema v2 JSONL files."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

LABELS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VALID_SPLITS = frozenset({"unassigned", "train", "val", "test"})
LANDMARK_STATUSES = frozenset({"ok", "no_hand", "low_conf", "failed"})
CAPTURE_FIELDS = (
    "device",
    "lighting",
    "background",
    "distance_cm_bucket",
    "hand_dominance",
    "resolution",
    "fps",
)
REQUIRED_FIELDS = (
    "signer_id",
    "recording_id",
    "sample_id",
    "label",
    "source_path",
    "image_sha256",
    "split",
    "capture_metadata",
    "consent_id",
    "landmark_status",
    "failure_reason",
)


class ManifestValidationError(ValueError):
    """Raised when a manifest violates schema v2 or split integrity."""


def validate_manifest_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate parsed rows and return them unchanged.

    Sample IDs are unique across the manifest. Signer and recording IDs are
    intentionally repeated across samples, but a recording must belong to one
    signer and an assigned signer may appear in exactly one of train/val/test.
    """
    rows = list(records)
    errors: list[str] = []
    seen_samples: dict[str, int] = {}
    recording_signers: dict[str, str] = {}
    signer_splits: dict[str, set[str]] = defaultdict(set)
    split_values: set[str] = set()

    for index, row in enumerate(rows, start=1):
        prefix = f"line {index}"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: row must be a JSON object")
            continue

        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"{prefix}: missing required field(s): {', '.join(missing)}")

        for field in ("signer_id", "recording_id", "sample_id", "source_path", "consent_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}: {field} must be a non-empty string")

        signer_id = row.get("signer_id")
        recording_id = row.get("recording_id")
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str) and sample_id.strip():
            previous = seen_samples.get(sample_id)
            if previous is not None:
                errors.append(
                    f"{prefix}: duplicate sample_id {sample_id!r} (first seen on line {previous})"
                )
            else:
                seen_samples[sample_id] = index

        if (
            isinstance(signer_id, str) and signer_id.strip()
            and isinstance(recording_id, str) and recording_id.strip()
        ):
            prior_signer = recording_signers.get(recording_id)
            if prior_signer is not None and prior_signer != signer_id:
                errors.append(
                    f"{prefix}: recording_id {recording_id!r} is associated with multiple signer IDs"
                )
            recording_signers[recording_id] = signer_id

        label = row.get("label")
        if not isinstance(label, str) or label not in LABELS:
            errors.append(f"{prefix}: label must be exactly one uppercase letter A-Z")

        split = row.get("split")
        if not isinstance(split, str) or split not in VALID_SPLITS:
            errors.append(
                f"{prefix}: split must be one of {', '.join(sorted(VALID_SPLITS))}"
            )
        else:
            split_values.add(split)
            if split != "unassigned" and isinstance(signer_id, str) and signer_id.strip():
                signer_splits[signer_id].add(split)

        image_sha256 = row.get("image_sha256")
        if not isinstance(image_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", image_sha256) is None:
            errors.append(f"{prefix}: image_sha256 must be a 64-character hexadecimal SHA-256")

        capture = row.get("capture_metadata")
        if not isinstance(capture, dict):
            errors.append(f"{prefix}: capture_metadata must be an object")
        else:
            absent_capture_fields = [field for field in CAPTURE_FIELDS if field not in capture]
            if absent_capture_fields:
                errors.append(
                    f"{prefix}: capture_metadata missing field(s): {', '.join(absent_capture_fields)}"
                )
            fps = capture.get("fps")
            if fps is not None and (
                isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0
            ):
                errors.append(f"{prefix}: capture_metadata.fps must be a positive number or null")

        status = row.get("landmark_status")
        if not isinstance(status, str) or status not in LANDMARK_STATUSES:
            errors.append(
                f"{prefix}: landmark_status must be one of {', '.join(sorted(LANDMARK_STATUSES))}"
            )
        failure_reason = row.get("failure_reason")
        if failure_reason is not None and not isinstance(failure_reason, str):
            errors.append(f"{prefix}: failure_reason must be a string or null")
        if isinstance(status, str) and status in {"no_hand", "low_conf", "failed"} and (
            not isinstance(failure_reason, str) or not failure_reason.strip()
        ):
            errors.append(f"{prefix}: failure_reason is required when landmark_status is not ok")

    if "unassigned" in split_values and len(split_values) > 1:
        errors.append("manifest must use either all split='unassigned' rows or fully assigned train/val/test rows")

    for signer_id, splits in sorted(signer_splits.items()):
        if len(splits) > 1:
            errors.append(
                f"signer_id {signer_id!r} appears in more than one split: {', '.join(sorted(splits))}"
            )

    if not rows:
        errors.append("manifest must contain at least one sample row")

    if errors:
        raise ManifestValidationError("\n".join(errors))
    return rows


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            records.append(record)
    return validate_manifest_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a VisionBridge manifest schema v2 JSONL file.")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    signers = {record["signer_id"] for record in records}
    splits = sorted({record["split"] for record in records})
    print(f"manifest={args.manifest}")
    print(f"samples={len(records)}")
    print(f"signers={len(signers)}")
    print(f"splits={','.join(splits)}")
    print("manifest_validation=PASS")


if __name__ == "__main__":
    main()
