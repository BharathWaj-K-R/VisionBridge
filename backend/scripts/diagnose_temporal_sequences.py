"""Diagnose whether current labelled manifests support ordered frame sequences.

A temporal experiment requires labelled rows with recording_id plus a numeric
frame_index and at least two distinct, non-duplicate frames per recording/label.
This script only inventories metadata; it never trains a temporal model.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FRAME_INDEX_FIELD = "frame_index"


def _frame_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def diagnose_manifests(manifest_paths: list[Path]) -> dict[str, Any]:
    paths = sorted({path.resolve() for path in manifest_paths})
    recording_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    total_rows = 0
    labelled_rows = 0
    rows_with_recording = 0
    rows_with_frame_index = 0
    eligible_rows = 0
    missing_recording_rows = 0
    missing_frame_index_rows = 0
    invalid_labels = 0
    parse_errors: list[dict[str, Any]] = []
    scanned_files = 0

    for path in paths:
        if not path.is_file():
            parse_errors.append({"path": str(path), "error": "file_not_found"})
            continue
        scanned_files += 1
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                total_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors.append({"path": str(path), "line": line_number, "error": "invalid_json"})
                    continue
                if not isinstance(row, dict):
                    parse_errors.append({"path": str(path), "line": line_number, "error": "row_is_not_object"})
                    continue

                label = row.get("label", row.get("letter"))
                if label not in LABELS:
                    invalid_labels += 1
                    continue
                labelled_rows += 1

                recording_id = row.get("recording_id")
                has_recording = isinstance(recording_id, str) and bool(recording_id.strip())
                if has_recording:
                    rows_with_recording += 1
                else:
                    missing_recording_rows += 1

                frame_index = _frame_index(row.get(FRAME_INDEX_FIELD))
                if frame_index is not None:
                    rows_with_frame_index += 1
                else:
                    missing_frame_index_rows += 1

                if has_recording and frame_index is not None:
                    eligible_rows += 1
                    recording_groups[(recording_id.strip(), label)].append(frame_index)

    ordered_groups = []
    duplicate_groups = []
    singleton_groups = []
    for (recording_id, label), indexes in sorted(recording_groups.items()):
        distinct_indexes = sorted(set(indexes))
        group = {
            "recording_id": recording_id,
            "label": label,
            "row_count": len(indexes),
            "distinct_frame_count": len(distinct_indexes),
            "first_frame_index": distinct_indexes[0] if distinct_indexes else None,
            "last_frame_index": distinct_indexes[-1] if distinct_indexes else None,
        }
        if len(distinct_indexes) != len(indexes):
            duplicate_groups.append(group)
        elif len(distinct_indexes) >= 2:
            ordered_groups.append(group)
        else:
            singleton_groups.append(group)

    sequences_found = bool(ordered_groups)
    if sequences_found:
        status = "ordered_sequences_found"
        reason = (
            "At least one recording_id/label group has two or more distinct numeric "
            "frame_index values. Confirm labels represent stable frame-level segments "
            "and inspect sequence boundaries before training a temporal model."
        )
    elif not scanned_files:
        status = "not_testable"
        reason = "No manifest JSONL files were found under the supplied path."
    elif not labelled_rows:
        status = "not_testable"
        reason = "No valid A-Z labelled rows were found in the scanned manifests."
    else:
        status = "not_testable"
        reason = (
            "No labelled recording_id/label group has at least two distinct frame_index "
            "values. Current manifests do not support an ordered temporal experiment."
        )

    return {
        "diagnostic_version": 1,
        "status": status,
        "reason": reason,
        "frame_index_field": FRAME_INDEX_FIELD,
        "scanned_files": scanned_files,
        "files": [str(path) for path in paths],
        "total_rows": total_rows,
        "labelled_rows": labelled_rows,
        "invalid_or_missing_labels": invalid_labels,
        "rows_with_recording_id": rows_with_recording,
        "rows_with_frame_index": rows_with_frame_index,
        "rows_with_recording_and_frame_index": eligible_rows,
        "rows_missing_recording_id": missing_recording_rows,
        "rows_missing_frame_index": missing_frame_index_rows,
        "ordered_sequence_group_count": len(ordered_groups),
        "ordered_sequences": ordered_groups,
        "singleton_group_count": len(singleton_groups),
        "duplicate_frame_index_group_count": len(duplicate_groups),
        "duplicate_frame_index_groups": duplicate_groups,
        "parse_errors": parse_errors,
        "temporal_model_training_allowed": sequences_found,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether labelled JSONL manifests contain recording_id + frame_index sequences."
    )
    parser.add_argument(
        "--manifest-root",
        required=True,
        type=Path,
        help="Root to scan recursively for manifest JSONL files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to persist the diagnostic report outside the source dataset.",
    )
    args = parser.parse_args()

    paths = sorted(args.manifest_root.rglob("*manifest*.jsonl")) if args.manifest_root.exists() else []
    report = diagnose_manifests(paths)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
