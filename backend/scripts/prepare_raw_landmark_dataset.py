"""Extract raw, pre-normalization MediaPipe coordinates aligned to existing V1 splits.

By default only train and validation are read. The test split must be explicitly
requested after candidate selection, immediately before the one final test run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from app.services.letter_fewshot import normalize_hand_pair
from scripts.prepare_letter_dataset import (
    HAND_LANDMARKER_SHA256,
    _sha256_file,
    create_hand_landmarker,
)

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPLITS = ("train", "val", "test")
RAW_PREPROCESSING_VERSION = "raw-mediapipe-xyz-before-normalization-v1"
LANDMARK_RUNTIME = "mediapipe-hand-landmarker-0.10.35"
RAW_DIM = 126


def _usable_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: manifest row must be an object")
            if row.get("landmark_status") not in (None, "ok"):
                continue
            label = row.get("label", row.get("letter"))
            source_path = row.get("source_path")
            image_hash = row.get("image_sha256")
            if label not in LABELS:
                raise ValueError(f"{path}:{line_number}: label must be an uppercase A-Z letter")
            if not isinstance(source_path, str) or not source_path:
                raise ValueError(f"{path}:{line_number}: source_path is required")
            if not isinstance(image_hash, str) or len(image_hash) != 64:
                raise ValueError(f"{path}:{line_number}: image_sha256 is required")
            rows.append({**row, "_label": label})
    return rows


def _extract_raw_hand_pair(image_path: Path, landmarker) -> tuple[np.ndarray, bool, bool]:
    """Return raw xyz slots and explicit MediaPipe handedness flags, before any normalization."""
    import mediapipe as mp

    try:
        result = landmarker.detect(mp.Image.create_from_file(str(image_path)))
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"image_or_detection_error:{type(exc).__name__}") from exc

    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    left_present = False
    right_present = False
    low_confidence = False
    for landmarks, handedness in zip(result.hand_landmarks or [], result.handedness or []):
        if not handedness:
            continue
        category = handedness[0]
        confidence = getattr(category, "score", None)
        if confidence is not None and float(confidence) < 0.5:
            low_confidence = True
            continue
        label = str(getattr(category, "category_name", "")).strip().lower()
        if label not in {"left", "right"}:
            continue
        values = np.asarray([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32).reshape(-1)
        if values.shape != (63,) or not np.isfinite(values).all():
            raise RuntimeError("invalid_or_non_finite_landmark_vector")
        if label == "left":
            left[:] = values
            left_present = True
        else:
            right[:] = values
            right_present = True

    if not left_present and not right_present:
        reason = "handedness_confidence_below_0.5" if low_confidence else "no_recognized_hand"
        raise RuntimeError(reason)
    raw = np.concatenate((left, right)).astype(np.float32, copy=False)
    if raw.shape != (RAW_DIM,) or not np.isfinite(raw).all():
        raise RuntimeError("invalid_raw_landmark_vector")
    # Catch obvious extractor/data mismatch before storing raw samples.
    if np.allclose(normalize_hand_pair(raw), 0.0):
        raise RuntimeError("no_usable_hand_after_normalization")
    return raw, left_present, right_present


def _identity(row: dict) -> tuple:
    return (
        row.get("sample_id"),
        row.get("source_path"),
        row.get("image_sha256"),
        row.get("label", row.get("letter", row.get("_label"))),
    )


def prepare_raw_landmark_dataset(
    *,
    input_root: Path,
    v1_data_dir: Path,
    output_dir: Path,
    splits: tuple[str, ...] = ("train", "val"),
    hand_model_path: Path | None = None,
) -> dict:
    if not splits or len(set(splits)) != len(splits) or any(split not in SPLITS for split in splits):
        raise ValueError(f"splits must be unique values from {SPLITS}")
    input_root = input_root.resolve()
    v1_data_dir = v1_data_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir in {input_root, v1_data_dir}:
        raise ValueError("Raw output directory must be separate from raw images and V1 features")

    metadata_path = v1_data_dir / "labels.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing V1 labels.json: {metadata_path}")
    v1_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    runtime_meta = v1_meta.get("landmarker")
    runtime = runtime_meta.get("runtime") if isinstance(runtime_meta, dict) else None
    if (
        v1_meta.get("labels") != list(LABELS)
        or v1_meta.get("preprocessing_version") != "two-hand-wrist-scale-v1"
        or runtime != LANDMARK_RUNTIME
    ):
        raise ValueError("Raw extraction requires the documented V1 label/preprocessing/runtime contract")

    output_dir.mkdir(parents=True, exist_ok=True)
    landmarker = create_hand_landmarker(hand_model_path) if hand_model_path is not None else create_hand_landmarker(
        Path.home() / ".cache" / "visionbridge" / "hand_landmarker.task"
    )
    manifest_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    try:
        for split in splits:
            manifest_path = v1_data_dir / f"{split}_manifest.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing {split} manifest: {manifest_path}")
            manifest_bytes = manifest_path.read_bytes()
            manifest_hashes[split] = hashlib.sha256(manifest_bytes).hexdigest()
            rows = _usable_rows(manifest_path)

            v1_path = v1_data_dir / f"{split}.npz"
            with np.load(v1_path, allow_pickle=False) as prepared:
                normalized_x = np.asarray(prepared["x"], dtype=np.float32)
                labels_y = np.asarray(prepared["y"], dtype=np.int64)
            if (
                normalized_x.ndim != 2
                or normalized_x.shape[1] != RAW_DIM
                or not np.isfinite(normalized_x).all()
                or labels_y.ndim != 1
                or len(normalized_x) != len(labels_y)
                or len(rows) != len(labels_y)
            ):
                raise ValueError(f"{split}: prepared V1 arrays and manifest rows are not aligned")
            expected_y = np.asarray([LABELS.index(row["_label"]) for row in rows], dtype=np.int64)
            if not np.array_equal(expected_y, labels_y):
                raise ValueError(f"{split}: manifest labels/order differ from prepared V1 labels")

            raw_rows: list[np.ndarray] = []
            left_flags: list[bool] = []
            right_flags: list[bool] = []
            for row in rows:
                relative_path = Path(row["source_path"])
                if relative_path.is_absolute():
                    raise ValueError(f"source_path must be relative: {relative_path}")
                image_path = (input_root / relative_path).resolve()
                try:
                    image_path.relative_to(input_root)
                except ValueError as exc:
                    raise ValueError(f"source_path escapes input root: {relative_path}") from exc
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing source image: {image_path}")
                if _sha256_file(image_path).lower() != row["image_sha256"].lower():
                    raise RuntimeError(f"Source image SHA-256 mismatch: {row['source_path']}")
                try:
                    raw, left_present, right_present = _extract_raw_hand_pair(image_path, landmarker)
                except Exception as exc:
                    raise RuntimeError(
                        f"{split}: raw extraction failed for {row.get('sample_id', row['source_path'])}: {exc}"
                    ) from exc
                if raw.shape != (RAW_DIM,) or not np.isfinite(raw).all():
                    raise RuntimeError(f"{split}: extracted raw vector violates the 126D contract")
                # Verify raw extraction still normalizes successfully, but save raw coordinates.
                try:
                    normalize_hand_pair(raw)
                except ValueError as exc:
                    raise RuntimeError(f"{split}: extracted raw coordinates cannot be normalized") from exc
                raw_rows.append(raw)
                left_flags.append(left_present)
                right_flags.append(right_present)

            raw_x = np.stack(raw_rows).astype(np.float32, copy=False)
            if raw_x.shape != (len(labels_y), RAW_DIM):
                raise RuntimeError(f"{split}: raw row count changed during extraction")
            np.savez_compressed(
                output_dir / f"{split}_raw.npz",
                x=raw_x,
                left_present=np.asarray(left_flags, dtype=np.bool_),
                right_present=np.asarray(right_flags, dtype=np.bool_),
                y=labels_y,
            )
            with (output_dir / f"{split}_manifest.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps({key: value for key, value in row.items() if key != "_label"}, sort_keys=True) + "\n")
            counts[split] = len(labels_y)
            print(f"{split}: raw_shape={raw_x.shape} aligned_v1_labels=true")
    finally:
        landmarker.close()

    report = {
        "labels": list(LABELS),
        "raw_landmark_contract": RAW_PREPROCESSING_VERSION,
        "source_preprocessing_version": "two-hand-wrist-scale-v1",
        "landmark_runtime": LANDMARK_RUNTIME,
        "landmarker_model_sha256": HAND_LANDMARKER_SHA256,
        "input_root": str(input_root),
        "source_v1_data_dir": str(v1_data_dir),
        "source_split_manifest_sha256": manifest_hashes,
        "samples_by_split": counts,
        "omitted_splits": [split for split in SPLITS if split not in splits],
        "test_split_processed": "test" in splits,
    }
    (output_dir / "labels.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("raw_pre_normalization_extraction=PASS; test was not read unless explicitly requested")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract raw landmarks aligned with V1 manifest rows.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--data-dir-v1", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "val"])
    parser.add_argument("--hand-model-path", type=Path, default=None)
    args = parser.parse_args()
    prepare_raw_landmark_dataset(
        input_root=args.input_root,
        v1_data_dir=args.data_dir_v1,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        hand_model_path=args.hand_model_path,
    )


if __name__ == "__main__":
    main()
