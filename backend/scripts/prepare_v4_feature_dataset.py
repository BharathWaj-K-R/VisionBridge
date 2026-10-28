"""Prepare V4 141D arrays over the exact manifests used for the V1 splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from app.services.letter_fewshot import normalize_hand_pair_v2
from scripts.prepare_letter_dataset import _sha256_file, create_hand_landmarker

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPLITS = ("train", "val", "test")
LANDMARK_MODEL_DEFAULT = Path.home() / ".cache" / "visionbridge" / "hand_landmarker.task"


def _read_split_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing split manifest required for aligned V4 features: {path}")
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            # Phase 1 schema-v2 manifests retain failed rows for auditing. V1 NPZ
            # arrays contain only successful rows, so preserve that same order.
            status = row.get("landmark_status")
            if status is not None and status != "ok":
                continue
            label = row.get("label", row.get("letter"))
            if label not in LABELS:
                raise ValueError(f"{path}:{line_number}: label must be one uppercase A-Z letter")
            source_path = row.get("source_path")
            image_hash = row.get("image_sha256")
            if not isinstance(source_path, str) or not source_path:
                raise ValueError(f"{path}:{line_number}: source_path is missing")
            if not isinstance(image_hash, str) or len(image_hash) != 64:
                raise ValueError(f"{path}:{line_number}: image_sha256 is missing")
            identity = (row.get("sample_id"), source_path, image_hash, label)
            if identity in seen:
                raise ValueError(f"{path}:{line_number}: duplicate manifest sample identity")
            seen.add(identity)
            row["_v4_label"] = label
            rows.append(row)
    return rows


def _extract_v4_vector(image_path: Path, landmarker) -> tuple[np.ndarray, str | None, str | None]:
    import mediapipe as mp

    image = mp.Image.create_from_file(str(image_path))
    result = landmarker.detect(image)
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    left_present = False
    right_present = False

    for landmarks, handedness in zip(result.hand_landmarks or [], result.handedness or []):
        if not handedness:
            continue
        label = str(handedness[0].category_name).strip().lower()
        values = np.asarray(
            [[point.x, point.y, point.z] for point in landmarks],
            dtype=np.float32,
        ).reshape(-1)
        if values.size != 63 or not np.isfinite(values).all():
            raise ValueError("invalid_or_non_finite_landmark_vector")
        if label == "left":
            left[:] = values
            left_present = True
        elif label == "right":
            right[:] = values
            right_present = True

    if not left_present and not right_present:
        return np.empty(0, dtype=np.float32), "no_hand", "no_hand_detected"
    vector = normalize_hand_pair_v2(
        np.concatenate((left, right)).tolist(),
        left_present=left_present,
        right_present=right_present,
    )
    return vector, None, None


def prepare_v4_dataset(
    *,
    input_root: Path,
    v1_data_dir: Path,
    output_dir: Path,
    hand_model_path: Path = LANDMARK_MODEL_DEFAULT,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = input_root.resolve()
    manifest_hashes: dict[str, str] = {}
    landmarker = create_hand_landmarker(hand_model_path)
    metadata: dict[str, object] = {
        "labels": list(LABELS),
        "model_version": "visionbridge-letter-base-v4",
        "preprocessing_version": "two-hand-geometry-v2",
        "feature_layout": "two-hand-geometry-v2:141:left64,right64,relative13",
        "input_dim": 141,
        "source_v1_data_dir": str(v1_data_dir),
        "split_source": "existing V1 split manifests; sample order must match",
    }
    try:
        for split in SPLITS:
            manifest_path = v1_data_dir / f"{split}_manifest.jsonl"
            manifest_bytes = manifest_path.read_bytes()
            manifest_hashes[split] = hashlib.sha256(manifest_bytes).hexdigest()
            rows = _read_split_manifest(manifest_path)
            v1_npz = np.load(v1_data_dir / f"{split}.npz", allow_pickle=False)
            v1_y = np.asarray(v1_npz["y"], dtype=np.int64)
            expected_y = np.asarray([LABELS.index(row["_v4_label"]) for row in rows], dtype=np.int64)
            if len(rows) != len(v1_y) or not np.array_equal(expected_y, v1_y):
                raise RuntimeError(
                    f"{split}: successful manifest rows do not align with V1 NPZ labels; "
                    "refusing a non-identical comparison split"
                )

            features = []
            output_rows = []
            for row in rows:
                relative = Path(row["source_path"])
                if relative.is_absolute():
                    raise ValueError(f"Manifest source_path must be relative: {row['source_path']}")
                image_path = (input_root / relative).resolve()
                try:
                    image_path.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"Manifest source_path escapes input root: {row['source_path']}") from exc
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing source image: {image_path}")
                actual_hash = _sha256_file(image_path)
                if actual_hash.lower() != row["image_sha256"].lower():
                    raise RuntimeError(f"Image SHA-256 mismatch for {row['source_path']}")

                try:
                    vector, status, reason = _extract_v4_vector(image_path, landmarker)
                except Exception as exc:
                    raise RuntimeError(
                        f"{split}: V4 landmark extraction failed for sample "
                        f"{row.get('sample_id', row['source_path'])}: {type(exc).__name__}"
                    ) from exc
                if status is not None:
                    raise RuntimeError(
                        f"{split}: V1 successful sample became {status}: "
                        f"{row.get('sample_id', row['source_path'])}; "
                        "the comparison split would no longer match"
                    )
                if vector.shape != (141,):
                    raise RuntimeError(f"{split}: invalid V4 feature shape {vector.shape}")
                features.append(vector)
                saved_row = {key: value for key, value in row.items() if key != "_v4_label"}
                saved_row["landmark_status"] = "ok"
                saved_row["failure_reason"] = None
                output_rows.append(saved_row)

            x = np.stack(features).astype(np.float32, copy=False)
            if x.shape != (len(v1_y), 141) or not np.isfinite(x).all():
                raise RuntimeError(f"{split}: V4 feature array shape/finiteness check failed")
            np.savez_compressed(output_dir / f"{split}.npz", x=x, y=v1_y)
            with (output_dir / f"{split}_manifest.jsonl").open("w", encoding="utf-8") as handle:
                for row in output_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"{split}: samples={len(x)} shape={x.shape} labels_match_v1=true")
    finally:
        landmarker.close()

    metadata["source_split_manifest_sha256"] = manifest_hashes
    metadata["samples_by_split"] = {
        split: int(np.load(output_dir / f"{split}.npz", allow_pickle=False)["y"].shape[0])
        for split in SPLITS
    }
    (output_dir / "labels.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    failure_report = v1_data_dir / "extraction_failure_report.json"
    if failure_report.is_file():
        shutil.copy2(failure_report, output_dir / failure_report.name)
    print("v4_feature_preparation=PASS; no V3 checkpoint or model code was changed")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create aligned V4 141D features from existing V1 split manifests.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--data-dir-v1", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hand-model-path", type=Path, default=LANDMARK_MODEL_DEFAULT)
    args = parser.parse_args()
    prepare_v4_dataset(
        input_root=args.input_root,
        v1_data_dir=args.data_dir_v1,
        output_dir=args.output_dir,
        hand_model_path=args.hand_model_path,
    )


if __name__ == "__main__":
    main()
