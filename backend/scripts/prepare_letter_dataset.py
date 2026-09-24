"""Prepare ISL alphabet images as normalized two-hand landmark arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from app.services.letter_fewshot import normalize_hand_pair

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
SPLIT_NAMES = {
    "train": ("Training (A-Z)", "Training"),
    "val": ("Validation (A-Z)", "Validation", "Val"),
    "test": ("Testing (A-Z)", "Testing", "Test"),
}


def find_split(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        direct = root / name
        if direct.is_dir():
            return direct

    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name in names:
            return candidate

    raise FileNotFoundError(f"Could not find dataset split: {names}")


def ensure_hand_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path

    print(f"Downloading MediaPipe Hand Landmarker to {model_path}")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)

    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise RuntimeError("MediaPipe Hand Landmarker download failed")

    return model_path


def create_hand_landmarker(model_path: Path):
    base_options = python.BaseOptions(
        model_asset_path=str(ensure_hand_model(model_path))
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_landmarks(image_path: Path, landmarker) -> np.ndarray | None:
    try:
        image = mp.Image.create_from_file(str(image_path))
        result = landmarker.detect(image)
    except (OSError, RuntimeError, ValueError):
        return None

    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    for landmarks, handedness in zip(
        result.hand_landmarks,
        result.handedness,
    ):
        values = np.asarray(
            [[point.x, point.y, point.z] for point in landmarks],
            dtype=np.float32,
        ).reshape(-1)

        if values.size != 63 or not np.isfinite(values).all():
            continue

        label = handedness[0].category_name.strip().lower()
        if label == "left":
            left[:] = values
        elif label == "right":
            right[:] = values

    try:
        return normalize_hand_pair(
            np.concatenate([left, right]).tolist()
        )
    except ValueError:
        return None


def process_split(
    split_root: Path,
    input_root: Path,
    split_name: str,
    landmarker,
) -> tuple[np.ndarray, np.ndarray, dict, list[dict[str, str]]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    samples: list[dict[str, str]] = []
    counts = {letter: 0 for letter in LABELS}
    skipped = 0

    for folder in sorted(path for path in split_root.iterdir() if path.is_dir()):
        letter = folder.name.strip().upper()
        if letter not in LABELS:
            continue

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image_sha256 = _sha256_file(image_path)
            vector = extract_landmarks(image_path, landmarker)
            if vector is None:
                skipped += 1
                continue

            relative_path = image_path.relative_to(input_root).as_posix()
            features.append(vector)
            labels.append(LABELS.index(letter))
            samples.append(
                {
                    "source_path": relative_path,
                    "source_split": split_name,
                    "letter": letter,
                    "image_sha256": image_sha256,
                }
            )
            counts[letter] += 1

    if not features:
        raise RuntimeError(f"No usable samples in {split_root}")

    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        {"counts": counts, "skipped": skipped},
        samples,
    )


def make_train_validation_split(
    features: np.ndarray,
    labels: np.ndarray,
    samples: list[dict[str, str]],
    validation_ratio: float,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, str]],
    np.ndarray,
    np.ndarray,
    list[dict[str, str]],
]:
    if not 0 < validation_ratio < 1:
        raise ValueError("Validation ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for label_index, letter in enumerate(LABELS):
        indices = np.flatnonzero(labels == label_index)

        if len(indices) < 2:
            raise ValueError(
                f"Class {letter} needs at least two usable samples"
            )

        groups: dict[str, list[int]] = {}
        for index in indices:
            image_hash = samples[int(index)]["image_sha256"]
            groups.setdefault(image_hash, []).append(int(index))

        if len(groups) < 2:
            raise ValueError(
                f"Class {letter} has fewer than two unique images; "
                "an honest train/validation split is not possible"
            )

        group_list = list(groups.values())
        rng.shuffle(group_list)

        target_validation = max(
            1,
            int(round(len(indices) * validation_ratio)),
        )
        selected_validation: list[int] = []
        selected_count = 0

        for group in group_list:
            remaining_if_skipped = abs(target_validation - selected_count)
            remaining_if_selected = abs(
                target_validation - (selected_count + len(group))
            )

            if selected_count < target_validation or (
                remaining_if_selected < remaining_if_skipped
            ):
                selected_validation.extend(group)
                selected_count += len(group)

        validation_indices.extend(selected_validation)
        validation_set = set(selected_validation)
        train_indices.extend(
            int(index)
            for index in indices
            if int(index) not in validation_set
        )

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return (
        features[train_indices],
        labels[train_indices],
        [samples[index] for index in train_indices],
        features[validation_indices],
        labels[validation_indices],
        [samples[index] for index in validation_indices],
    )


def split_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        letter: int((labels == index).sum())
        for index, letter in enumerate(LABELS)
    }


def find_duplicate_hashes(samples: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = sample["image_sha256"]
        counts[key] = counts.get(key, 0) + 1
    return {
        image_hash: count
        for image_hash, count in counts.items()
        if count > 1
    }


def write_manifest(output_dir: Path, samples: list[dict[str, str]], split: str) -> None:
    path = output_dir / f"{split}_manifest.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = dict(sample)
            record["prepared_split"] = split
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ISL A-Z image landmarks."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root of the downloaded alphabet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for prepared NPZ files.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.20,
        help="Validation share from the combined training and validation pools.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the stratified train/validation split.",
    )
    parser.add_argument(
        "--hand-model-path",
        default=None,
        help="Path for the MediaPipe Hand Landmarker .task model.",
    )
    return parser.parse_args()


def prepare_dataset(
    input_root: Path,
    output_dir: Path,
    validation_ratio: float = 0.20,
    seed: int = 42,
    hand_model_path: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if hand_model_path is None:
        hand_model_path = (
            Path.home() / ".cache" / "visionbridge" / "hand_landmarker.task"
        )

    landmarker = create_hand_landmarker(hand_model_path)
    try:
        source_train_root = find_split(input_root, SPLIT_NAMES["train"])
        source_val_root = find_split(input_root, SPLIT_NAMES["val"])
        source_test_root = find_split(input_root, SPLIT_NAMES["test"])

        train_source_x, train_source_y, train_report, train_samples = process_split(
            source_train_root,
            input_root,
            "training",
            landmarker,
        )
        validation_source_x, validation_source_y, validation_report, validation_samples = process_split(
            source_val_root,
            input_root,
            "validation",
            landmarker,
        )
        test_x, test_y, test_report, test_samples = process_split(
            source_test_root,
            input_root,
            "testing",
            landmarker,
        )
    finally:
        landmarker.close()

    pooled_x = np.concatenate([train_source_x, validation_source_x])
    pooled_y = np.concatenate([train_source_y, validation_source_y])
    pooled_samples = train_samples + validation_samples

    train_x, train_y, train_manifest, val_x, val_y, val_manifest = (
        make_train_validation_split(
            pooled_x,
            pooled_y,
            pooled_samples,
            validation_ratio=validation_ratio,
            seed=seed,
        )
    )

    pool_hashes = {sample["image_sha256"] for sample in pooled_samples}
    test_overlap = {
        sample["image_sha256"]
        for sample in test_samples
        if sample["image_sha256"] in pool_hashes
    }

    if test_overlap:
        print(
            "WARNING: exact image duplicates occur between the source test split "
            f"and train/validation pools: {len(test_overlap)} hash(es). "
            "The test set was not modified."
        )

    np.savez_compressed(output_dir / "train.npz", x=train_x, y=train_y)
    np.savez_compressed(output_dir / "val.npz", x=val_x, y=val_y)
    np.savez_compressed(output_dir / "test.npz", x=test_x, y=test_y)

    write_manifest(output_dir, train_manifest, "train")
    write_manifest(output_dir, val_manifest, "val")
    write_manifest(output_dir, test_samples, "test")

    duplicate_report = {
        "train_duplicate_image_hashes": find_duplicate_hashes(train_manifest),
        "val_duplicate_image_hashes": find_duplicate_hashes(val_manifest),
        "test_duplicate_image_hashes": find_duplicate_hashes(test_samples),
        "train_val_overlap_hashes": sorted(
            set(sample["image_sha256"] for sample in train_manifest)
            & set(sample["image_sha256"] for sample in val_manifest)
        ),
        "train_test_overlap_hashes": sorted(
            set(sample["image_sha256"] for sample in train_manifest)
            & set(sample["image_sha256"] for sample in test_samples)
        ),
        "val_test_overlap_hashes": sorted(
            set(sample["image_sha256"] for sample in val_manifest)
            & set(sample["image_sha256"] for sample in test_samples)
        ),
    }
    (output_dir / "duplicate_report.json").write_text(
        json.dumps(duplicate_report, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "labels": list(LABELS),
        "preprocessing_version": "two-hand-wrist-scale-v1",
        "landmarker": {
            "api": "mediapipe.tasks.vision.HandLandmarker",
            "runtime": "mediapipe-hand-landmarker-0.10.35",
            "model_url": HAND_LANDMARKER_URL,
            "num_hands": 2,
        },
        "split_policy": {
            "train_validation_ratio": 1.0 - validation_ratio,
            "validation_ratio": validation_ratio,
            "source_train_and_validation_pooled": True,
            "test_source_left_untouched": True,
            "exact_image_duplicates_grouped_within_train_validation": True,
            "seed": seed,
        },
        "stats": {
            "source_training": train_report,
            "source_validation": validation_report,
            "train": {"counts": split_counts(train_y)},
            "val": {"counts": split_counts(val_y)},
            "test": test_report,
            "duplicate_report": {
                "train_val_overlap_count": len(
                    duplicate_report["train_val_overlap_hashes"]
                ),
                "train_test_overlap_count": len(
                    duplicate_report["train_test_overlap_hashes"]
                ),
                "val_test_overlap_count": len(
                    duplicate_report["val_test_overlap_hashes"]
                ),
            },
        },
    }
    (output_dir / "labels.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"train: samples={len(train_x)} validation={1.0 - validation_ratio:.0%}"
    )
    print(
        f"val: samples={len(val_x)} validation={validation_ratio:.0%}"
    )
    print(f"test: samples={len(test_x)} source=untouched")
    print(f"train_per_letter={split_counts(train_y)}")
    print(f"val_per_letter={split_counts(val_y)}")
    print(
        "exact_duplicate_overlap: "
        f"train/val={len(duplicate_report['train_val_overlap_hashes'])} "
        f"train/test={len(duplicate_report['train_test_overlap_hashes'])} "
        f"val/test={len(duplicate_report['val_test_overlap_hashes'])}"
    )


def main() -> None:
    args = parse_args()
    prepare_dataset(
        Path(args.input_root),
        Path(args.output_dir),
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        hand_model_path=Path(args.hand_model_path)
        if args.hand_model_path
        else None,
    )


if __name__ == "__main__":
    main()
