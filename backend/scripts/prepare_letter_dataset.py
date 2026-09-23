"""Prepare ISL alphabet images as normalized two-hand landmark arrays."""
from __future__ import annotations

import argparse
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
    landmarker,
) -> tuple[np.ndarray, np.ndarray, dict]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    counts = {letter: 0 for letter in LABELS}
    skipped = 0

    for folder in sorted(path for path in split_root.iterdir() if path.is_dir()):
        letter = folder.name.strip().upper()
        if letter not in LABELS:
            continue

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            vector = extract_landmarks(image_path, landmarker)
            if vector is None:
                skipped += 1
                continue

            features.append(vector)
            labels.append(LABELS.index(letter))
            counts[letter] += 1

    if not features:
        raise RuntimeError(f"No usable samples in {split_root}")

    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        {"counts": counts, "skipped": skipped},
    )


def make_train_validation_split(
    features: np.ndarray,
    labels: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for label_index, letter in enumerate(LABELS):
        indices = np.flatnonzero(labels == label_index)
        if len(indices) < 2:
            raise ValueError(
                f"Class {letter} needs at least two usable samples"
            )

        rng.shuffle(indices)
        validation_count = max(
            1,
            int(round(len(indices) * validation_ratio)),
        )
        validation_count = min(validation_count, len(indices) - 1)

        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return (
        features[train_indices],
        labels[train_indices],
        features[validation_indices],
        labels[validation_indices],
    )


def split_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        letter: int((labels == index).sum())
        for index, letter in enumerate(LABELS)
    }


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
        hand_model_path = Path.home() / ".cache" / "visionbridge" / "hand_landmarker.task"

    landmarker = create_hand_landmarker(hand_model_path)
    try:
        source_train_root = find_split(input_root, SPLIT_NAMES["train"])
        source_val_root = find_split(input_root, SPLIT_NAMES["val"])
        source_test_root = find_split(input_root, SPLIT_NAMES["test"])

        train_source_x, train_source_y, train_report = process_split(
            source_train_root,
            landmarker,
        )
        validation_source_x, validation_source_y, validation_report = process_split(
            source_val_root,
            landmarker,
        )
        test_x, test_y, test_report = process_split(
            source_test_root,
            landmarker,
        )
    finally:
        landmarker.close()

    pooled_x = np.concatenate([train_source_x, validation_source_x])
    pooled_y = np.concatenate([train_source_y, validation_source_y])

    train_x, train_y, val_x, val_y = make_train_validation_split(
        pooled_x,
        pooled_y,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    np.savez_compressed(output_dir / "train.npz", x=train_x, y=train_y)
    np.savez_compressed(output_dir / "val.npz", x=val_x, y=val_y)
    np.savez_compressed(output_dir / "test.npz", x=test_x, y=test_y)

    metadata = {
        "labels": list(LABELS),
        "landmarker": {
            "api": "mediapipe.tasks.vision.HandLandmarker",
            "model_url": HAND_LANDMARKER_URL,
            "num_hands": 2,
        },
        "split_policy": {
            "train_validation_ratio": 1.0 - validation_ratio,
            "validation_ratio": validation_ratio,
            "source_train_and_validation_pooled": True,
            "test_source_left_untouched": True,
            "seed": seed,
        },
        "stats": {
            "source_training": train_report,
            "source_validation": validation_report,
            "train": {"counts": split_counts(train_y)},
            "val": {"counts": split_counts(val_y)},
            "test": test_report,
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
