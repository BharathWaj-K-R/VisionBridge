"""Prepare ISL alphabet images as normalized two-hand landmark arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from app.services.letter_fewshot import normalize_hand_pair

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
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


def extract_landmarks(image: np.ndarray, hands) -> np.ndarray | None:
    result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    for landmarks, handedness in zip(
        result.multi_hand_landmarks or [],
        result.multi_handedness or [],
    ):
        label = handedness.classification[0].label.lower()
        values = np.asarray(
            [[point.x, point.y, point.z] for point in landmarks.landmark],
            dtype=np.float32,
        ).reshape(-1)

        if values.size != 63:
            continue

        if label == "left":
            left[:] = values
        elif label == "right":
            right[:] = values

    try:
        return normalize_hand_pair(np.concatenate([left, right]).tolist())
    except ValueError:
        return None


def process_split(split_root: Path, hands) -> tuple[np.ndarray, np.ndarray, dict]:
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

            image = cv2.imread(str(image_path))
            if image is None:
                skipped += 1
                continue

            vector = extract_landmarks(image, hands)
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

    for label_index in range(len(LABELS)):
        indices = np.flatnonzero(labels == label_index)
        if len(indices) < 2:
            raise ValueError(
                f"Class {LABELS[label_index]} needs at least two usable samples"
            )

        rng.shuffle(indices)
        validation_count = max(1, int(round(len(indices) * validation_ratio)))
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
    return parser.parse_args()


def prepare_dataset(
    input_root: Path,
    output_dir: Path,
    validation_ratio: float = 0.20,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands:
        source_train_root = find_split(input_root, SPLIT_NAMES["train"])
        source_val_root = find_split(input_root, SPLIT_NAMES["val"])
        source_test_root = find_split(input_root, SPLIT_NAMES["test"])

        train_source_x, train_source_y, train_report = process_split(
            source_train_root,
            hands,
        )
        validation_source_x, validation_source_y, validation_report = process_split(
            source_val_root,
            hands,
        )
        test_x, test_y, test_report = process_split(
            source_test_root,
            hands,
        )

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
    )


if __name__ == "__main__":
    main()
