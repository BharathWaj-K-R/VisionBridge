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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ISL A-Z image landmarks.")
    parser.add_argument("--input-root", required=True, help="Root of the downloaded alphabet dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared NPZ files.")
    return parser.parse_args()


def prepare_dataset(input_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands:
        for split_name, candidate_names in SPLIT_NAMES.items():
            split_root = find_split(input_root, candidate_names)
            features, labels, report = process_split(split_root, hands)
            np.savez_compressed(
                output_dir / f"{split_name}.npz",
                x=features,
                y=labels,
            )
            stats[split_name] = report
            print(
                f"{split_name}: samples={len(features)} skipped={report['skipped']}"
            )

    metadata = {"labels": list(LABELS), "stats": stats}
    (output_dir / "labels.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    prepare_dataset(Path(args.input_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
