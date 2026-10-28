"""Prepare ISL alphabet images as normalized two-hand landmark arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from app.services.letter_fewshot import normalize_hand_pair
from scripts.validate_manifest import CAPTURE_FIELDS, load_manifest

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_SIZE = 7_819_105
HAND_LANDMARKER_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
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

    if model_path.is_file():
        if (
            model_path.stat().st_size == HAND_LANDMARKER_SIZE
            and _sha256_file(model_path) == HAND_LANDMARKER_SHA256
        ):
            return model_path
        print("Cached MediaPipe Hand Landmarker does not match the pinned asset; redownloading.")
        model_path.unlink()

    print(f"Downloading MediaPipe Hand Landmarker to {model_path}")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)

    if (
        not model_path.is_file()
        or model_path.stat().st_size != HAND_LANDMARKER_SIZE
        or _sha256_file(model_path) != HAND_LANDMARKER_SHA256
    ):
        raise RuntimeError("MediaPipe Hand Landmarker download failed integrity validation")

    return model_path

def create_hand_landmarker(model_path: Path):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

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


def extract_landmarks_with_status(
    image_path: Path,
    landmarker,
) -> tuple[np.ndarray | None, str, str | None]:
    """Extract the v1 feature vector and preserve a useful failure category."""
    import mediapipe as mp

    try:
        image = mp.Image.create_from_file(str(image_path))
        result = landmarker.detect(image)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, "failed", f"image_or_detection_error:{type(exc).__name__}"

    hands = result.hand_landmarks or []
    handedness_rows = result.handedness or []
    if not hands:
        return None, "no_hand", "no_hand_detected"

    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    low_confidence = False
    recognized_hand = False

    for landmarks, handedness in zip(hands, handedness_rows):
        if not handedness:
            continue
        category = handedness[0]
        score = getattr(category, "score", None)
        if score is not None and float(score) < 0.5:
            low_confidence = True
            continue

        label = str(getattr(category, "category_name", "")).strip().lower()
        if label not in {"left", "right"}:
            continue

        values = np.asarray(
            [[point.x, point.y, point.z] for point in landmarks],
            dtype=np.float32,
        ).reshape(-1)
        if values.size != 63 or not np.isfinite(values).all():
            return None, "failed", "invalid_or_non_finite_landmark_vector"

        recognized_hand = True
        if label == "left":
            left[:] = values
        else:
            right[:] = values

    if low_confidence and not recognized_hand:
        return None, "low_conf", "handedness_confidence_below_0.5"
    if not recognized_hand:
        return None, "failed", "handedness_missing_or_unrecognized"

    try:
        vector = normalize_hand_pair(np.concatenate([left, right]).tolist())
    except ValueError:
        return None, "failed", "v1_normalization_failed"

    if not np.isfinite(vector).all():
        return None, "failed", "normalized_landmarks_non_finite"
    if not np.any(np.abs(vector) > 0):
        return None, "no_hand", "no_usable_hand_after_normalization"
    return vector, "ok", None


def extract_landmarks(image_path: Path, landmarker) -> np.ndarray | None:
    """Compatibility wrapper for callers that only need the feature vector."""
    vector, _, _ = extract_landmarks_with_status(image_path, landmarker)
    return vector


def _empty_capture_metadata() -> dict[str, object]:
    return {field: None for field in CAPTURE_FIELDS}

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
    attempted = 0
    failures: list[dict] = []

    for folder in sorted(path for path in split_root.iterdir() if path.is_dir()):
        letter = folder.name.strip().upper()
        if letter not in LABELS:
            continue

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            attempted += 1
            image_sha256 = _sha256_file(image_path)
            vector, landmark_status, failure_reason = extract_landmarks_with_status(image_path, landmarker)
            if vector is None:
                skipped += 1
                failures.append({
                    "source_path": image_path.relative_to(input_root).as_posix(),
                    "source_split": split_name,
                    "letter": letter,
                    "image_sha256": image_sha256,
                    "landmark_status": landmark_status,
                    "failure_reason": failure_reason,
                    "signer_id": None,
                    "recording_id": None,
                    "capture_metadata": _empty_capture_metadata(),
                })
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

    feature_matrix = (
        np.stack(features)
        if features
        else np.empty((0, 126), dtype=np.float32)
    )

    return (
        feature_matrix,
        np.asarray(labels, dtype=np.int64),
        {"counts": counts, "attempted": attempted, "skipped": skipped, "failures": failures},
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

    hash_labels: dict[str, int] = {}
    for index, sample in enumerate(samples):
        image_hash = sample["image_sha256"]
        label_index = int(labels[index])
        previous = hash_labels.get(image_hash)
        if previous is not None and previous != label_index:
            raise ValueError(
                "Exact duplicate image hash appears under multiple labels: "
                f"{image_hash}"
            )
        hash_labels[image_hash] = label_index

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

    train_labels = labels[train_indices]
    validation_labels = labels[validation_indices]
    for label_index, letter in enumerate(LABELS):
        if not np.any(train_labels == label_index) or not np.any(validation_labels == label_index):
            raise ValueError(
                f"Class {letter} does not have at least one usable sample in both train and validation"
            )

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


def build_extraction_failure_report(records: list[dict]) -> dict[str, object]:
    """Summarize extraction outcomes separately from classifier accuracy."""
    per_letter: dict[str, dict] = {
        letter: {"attempted": 0, "skipped": 0, "status_counts": {}}
        for letter in LABELS
    }
    per_signer: dict[str, dict] = {}
    per_condition: dict[str, dict[str, dict]] = {
        field: {} for field in CAPTURE_FIELDS
    }
    per_split: dict[str, dict] = {}
    failed_samples: list[dict] = []
    status_totals = {"ok": 0, "no_hand": 0, "low_conf": 0, "failed": 0}

    def bucket(target: dict, key: str) -> dict:
        if key not in target:
            target[key] = {"attempted": 0, "skipped": 0, "status_counts": {}}
        return target[key]

    for record in records:
        letter = record.get("label", record.get("letter"))
        if letter not in LABELS:
            continue
        status = record.get("landmark_status", "ok")
        if status not in status_totals:
            status = "failed"
        skipped = status != "ok"
        status_totals[status] += 1

        split_value = record.get("split", record.get("prepared_split", record.get("source_split")))
        split_aliases = {
            "training": "source_training",
            "source_training": "source_training",
            "validation": "source_validation",
            "source_validation": "source_validation",
            "testing": "test",
            "source_testing": "test",
        }
        split_key = split_aliases.get(split_value, split_value or "<unknown>")
        targets = [per_letter[letter]]
        signer_id = record.get("signer_id") or "<unknown>"
        targets.append(bucket(per_signer, str(signer_id)))
        targets.append(bucket(per_split, str(split_key)))
        for target in targets:
            target["attempted"] += 1
            target["skipped"] += int(skipped)
            target["status_counts"][status] = target["status_counts"].get(status, 0) + 1

        capture = record.get("capture_metadata")
        if not isinstance(capture, dict):
            capture = _empty_capture_metadata()
        for field in CAPTURE_FIELDS:
            value = capture.get(field)
            condition = "<unknown>" if value is None or value == "" else str(value)
            entry = bucket(per_condition[field], condition)
            entry["attempted"] += 1
            entry["skipped"] += int(skipped)
            entry["status_counts"][status] = entry["status_counts"].get(status, 0) + 1

        if skipped:
            failed_samples.append({
                "sample_id": record.get("sample_id"),
                "source_path": record.get("source_path"),
                "label": letter,
                "signer_id": record.get("signer_id"),
                "recording_id": record.get("recording_id"),
                "split": record.get("split", record.get("prepared_split", record.get("source_split"))),
                "landmark_status": status,
                "failure_reason": record.get("failure_reason"),
                "capture_metadata": capture,
            })

    def finalize(counter: dict) -> None:
        attempted = counter["attempted"]
        counter["failure_rate"] = counter["skipped"] / attempted if attempted else 0.0

    for counter in per_letter.values():
        finalize(counter)
    for counter in per_signer.values():
        finalize(counter)
    for counter in per_split.values():
        finalize(counter)
    for condition_buckets in per_condition.values():
        for counter in condition_buckets.values():
            finalize(counter)

    total_attempted = sum(status_totals.values())
    total_skipped = total_attempted - status_totals["ok"]
    return {
        "schema_version": 1,
        "total_attempted": total_attempted,
        "total_successful": status_totals["ok"],
        "total_skipped": total_skipped,
        "failure_rate": total_skipped / total_attempted if total_attempted else 0.0,
        "status_counts": status_totals,
        "per_letter": per_letter,
        "per_signer": per_signer,
        "per_split": per_split,
        "per_condition": per_condition,
        "skipped_samples": failed_samples,
    }


def assign_signer_splits(
    records: list[dict],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> list[dict]:
    """Assign whole signer groups to train/val/test with deterministic label balancing."""
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Signer split ratios must be three positive values summing to 1")

    groups: dict[str, list[dict]] = {}
    for record in records:
        signer_id = record.get("signer_id")
        if not isinstance(signer_id, str) or not signer_id.strip():
            raise ValueError("Signer-aware splitting requires real signer_id values")
        if record.get("split") != "unassigned":
            raise ValueError("Signer splitting requires all manifest rows to have split='unassigned'")
        groups.setdefault(signer_id, []).append(record)

    if len(groups) < 3:
        raise ValueError("At least three distinct signers are required for train/val/test separation")

    split_names = ("train", "val", "test")
    ratio_array = np.asarray(ratios, dtype=np.float64)
    rng = np.random.default_rng(seed)
    tie_breakers = {signer: float(rng.random()) for signer in groups}
    signer_labels = {
        signer: {record["label"] for record in samples}
        for signer, samples in groups.items()
    }
    label_signer_support = {
        letter: sum(letter in covered for covered in signer_labels.values())
        for letter in LABELS
    }
    rarity = {
        signer: sum(1.0 / max(label_signer_support[label], 1) for label in signer_labels[signer])
        for signer in groups
    }
    ordered_signers = sorted(
        groups,
        key=lambda signer: (
            -rarity[signer],
            -len(signer_labels[signer]),
            -len(groups[signer]),
            tie_breakers[signer],
        ),
    )

    target_groups = ratio_array * len(groups)
    target_samples = ratio_array * len(records)
    total_label_samples = np.asarray(
        [sum(record["label"] == letter for record in records) for letter in LABELS],
        dtype=np.float64,
    )
    target_label_samples = ratio_array[:, None] * total_label_samples[None, :]
    current_groups = np.zeros(3, dtype=np.int64)
    current_samples = np.zeros(3, dtype=np.float64)
    current_label_samples = np.zeros((3, len(LABELS)), dtype=np.float64)
    assigned: dict[str, str] = {}

    for index, signer in enumerate(ordered_signers):
        group = groups[signer]
        group_label_counts = np.asarray(
            [sum(record["label"] == letter for record in group) for letter in LABELS],
            dtype=np.float64,
        )
        remaining_after = len(ordered_signers) - index - 1
        empty_splits = [i for i in range(3) if current_groups[i] == 0]
        candidates = empty_splits if remaining_after < len(empty_splits) else list(range(3))

        scored: list[tuple[float, int]] = []
        for split_index in candidates:
            projected_group_count = current_groups[split_index] + 1
            projected_sample_count = current_samples[split_index] + len(group)
            group_error = (
                (projected_group_count - target_groups[split_index]) ** 2
                / max(target_groups[split_index], 1.0)
            )
            sample_error = (
                (projected_sample_count - target_samples[split_index]) ** 2
                / max(target_samples[split_index], 1.0)
            )
            projected_labels = current_label_samples[split_index] + group_label_counts
            label_error = np.sum(
                (projected_labels - target_label_samples[split_index]) ** 2
                / np.maximum(target_label_samples[split_index], 1.0)
            )
            coverage_bonus = sum(
                1
                for label_index, letter in enumerate(LABELS)
                if label_signer_support[letter] >= 3
                and group_label_counts[label_index] > 0
                and current_label_samples[split_index, label_index] == 0
            )
            score = float(sample_error + 0.75 * label_error + 0.2 * group_error - 0.35 * coverage_bonus)
            score += tie_breakers[signer] * 1e-9 + split_index * 1e-12
            scored.append((score, split_index))

        selected = min(scored)[1]
        assigned[signer] = split_names[selected]
        current_groups[selected] += 1
        current_samples[selected] += len(group)
        current_label_samples[selected] += group_label_counts

    return [
        {**record, "split": assigned[record["signer_id"]]}
        for record in records
    ]


def find_near_duplicate_landmarks(
    features: np.ndarray,
    samples: list[dict],
    threshold: float = 0.01,
    max_reported_pairs: int = 250,
) -> dict[str, object]:
    """Find pairs below an RMS Euclidean threshold in normalized 126D landmark space."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Near-duplicate threshold must be finite and non-negative")
    values = np.asarray(features, dtype=np.float64)
    if values.size == 0:
        values = np.empty((0, 126), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 126 or len(values) != len(samples):
        raise ValueError("Near-duplicate inputs must align as [N,126] features and N sample rows")
    if not np.isfinite(values).all():
        raise ValueError("Near-duplicate features contain NaN or Inf")

    within_count = 0
    cross_split_count = 0
    within_examples: list[dict] = []
    cross_split_examples: list[dict] = []
    norms = np.einsum("ij,ij->i", values, values)
    threshold_squared = float(threshold) ** 2 * values.shape[1]
    block_size = 256

    def sample_label(sample: dict) -> str:
        return str(sample.get("sample_id") or sample.get("source_path") or "<unknown>")

    def pair_record(i: int, j: int, distance: float) -> dict:
        left = samples[i]
        right = samples[j]
        return {
            "sample_a": sample_label(left),
            "sample_b": sample_label(right),
            "source_path_a": left.get("source_path"),
            "source_path_b": right.get("source_path"),
            "split_a": left.get("prepared_split", left.get("split", left.get("source_split"))),
            "split_b": right.get("prepared_split", right.get("split", right.get("source_split"))),
            "signer_id_a": left.get("signer_id"),
            "signer_id_b": right.get("signer_id"),
            "recording_id_a": left.get("recording_id"),
            "recording_id_b": right.get("recording_id"),
            "label_a": left.get("label", left.get("letter")),
            "label_b": right.get("label", right.get("letter")),
            "rms_distance": distance,
        }

    for start in range(0, len(values), block_size):
        stop = min(start + block_size, len(values))
        products = values[start:stop] @ values.T
        squared = norms[start:stop, None] + norms[None, :] - 2.0 * products
        np.maximum(squared, 0.0, out=squared)
        for local_index in range(stop - start):
            i = start + local_index
            candidates = np.flatnonzero(
                (squared[local_index] <= threshold_squared)
                & (np.arange(len(values)) > i)
            )
            left_hash = samples[i].get("image_sha256")
            for j in candidates:
                right_hash = samples[int(j)].get("image_sha256")
                if left_hash and right_hash and left_hash == right_hash:
                    continue
                distance = float(np.sqrt(squared[local_index, int(j)] / values.shape[1]))
                record = pair_record(i, int(j), distance)
                if record["split_a"] != record["split_b"]:
                    cross_split_count += 1
                    if len(cross_split_examples) < max_reported_pairs:
                        cross_split_examples.append(record)
                same_signer_recording = (
                    bool(record["signer_id_a"])
                    and bool(record["recording_id_a"])
                    and record["signer_id_a"] == record["signer_id_b"]
                    and record["recording_id_a"] == record["recording_id_b"]
                )
                if same_signer_recording:
                    within_count += 1
                    if len(within_examples) < max_reported_pairs:
                        within_examples.append(record)

    return {
        "metric": "RMS Euclidean distance across the 126D normalized feature vector",
        "threshold": float(threshold),
        "within_recording_pair_count": within_count,
        "within_recording_pairs": within_examples,
        "within_recording_examples_truncated": within_count > len(within_examples),
        "cross_split_pair_count": cross_split_count,
        "cross_split_pairs": cross_split_examples,
        "cross_split_examples_truncated": cross_split_count > len(cross_split_examples),
    }


def enforce_no_cross_split_near_duplicates(report: dict) -> None:
    count = int(report.get("cross_split_pair_count", 0))
    if count:
        raise RuntimeError(
            f"Found {count} cross-split landmark near-duplicate pair(s); "
            "see duplicate_report.json and repair the split before evaluation."
        )


def _exact_cross_split_hashes(records: list[dict]) -> dict[str, list[str]]:
    per_split: dict[str, set[str]] = {split: set() for split in ("train", "val", "test")}
    for record in records:
        split = record.get("split", record.get("prepared_split"))
        image_hash = record.get("image_sha256")
        if split in per_split and image_hash:
            per_split[split].add(image_hash)
    return {
        "train_val_overlap_hashes": sorted(per_split["train"] & per_split["val"]),
        "train_test_overlap_hashes": sorted(per_split["train"] & per_split["test"]),
        "val_test_overlap_hashes": sorted(per_split["val"] & per_split["test"]),
    }


def prepare_manifest_v2_dataset(
    input_root: Path,
    output_dir: Path,
    manifest_path: Path,
    seed: int,
    hand_model_path: Path,
    near_duplicate_threshold: float,
) -> None:
    records = load_manifest(manifest_path)
    input_splits = {record["split"] for record in records}
    if input_splits == {"unassigned"}:
        assigned_records = assign_signer_splits(records, seed=seed)
        signer_split = "generated_by_signer_group_split"
        seed_applied = True
    else:
        assigned_records = [dict(record) for record in records]
        signer_split = "preassigned_manifest_groups"
        seed_applied = False

    root_resolved = input_root.resolve()
    landmarker = create_hand_landmarker(hand_model_path)
    outcomes: list[dict] = []
    successful_features: list[np.ndarray] = []
    successful_labels: list[int] = []
    successful_rows: list[dict] = []
    try:
        for record in assigned_records:
            row = dict(record)
            source_path = Path(row["source_path"])
            if source_path.is_absolute():
                raise ValueError(f"Manifest source_path must be relative: {row['source_path']}")
            image_path = (input_root / source_path).resolve()
            try:
                image_path.relative_to(root_resolved)
            except ValueError as exc:
                raise ValueError(f"Manifest source_path escapes the input root: {row['source_path']}") from exc

            if not image_path.is_file():
                row["landmark_status"] = "failed"
                row["failure_reason"] = "source_file_missing"
                outcomes.append(row)
                continue

            actual_hash = _sha256_file(image_path)
            if actual_hash.lower() != row["image_sha256"].lower():
                raise RuntimeError(
                    f"image_sha256 mismatch for {row['source_path']}: "
                    f"manifest={row['image_sha256']} actual={actual_hash}"
                )

            vector, status, reason = extract_landmarks_with_status(image_path, landmarker)
            row["landmark_status"] = status
            row["failure_reason"] = reason
            outcomes.append(row)
            if vector is not None:
                successful_features.append(vector)
                successful_labels.append(LABELS.index(row["label"]))
                successful_rows.append(row)
    finally:
        landmarker.close()

    feature_matrix = (
        np.stack(successful_features).astype(np.float32, copy=False)
        if successful_features
        else np.empty((0, 126), dtype=np.float32)
    )
    extraction_report = build_extraction_failure_report(outcomes)
    (output_dir / "extraction_failure_report.json").write_text(
        json.dumps(extraction_report, indent=2),
        encoding="utf-8",
    )

    overlaps = _exact_cross_split_hashes(outcomes)
    near_report = find_near_duplicate_landmarks(
        feature_matrix,
        successful_rows,
        threshold=near_duplicate_threshold,
    )
    duplicate_report = {
        "manifest_schema_version": 2,
        **overlaps,
        "near_duplicates": near_report,
        "near_duplicate_threshold": near_duplicate_threshold,
        "exact_duplicate_hashes_by_split": {
            split: find_duplicate_hashes([
                row for row in outcomes if row["split"] == split
            ])
            for split in ("train", "val", "test")
        },
    }
    (output_dir / "duplicate_report.json").write_text(
        json.dumps(duplicate_report, indent=2),
        encoding="utf-8",
    )
    if any(overlaps.values()):
        raise RuntimeError("Exact duplicate image hashes overlap between signer-held-out splits; see duplicate_report.json")
    enforce_no_cross_split_near_duplicates(near_report)

    split_features: dict[str, list[np.ndarray]] = {split: [] for split in ("train", "val", "test")}
    split_labels: dict[str, list[int]] = {split: [] for split in ("train", "val", "test")}
    rows_by_split: dict[str, list[dict]] = {split: [] for split in ("train", "val", "test")}
    for row in outcomes:
        rows_by_split[row["split"]].append(row)
    for vector, label, row in zip(successful_features, successful_labels, successful_rows):
        split_features[row["split"]].append(vector)
        split_labels[row["split"]].append(label)

    missing_successful_splits = [
        split for split in ("train", "val", "test") if not split_features[split]
    ]
    if missing_successful_splits:
        raise RuntimeError(
            "No usable landmarks remain in split(s): " + ", ".join(missing_successful_splits)
        )

    for split in ("train", "val", "test"):
        np.savez_compressed(
            output_dir / f"{split}.npz",
            x=np.stack(split_features[split]).astype(np.float32, copy=False),
            y=np.asarray(split_labels[split], dtype=np.int64),
        )
        with (output_dir / f"{split}_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows_by_split[split]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    label_coverage = {
        split: {
            "present": sorted({row["label"] for row in rows_by_split[split]}),
            "missing": sorted(set(LABELS) - {row["label"] for row in rows_by_split[split]}),
            "successful_present": sorted({
                row["label"] for row in successful_rows if row["split"] == split
            }),
        }
        for split in ("train", "val", "test")
    }
    metadata = {
        "labels": list(LABELS),
        "preprocessing_version": "two-hand-wrist-scale-v1",
        "signer_split": signer_split,
        "manifest_schema_version": 2,
        "landmarker": {
            "api": "mediapipe.tasks.vision.HandLandmarker",
            "runtime": "mediapipe-hand-landmarker-0.10.35",
            "model_url": HAND_LANDMARKER_URL,
            "model_size_bytes": HAND_LANDMARKER_SIZE,
            "model_sha256": HAND_LANDMARKER_SHA256,
            "num_hands": 2,
        },
        "split_policy": {
            "method": signer_split,
            "signers_disjoint": True,
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "seed": seed,
            "seed_applied": seed_applied,
            "label_coverage": label_coverage,
            "test_signers_partitioned_separately": True,
            "test_selection_history": (
                "NOT_VERIFIABLE_FROM_MANIFEST; record actual use in the promotion provenance report"
            ),
        },
        "stats": {
            "samples_by_split": {
                split: len(rows_by_split[split]) for split in ("train", "val", "test")
            },
            "successful_features_by_split": {
                split: len(split_features[split]) for split in ("train", "val", "test")
            },
            "extraction_failure_report": {
                "total_attempted": extraction_report["total_attempted"],
                "total_skipped": extraction_report["total_skipped"],
                "failure_rate": extraction_report["failure_rate"],
            },
            "duplicate_report": {
                "train_val_overlap_count": len(overlaps["train_val_overlap_hashes"]),
                "train_test_overlap_count": len(overlaps["train_test_overlap_hashes"]),
                "val_test_overlap_count": len(overlaps["val_test_overlap_hashes"]),
                "near_duplicate_cross_split_count": near_report["cross_split_pair_count"],
            },
            "signer_independent_evaluation": (
                "BLOCKED pending verified signer provenance and an untouched test-cohort promotion run"
            ),
        },
    }
    (output_dir / "labels.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"signer_split={signer_split} seed={seed} signers={len({row['signer_id'] for row in outcomes})}")
    print("signer_split_counts=" + json.dumps({
        split: len({row["signer_id"] for row in rows_by_split[split]})
        for split in ("train", "val", "test")
    }, sort_keys=True))
    print(
        f"landmark_extraction_failure_rate={extraction_report['failure_rate']:.4%} "
        f"skipped={extraction_report['total_skipped']} attempted={extraction_report['total_attempted']}"
    )
    print(f"near_duplicate_cross_split_pairs={near_report['cross_split_pair_count']}")
    print("signer-independent evaluation remains blocked until signer provenance is verified")


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
    parser.add_argument(
        "--manifest-v2",
        type=Path,
        default=None,
        help="Optional JSONL manifest v2; otherwise auto-detect input-root/manifest_v2.jsonl.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.01,
        help="Maximum RMS Euclidean distance in normalized 126D landmark space.",
    )
    return parser.parse_args()


def prepare_dataset(
    input_root: Path,
    output_dir: Path,
    validation_ratio: float = 0.20,
    seed: int = 42,
    hand_model_path: Path | None = None,
    manifest_v2_path: Path | None = None,
    near_duplicate_threshold: float = 0.01,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(near_duplicate_threshold) or near_duplicate_threshold < 0:
        raise ValueError("near_duplicate_threshold must be finite and non-negative")

    if manifest_v2_path is None:
        auto_manifest = input_root / "manifest_v2.jsonl"
        if auto_manifest.is_file():
            manifest_v2_path = auto_manifest
    if manifest_v2_path is not None:
        if not manifest_v2_path.is_file():
            raise FileNotFoundError(f"Manifest v2 file not found: {manifest_v2_path}")
        if hand_model_path is None:
            hand_model_path = (
                Path.home() / ".cache" / "visionbridge" / "hand_landmarker.task"
            )
        prepare_manifest_v2_dataset(
            input_root=input_root,
            output_dir=output_dir,
            manifest_path=manifest_v2_path,
            seed=seed,
            hand_model_path=hand_model_path,
            near_duplicate_threshold=near_duplicate_threshold,
        )
        return

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

    legacy_attempts: list[dict] = []
    for successful_rows, failure_rows, prepared_split in (
        (train_samples, train_report.get("failures", []), "source_training"),
        (validation_samples, validation_report.get("failures", []), "source_validation"),
        (test_samples, test_report.get("failures", []), "source_testing"),
    ):
        for sample in successful_rows:
            legacy_attempts.append({
                **sample,
                "label": sample["letter"],
                "split": prepared_split,
                "landmark_status": "ok",
                "failure_reason": None,
                "capture_metadata": _empty_capture_metadata(),
            })
        legacy_attempts.extend(failure_rows)
    extraction_failure_report = build_extraction_failure_report(legacy_attempts)
    (output_dir / "extraction_failure_report.json").write_text(
        json.dumps(extraction_failure_report, indent=2),
        encoding="utf-8",
    )

    test_hashes = {sample["image_sha256"] for sample in test_samples}
    pooled_records = [
        (vector, label, sample)
        for vector, label, sample in zip(
            np.concatenate([train_source_x, validation_source_x]),
            np.concatenate([train_source_y, validation_source_y]),
            train_samples + validation_samples,
        )
        if sample["image_sha256"] not in test_hashes
    ]

    excluded_test_overlap = len(train_samples) + len(validation_samples) - len(pooled_records)
    if excluded_test_overlap:
        print(
            "Excluded {} training/validation sample(s) whose exact image hash "
            "also occurs in the source test split.".format(excluded_test_overlap)
        )

    if not pooled_records:
        raise RuntimeError("No training/validation samples remain after test-overlap filtering")

    pooled_x = np.stack([item[0] for item in pooled_records])
    pooled_y = np.asarray([item[1] for item in pooled_records], dtype=np.int64)
    pooled_samples = [item[2] for item in pooled_records]

    train_x, train_y, train_manifest, val_x, val_y, val_manifest = (
        make_train_validation_split(
            pooled_x,
            pooled_y,
            pooled_samples,
            validation_ratio=validation_ratio,
            seed=seed,
        )
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
    duplicate_rows = (
        [{**sample, "prepared_split": "train"} for sample in train_manifest]
        + [{**sample, "prepared_split": "val"} for sample in val_manifest]
        + [{**sample, "prepared_split": "test"} for sample in test_samples]
    )
    duplicate_features = np.concatenate([train_x, val_x, test_x], axis=0)
    near_duplicate_report = find_near_duplicate_landmarks(
        duplicate_features,
        duplicate_rows,
        threshold=near_duplicate_threshold,
    )
    duplicate_report["near_duplicate_threshold"] = near_duplicate_threshold
    duplicate_report["near_duplicates"] = near_duplicate_report

    if duplicate_report["train_val_overlap_hashes"]:
        raise RuntimeError(
            "Exact duplicate image hashes overlap between train and validation"
        )
    if duplicate_report["train_test_overlap_hashes"] or duplicate_report["val_test_overlap_hashes"]:
        raise RuntimeError(
            "Exact duplicate image hashes overlap the untouched source test split"
        )

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
            "model_size_bytes": HAND_LANDMARKER_SIZE,
            "model_sha256": HAND_LANDMARKER_SHA256,
            "num_hands": 2,
        },
        "signer_split": "unavailable",
        "split_policy": {
            "signer_split": "unavailable",
            "train_validation_ratio": 1.0 - validation_ratio,
            "validation_ratio": validation_ratio,
            "source_train_and_validation_pooled": True,
            "test_source_left_untouched": True,
            "exact_image_duplicates_grouped_within_train_validation": True,
            "seed": seed,
        },
        "stats": {
            "source_training": {key: value for key, value in train_report.items() if key != "failures"},
            "source_validation": {key: value for key, value in validation_report.items() if key != "failures"},
            "extraction_failure_report": {
                "total_attempted": extraction_failure_report["total_attempted"],
                "total_skipped": extraction_failure_report["total_skipped"],
                "failure_rate": extraction_failure_report["failure_rate"],
            },
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

    enforce_no_cross_split_near_duplicates(near_duplicate_report)
    print(
        f"landmark_extraction_failure_rate={extraction_failure_report['failure_rate']:.4%} "
        f"skipped={extraction_failure_report['total_skipped']} "
        f"attempted={extraction_failure_report['total_attempted']}"
    )
    print("signer_split=unavailable (legacy dataset has no verified signer manifest)")
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
        manifest_v2_path=args.manifest_v2,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )


if __name__ == "__main__":
    main()
