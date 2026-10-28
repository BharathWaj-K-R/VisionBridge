"""Seeded augmentation on raw MediaPipe hand coordinates, before normalization.

All functions accept a flat 126-value vector (left hand xyz[21], then right hand
xyz[21]) and return a new float32 vector. A missing hand is encoded as 63 zeros.
Mirroring and random hand dropout are intentionally not implemented.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

RAW_HAND_DIM = 63
RAW_PAIR_DIM = 126
LANDMARK_COUNT = 21
COORDS_PER_LANDMARK = 3
BONE_LENGTH_TOLERANCE = 0.05
ANGLE_PERTURBATION_MAX_DEGREES = 5.0
AUGMENTATION_NAMES = (
    "none",
    "gaussian_jitter",
    "in_plane_rotation",
    "depth_scale",
    "finger_joint_angles",
)

# MediaPipe Hands connections, with wrist/index landmarks as documented by
# MediaPipe. Links are used for relative bone-length checks, not as a hand mask.
BONE_PAIRS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

# Flexion is approximated in 3D as 180 degrees minus the internal angle
# between the adjacent landmark rays. Bounds are conservative upper bounds,
# not an ISL-specific hand-shape model. Source references are documented in
# docs/experiments/augmentation_comparison.md.
# Format: parent, joint, child, distal subtree, maximum flexion (degrees).
_FINGER_JOINTS = {
    "thumb": ((2, 3, 4, (4,), 90.0),),
    "index": ((5, 6, 7, (7, 8), 120.0), (6, 7, 8, (8,), 80.0)),
    "middle": ((9, 10, 11, (11, 12), 120.0), (10, 11, 12, (12,), 80.0)),
    "ring": ((13, 14, 15, (15, 16), 120.0), (14, 15, 16, (16,), 80.0)),
    "pinky": ((17, 18, 19, (19, 20), 120.0), (18, 19, 20, (20,), 80.0)),
}


class ImplausibleLandmarkSample(ValueError):
    """Raised when augmentation would return a geometrically invalid sample."""


def _as_raw_pair(raw: np.ndarray | list[float]) -> np.ndarray:
    try:
        values = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Raw landmarks must be numeric") from exc
    if values.shape != (RAW_PAIR_DIM,):
        raise ValueError(f"Expected {RAW_PAIR_DIM} raw xyz values, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Raw landmarks contain NaN or Inf")
    return values.copy()


def _seeded_rng(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Augmentation seed must be an integer")
    return np.random.default_rng(int(seed))


def _hand_points(raw: np.ndarray, side: int) -> np.ndarray:
    start = side * RAW_HAND_DIM
    return raw[start:start + RAW_HAND_DIM].reshape(LANDMARK_COUNT, COORDS_PER_LANDMARK)


def _present_sides(raw: np.ndarray) -> list[int]:
    return [
        side for side in (0, 1)
        if np.any(np.abs(_hand_points(raw, side)) > 1e-8)
    ]


def _wrist_center(raw: np.ndarray, sides: list[int]) -> np.ndarray:
    if not sides:
        raise ValueError("No visible hand landmarks were provided")
    return np.mean(
        np.stack([_hand_points(raw, side)[0] for side in sides], axis=0),
        axis=0,
    ).astype(np.float32)


def gaussian_jitter(
    raw: np.ndarray | list[float],
    *,
    seed: int,
    standard_deviation: float = 0.002,
) -> np.ndarray:
    """Add independent, small xyz Gaussian noise to detected hands only."""
    if not np.isfinite(standard_deviation) or standard_deviation < 0:
        raise ValueError("standard_deviation must be finite and non-negative")
    values = _as_raw_pair(raw)
    rng = _seeded_rng(seed)
    for side in _present_sides(values):
        points = _hand_points(values, side)
        points += rng.normal(0.0, standard_deviation, size=points.shape).astype(np.float32)
    return values


def in_plane_rotation(
    raw: np.ndarray | list[float],
    *,
    seed: int,
    max_degrees: float = 5.0,
) -> np.ndarray:
    """Rotate all detected hand landmarks together in image x/y about wrist center."""
    if not np.isfinite(max_degrees) or not 0 <= max_degrees <= 15:
        raise ValueError("max_degrees must be between 0 and 15")
    values = _as_raw_pair(raw)
    sides = _present_sides(values)
    center = _wrist_center(values, sides)
    angle = np.deg2rad(_seeded_rng(seed).uniform(-max_degrees, max_degrees))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    for side in sides:
        points = _hand_points(values, side)
        points[:, :2] = (points[:, :2] - center[:2]) @ rotation.T + center[:2]
    return values


def depth_scale(
    raw: np.ndarray | list[float],
    *,
    seed: int,
    xy_scale_range: tuple[float, float] = (0.95, 1.05),
    z_scale_range: tuple[float, float] = (0.95, 1.05),
) -> np.ndarray:
    """Perturb global in-plane scale and relative depth scale before normalization."""
    for name, bounds in (("xy_scale_range", xy_scale_range), ("z_scale_range", z_scale_range)):
        if (
            len(bounds) != 2
            or not np.isfinite(bounds).all()
            or bounds[0] <= 0
            or bounds[0] > bounds[1]
            or bounds[1] > 1.25
            or bounds[0] < 0.75
        ):
            raise ValueError(f"{name} must be an ordered positive range within [0.75, 1.25]")
    values = _as_raw_pair(raw)
    sides = _present_sides(values)
    center = _wrist_center(values, sides)
    rng = _seeded_rng(seed)
    xy_scale = float(rng.uniform(*xy_scale_range))
    z_scale = float(rng.uniform(*z_scale_range))
    for side in sides:
        points = _hand_points(values, side)
        points[:, :2] = (points[:, :2] - center[:2]) * xy_scale + center[:2]
        points[:, 2] = (points[:, 2] - center[2]) * z_scale + center[2]
    return values


def _bone_lengths(points: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.linalg.norm(points[left] - points[right]) for left, right in BONE_PAIRS],
        dtype=np.float64,
    )


def _flexion_degrees(points: np.ndarray, parent: int, joint: int, child: int) -> float:
    toward_parent = points[parent].astype(np.float64) - points[joint].astype(np.float64)
    toward_child = points[child].astype(np.float64) - points[joint].astype(np.float64)
    norm_parent = float(np.linalg.norm(toward_parent))
    norm_child = float(np.linalg.norm(toward_child))
    if norm_parent < 1e-7 or norm_child < 1e-7:
        raise ImplausibleLandmarkSample("zero_length_joint_bone")
    cosine = float(np.dot(toward_parent, toward_child) / (norm_parent * norm_child))
    internal_angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return 180.0 - internal_angle


def _palm_normal(points: np.ndarray) -> np.ndarray:
    across_palm = points[17].astype(np.float64) - points[5].astype(np.float64)
    toward_fingers = (points[5].astype(np.float64) + points[17].astype(np.float64)) / 2 - points[0].astype(np.float64)
    normal = np.cross(across_palm, toward_fingers)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ImplausibleLandmarkSample("palm_plane_is_degenerate")
    return normal / norm


def _rotate_points_about_joint(
    points: np.ndarray,
    *,
    joint: int,
    distal: tuple[int, ...],
    axis: np.ndarray,
    angle_radians: float,
) -> None:
    origin = points[joint].astype(np.float64)
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    axis = axis.astype(np.float64)
    for index in distal:
        vector = points[index].astype(np.float64) - origin
        rotated = (
            vector * cosine
            + np.cross(axis, vector) * sine
            + axis * np.dot(axis, vector) * (1.0 - cosine)
        )
        points[index] = (origin + rotated).astype(np.float32)


def assess_anatomical_plausibility(
    raw: np.ndarray | list[float],
    *,
    reference: np.ndarray | list[float] | None = None,
    bone_length_tolerance: float = BONE_LENGTH_TOLERANCE,
) -> dict:
    """Check finite coordinates, non-degenerate connected bones, angle caps and optional length drift."""
    if (
        not np.isfinite(bone_length_tolerance)
        or bone_length_tolerance < 0
        or bone_length_tolerance > 0.25
    ):
        raise ValueError("bone_length_tolerance must be in [0, 0.25]")
    try:
        values = _as_raw_pair(raw)
        reference_values = _as_raw_pair(reference) if reference is not None else None
    except ValueError as exc:
        return {"plausible": False, "issues": [str(exc)], "bone_length_tolerance": bone_length_tolerance}

    issues: list[str] = []
    flexions: dict[str, dict[str, dict[str, float]]] = {}
    sides = _present_sides(values)
    if not sides:
        return {
            "plausible": False,
            "issues": ["no_visible_hand"],
            "bone_length_tolerance": bone_length_tolerance,
            "flexion_degrees": {},
        }

    for side in sides:
        side_name = "left" if side == 0 else "right"
        points = _hand_points(values, side)
        reference_points = _hand_points(reference_values, side) if reference_values is not None else None
        lengths = _bone_lengths(points)
        if not np.isfinite(lengths).all() or np.any(lengths < 1e-7):
            issues.append(f"{side_name}:zero_or_invalid_bone_length")
        if reference_points is not None:
            reference_lengths = _bone_lengths(reference_points)
            ratios = np.abs(lengths - reference_lengths) / np.maximum(reference_lengths, 1e-8)
            if not np.isfinite(ratios).all() or np.any(ratios > bone_length_tolerance + 1e-6):
                issues.append(f"{side_name}:bone_length_drift_exceeds_tolerance")

        flexions[side_name] = {}
        for finger, joint_specs in _FINGER_JOINTS.items():
            flexions[side_name][finger] = {}
            for parent, joint, child, _, max_flexion in joint_specs:
                try:
                    angle = _flexion_degrees(points, parent, joint, child)
                except ImplausibleLandmarkSample:
                    issues.append(f"{side_name}:{finger}:{joint}:degenerate_joint")
                    continue
                flexions[side_name][finger][str(joint)] = angle
                if angle > max_flexion + 1e-4:
                    issues.append(f"{side_name}:{finger}:{joint}:flexion_above_{max_flexion:g}_degrees")

    return {
        "plausible": not issues,
        "issues": issues,
        "bone_length_tolerance": bone_length_tolerance,
        "flexion_degrees": flexions,
    }


def finger_joint_angle_perturbation(
    raw: np.ndarray | list[float],
    *,
    seed: int,
    max_delta_degrees: float = ANGLE_PERTURBATION_MAX_DEGREES,
    bone_length_tolerance: float = BONE_LENGTH_TOLERANCE,
    attempts_per_joint: int = 12,
) -> np.ndarray:
    """Perturb one PIP/DIP angle per finger, preserving bone lengths by rigid subtree rotation.

    A sample that violates the documented angle caps or bone-length tolerance is
    rejected by raising ImplausibleLandmarkSample. Callers should record the
    exception as a rejected augmentation and can fall back to the original sample.
    """
    if not np.isfinite(max_delta_degrees) or not 0 <= max_delta_degrees <= 10:
        raise ValueError("max_delta_degrees must be between 0 and 10")
    if isinstance(attempts_per_joint, bool) or attempts_per_joint < 1:
        raise ValueError("attempts_per_joint must be positive")
    original = _as_raw_pair(raw)
    sides = _present_sides(original)
    if not sides:
        raise ImplausibleLandmarkSample("no_visible_hand")
    baseline_report = assess_anatomical_plausibility(
        original, bone_length_tolerance=bone_length_tolerance
    )
    if not baseline_report["plausible"]:
        raise ImplausibleLandmarkSample("baseline:" + ",".join(baseline_report["issues"]))

    rng = _seeded_rng(seed)
    values = original.copy()
    for side in sides:
        points = _hand_points(values, side)
        axis = _palm_normal(points)
        for finger, joint_specs in _FINGER_JOINTS.items():
            parent, joint, child, distal, maximum = joint_specs[int(rng.integers(0, len(joint_specs)))]
            before = _flexion_degrees(points, parent, joint, child)
            if before < 0 or before > maximum:
                raise ImplausibleLandmarkSample(
                    f"{'left' if side == 0 else 'right'}:{finger}:baseline_flexion_out_of_range"
                )

            accepted = None
            for _ in range(attempts_per_joint):
                delta = float(rng.uniform(-max_delta_degrees, max_delta_degrees))
                candidate = points.copy()
                _rotate_points_about_joint(
                    candidate,
                    joint=joint,
                    distal=distal,
                    axis=axis,
                    angle_radians=np.deg2rad(delta),
                )
                after = _flexion_degrees(candidate, parent, joint, child)
                if 0.0 <= after <= maximum:
                    accepted = candidate
                    break
            if accepted is None:
                raise ImplausibleLandmarkSample(
                    f"{'left' if side == 0 else 'right'}:{finger}:no_plausible_angle_perturbation"
                )
            points[:] = accepted

    report = assess_anatomical_plausibility(
        values,
        reference=original,
        bone_length_tolerance=bone_length_tolerance,
    )
    if not report["plausible"]:
        raise ImplausibleLandmarkSample("augmented:" + ",".join(report["issues"]))
    return values


def augment_raw_landmarks(
    raw: np.ndarray | list[float],
    *,
    augmentation: str,
    seed: int,
) -> np.ndarray:
    """Apply exactly one named augmentation; 'none' is the matched baseline."""
    if augmentation not in AUGMENTATION_NAMES:
        raise ValueError(f"Unsupported single augmentation: {augmentation}")
    if augmentation == "none":
        return _as_raw_pair(raw)
    operations: dict[str, Callable[..., np.ndarray]] = {
        "gaussian_jitter": gaussian_jitter,
        "in_plane_rotation": in_plane_rotation,
        "depth_scale": depth_scale,
        "finger_joint_angles": finger_joint_angle_perturbation,
    }
    return operations[augmentation](raw, seed=seed)


def apply_augmentation_with_fallback(
    raw_samples: np.ndarray,
    *,
    augmentation: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Keep sample count stable; replace rejected augmented rows with originals and report rejection rate."""
    samples = np.asarray(raw_samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != RAW_PAIR_DIM or len(samples) == 0:
        raise ValueError(f"raw_samples must be a non-empty [N, {RAW_PAIR_DIM}] float32 matrix")
    if augmentation not in AUGMENTATION_NAMES:
        raise ValueError(f"Unsupported single augmentation: {augmentation}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Augmentation seed must be an integer")

    output = np.empty_like(samples)
    rejections: Counter[str] = Counter()
    accepted = 0
    for index, sample in enumerate(samples):
        row_seed = int(np.random.SeedSequence([int(seed), index]).generate_state(1)[0])
        try:
            output[index] = augment_raw_landmarks(
                sample,
                augmentation=augmentation,
                seed=row_seed,
            )
            accepted += 1
        except ImplausibleLandmarkSample as exc:
            if augmentation != "finger_joint_angles":
                raise
            output[index] = sample
            reason = str(exc).split(":", 1)[0]
            rejections[reason] += 1
    attempted = len(samples) if augmentation != "none" else 0
    rejected = attempted - accepted if attempted else 0
    report = {
        "augmentation": augmentation,
        "seed": int(seed),
        "attempted": attempted,
        "accepted": accepted if attempted else 0,
        "rejected": rejected,
        "rejection_rate": rejected / attempted if attempted else 0.0,
        "rejection_reasons": dict(sorted(rejections.items())),
        "rejected_samples_fallback": "original raw sample",
    }
    return output, report



def combined_raw_augmentation(
    raw: np.ndarray | list[float],
    *,
    augmentations: tuple[str, ...] | list[str],
    seed: int,
) -> np.ndarray:
    """Apply only a previously validated keep-set, in deterministic component order."""
    components = tuple(augmentations)
    valid_components = tuple(name for name in AUGMENTATION_NAMES if name not in {"none"})
    if not components or len(set(components)) != len(components):
        raise ValueError("A combined run requires a non-empty list of unique kept augmentations")
    if any(name not in valid_components for name in components):
        raise ValueError("Combined components must be standalone augmentation IDs")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Augmentation seed must be an integer")

    value = _as_raw_pair(raw)
    for position, augmentation in enumerate(components):
        child_seed = int(np.random.SeedSequence([int(seed), position]).generate_state(1)[0])
        value = augment_raw_landmarks(value, augmentation=augmentation, seed=child_seed)
    return value


def apply_combined_augmentation_with_fallback(
    raw_samples: np.ndarray,
    *,
    augmentations: tuple[str, ...] | list[str],
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Apply one gated combined run; joint-angle rejection falls back to the original row."""
    samples = np.asarray(raw_samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != RAW_PAIR_DIM or len(samples) == 0:
        raise ValueError(f"raw_samples must be a non-empty [N, {RAW_PAIR_DIM}] float32 matrix")
    components = tuple(augmentations)
    # Validate the component list even for an empty input or an all-rejected batch.
    combined_raw_augmentation(samples[0], augmentations=components, seed=seed)
    output = np.empty_like(samples)
    rejections: Counter[str] = Counter()
    accepted = 0
    for index, sample in enumerate(samples):
        row_seed = int(np.random.SeedSequence([int(seed), index]).generate_state(1)[0])
        try:
            output[index] = combined_raw_augmentation(
                sample,
                augmentations=components,
                seed=row_seed,
            )
            accepted += 1
        except ImplausibleLandmarkSample as exc:
            if "finger_joint_angles" not in components:
                raise
            output[index] = sample
            rejections[str(exc).split(":", 1)[0]] += 1
    attempted = len(samples)
    rejected = attempted - accepted
    return output, {
        "augmentation": "combined",
        "components": list(components),
        "seed": int(seed),
        "attempted": attempted,
        "accepted": accepted,
        "rejected": rejected,
        "rejection_rate": rejected / attempted,
        "rejection_reasons": dict(sorted(rejections.items())),
        "rejected_samples_fallback": "original raw sample",
    }
