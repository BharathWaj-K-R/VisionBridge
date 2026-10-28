from __future__ import annotations

import numpy as np
import pytest

from app.training.landmark_augment import (
    BONE_LENGTH_TOLERANCE,
    ImplausibleLandmarkSample,
    apply_augmentation_with_fallback,
    assess_anatomical_plausibility,
    depth_scale,
    finger_joint_angle_perturbation,
    gaussian_jitter,
    in_plane_rotation,
)


def _synthetic_hand(center_x: float) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = [center_x, 0.65, 0.0]
    # Thumb chain (CMC, MCP, IP, tip).
    points[1] = [center_x - 0.020, 0.630, 0.002]
    points[2] = [center_x - 0.038, 0.610, 0.004]
    points[3] = [center_x - 0.050, 0.590, 0.006]
    points[4] = [center_x - 0.057, 0.571, 0.008]
    for base, offset in ((5, -0.030), (9, -0.010), (13, 0.010), (17, 0.030)):
        points[base] = [center_x + offset, 0.620, 0.0]
        points[base + 1] = [center_x + offset * 1.03, 0.592, 0.001]
        points[base + 2] = [center_x + offset * 1.06, 0.564, 0.002]
        points[base + 3] = [center_x + offset * 1.09, 0.536, 0.003]
    return points


def _pair() -> np.ndarray:
    return np.concatenate((_synthetic_hand(0.30).reshape(-1), _synthetic_hand(0.70).reshape(-1)))


def test_gaussian_jitter_is_seeded_and_keeps_missing_hand_zero() -> None:
    raw = np.concatenate((_synthetic_hand(0.30).reshape(-1), np.zeros(63, dtype=np.float32)))
    one = gaussian_jitter(raw, seed=17, standard_deviation=0.001)
    two = gaussian_jitter(raw, seed=17, standard_deviation=0.001)
    np.testing.assert_array_equal(one, two)
    assert np.any(one[:63] != raw[:63])
    assert np.count_nonzero(one[63:]) == 0


def test_in_plane_rotation_is_seeded_and_preserves_distances() -> None:
    raw = _pair()
    one = in_plane_rotation(raw, seed=3, max_degrees=5.0)
    two = in_plane_rotation(raw, seed=3, max_degrees=5.0)
    np.testing.assert_array_equal(one, two)
    before = raw.reshape(2, 21, 3)
    after = one.reshape(2, 21, 3)
    for side in range(2):
        np.testing.assert_allclose(
            np.linalg.norm(before[side, 0] - before[side, 8]),
            np.linalg.norm(after[side, 0] - after[side, 8]),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(before[side, :, 2], after[side, :, 2], atol=0)


def test_depth_scale_is_seeded_finite_and_changes_geometry() -> None:
    raw = _pair()
    one = depth_scale(raw, seed=22)
    two = depth_scale(raw, seed=22)
    np.testing.assert_array_equal(one, two)
    assert np.isfinite(one).all()
    assert not np.array_equal(one, raw)


def test_joint_angle_perturbation_preserves_bone_lengths_within_tolerance() -> None:
    raw = _pair()
    report = assess_anatomical_plausibility(raw)
    assert report["plausible"], report["issues"]
    augmented = finger_joint_angle_perturbation(raw, seed=29, max_delta_degrees=5.0)
    checked = assess_anatomical_plausibility(
        augmented,
        reference=raw,
        bone_length_tolerance=BONE_LENGTH_TOLERANCE,
    )
    assert checked["plausible"], checked["issues"]
    assert checked["bone_length_tolerance"] == 0.05
    assert np.isfinite(augmented).all()
    assert not np.array_equal(augmented, raw)


def test_joint_angle_perturbation_rejects_degenerate_bones() -> None:
    raw = _pair().reshape(2, 21, 3)
    raw[0, 8] = raw[0, 7]
    with pytest.raises(ImplausibleLandmarkSample):
        finger_joint_angle_perturbation(raw.reshape(-1), seed=1)


def test_joint_angle_batch_logs_rejections_and_preserves_sample_count() -> None:
    valid = _pair()
    invalid = valid.reshape(2, 21, 3).copy()
    invalid[0, 8] = invalid[0, 7]
    batch = np.stack((valid, invalid.reshape(-1))).astype(np.float32)
    augmented, report = apply_augmentation_with_fallback(
        batch,
        augmentation="finger_joint_angles",
        seed=7,
    )
    assert augmented.shape == batch.shape
    assert report["attempted"] == 2
    assert report["rejected"] == 1
    assert report["rejection_rate"] == 0.5
    np.testing.assert_array_equal(augmented[1], batch[1])


@pytest.mark.parametrize("function", [gaussian_jitter, in_plane_rotation, depth_scale])
def test_augmentations_reject_invalid_raw_shapes(function) -> None:
    with pytest.raises(ValueError):
        function([0.0] * 125, seed=1)


def test_mirroring_and_random_hand_dropout_are_not_supported() -> None:
    from app.training.landmark_augment import augment_raw_landmarks

    with pytest.raises(ValueError, match="Unsupported single augmentation"):
        augment_raw_landmarks(_pair(), augmentation="mirror", seed=1)
    with pytest.raises(ValueError, match="Unsupported single augmentation"):
        augment_raw_landmarks(_pair(), augmentation="single_hand_dropout", seed=1)
