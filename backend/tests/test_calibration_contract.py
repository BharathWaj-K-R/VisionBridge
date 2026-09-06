import pytest

from app.api.calibration import _ctc_min_input_length, _downsample
from app.core.config import get_settings


def test_ctc_minimum_length_accounts_for_adjacent_repeats():
    assert _ctc_min_input_length([1, 2, 2, 3, 3, 3]) == 9


def test_calibration_downsample_preserves_multimodal_alignment(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "CALIBRATION_MAX_FRAMES", 4)
    pose = [[float(i), 0.0] for i in range(10)]
    face = [[float(i), 1.0] for i in range(10)]
    left_hand = [[float(i), 2.0] for i in range(10)]
    right_hand = [[float(i), 3.0] for i in range(10)]

    sampled_pose, sampled_face, sampled_left, sampled_right = _downsample(
        pose,
        face,
        left_hand,
        right_hand,
    )

    assert all(len(stream) == 4 for stream in (sampled_pose, sampled_face, sampled_left, sampled_right))
    assert [row[0] for row in sampled_pose] == [row[0] for row in sampled_face]
    assert [row[0] for row in sampled_pose] == [row[0] for row in sampled_left]
    assert [row[0] for row in sampled_pose] == [row[0] for row in sampled_right]
    assert sampled_pose[0][0] == 0.0
    assert sampled_pose[-1][0] == 9.0


def test_calibration_max_frames_cannot_exceed_inference_max(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "MAX_INFERENCE_FRAMES", 100)
    monkeypatch.setattr(settings, "CALIBRATION_MAX_FRAMES", 101)
    with pytest.raises(RuntimeError, match="cannot exceed"):
        settings.validate_for_runtime()
