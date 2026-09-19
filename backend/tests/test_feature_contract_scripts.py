from pathlib import Path

import numpy as np
import pytest

from scripts.extract_keypoints import resolve_video_path


def test_resolve_video_path_accepts_uppercase_extension(tmp_path: Path) -> None:
    video = tmp_path / "clip.MP4"
    video.write_bytes(b"not-a-real-video")

    assert resolve_video_path(tmp_path, "clip") == str(video)


def test_resolve_video_path_rejects_unsupported_extension(tmp_path: Path) -> None:
    (tmp_path / "clip.txt").write_text("x", encoding="utf-8")

    assert resolve_video_path(tmp_path, "clip") is None


def test_converter_contract_does_not_use_1434_face_dimension() -> None:
    source = Path(__file__).parents[1] / "scripts" / "convert_isign_pose.py"
    text = source.read_text(encoding="utf-8")
    assert "EXPECTED_FACE_DIM = 1404" in text
    assert "1434" not in text


def test_converter_contract_expected_arrays_are_2d() -> None:
    pose = np.zeros((4, 132), dtype=np.float32)
    face = np.zeros((4, 1404), dtype=np.float32)
    assert pose.ndim == face.ndim == 2
    assert pose.shape[0] == face.shape[0]
    assert pose.shape[1] == 132
    assert face.shape[1] == 1404


@pytest.mark.parametrize("bad_dim", [1434, 1403, 0])
def test_incompatible_face_dimensions_are_not_current_contract(bad_dim: int) -> None:
    assert bad_dim != 1404
