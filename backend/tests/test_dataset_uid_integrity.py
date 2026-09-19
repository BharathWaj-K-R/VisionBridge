from pathlib import Path

import numpy as np
import pytest

from app.training.isltranslate import ISLTranslateKeypointDataset


def _write_example(root: Path, uid: str) -> None:
    np.save(root / "pose" / f"{uid}.npy", np.zeros((2, 132), dtype=np.float32))
    np.save(root / "face" / f"{uid}.npy", np.zeros((2, 1404), dtype=np.float32))


def test_duplicate_uid_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "isltranslate"
    (root / "pose").mkdir(parents=True)
    (root / "face").mkdir(parents=True)
    _write_example(root, "same")
    (root / "ISLTranslate.csv").write_text(
        "uid,text\nsame,hello\nsame,goodbye\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate uid"):
        ISLTranslateKeypointDataset(root)


def test_empty_target_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "isltranslate"
    (root / "pose").mkdir(parents=True)
    (root / "face").mkdir(parents=True)
    _write_example(root, "sample")
    (root / "ISLTranslate.csv").write_text(
        "uid,text\nsample,\n",
        encoding="utf-8",
    )

    # Empty text rows are ignored by the manifest reader rather than creating
    # an invalid CTC target; the dataset therefore has no usable examples.
    with pytest.raises(ValueError, match="No usable ISLTranslate examples"):
        ISLTranslateKeypointDataset(root)
