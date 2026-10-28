from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.prepare_raw_landmark_dataset as raw_prep


def _synthetic_pair() -> tuple[np.ndarray, bool, bool]:
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    for points, center in ((left, 0.3), (right, 0.7)):
        points[0] = [center, 0.65, 0.0]
        points[1] = [center - 0.02, 0.63, 0.001]
        points[2] = [center - 0.03, 0.61, 0.002]
        points[3] = [center - 0.04, 0.59, 0.003]
        points[4] = [center - 0.05, 0.57, 0.004]
        for base, offset in ((5, -0.03), (9, -0.01), (13, 0.01), (17, 0.03)):
            points[base] = [center + offset, 0.62, 0.0]
            points[base + 1] = [center + offset, 0.59, 0.001]
            points[base + 2] = [center + offset, 0.56, 0.002]
            points[base + 3] = [center + offset, 0.53, 0.003]
    return np.concatenate((left.reshape(-1), right.reshape(-1))), True, True


def _write_manifest(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_raw_prep_defaults_to_train_val_and_aligns_existing_manifest_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_images = tmp_path / "images"
    v1_root = tmp_path / "v1"
    output = tmp_path / "raw"
    raw_images.mkdir()
    v1_root.mkdir()
    (v1_root / "labels.json").write_text(
        json.dumps({
            "labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            "preprocessing_version": "two-hand-wrist-scale-v1",
            "landmarker": {"runtime": "mediapipe-hand-landmarker-0.10.35"},
        }),
        encoding="utf-8",
    )
    fake_vector, left_present, right_present = _synthetic_pair()

    class FakeLandmarker:
        closed = False
        def close(self):
            self.closed = True

    landmarker = FakeLandmarker()
    monkeypatch.setattr(raw_prep, "create_hand_landmarker", lambda path: landmarker)
    monkeypatch.setattr(raw_prep, "_extract_raw_hand_pair", lambda path, obj: (fake_vector.copy(), left_present, right_present))

    for split in ("train", "val", "test"):
        source_rel = f"{split}/a.png"
        image = raw_images / source_rel
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(split.encode("utf-8"))
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        row = {
            "sample_id": f"{split}-a",
            "source_path": source_rel,
            "image_sha256": image_hash,
            "label": "A",
            "letter": "A",
            "landmark_status": "ok",
            "split": split,
        }
        _write_manifest(v1_root / f"{split}_manifest.jsonl", row)
        np.savez_compressed(
            v1_root / f"{split}.npz",
            x=np.zeros((1, 126), dtype=np.float32) + np.eye(1, 126, dtype=np.float32) * 0.1,
            y=np.asarray([0], dtype=np.int64),
        )

    report = raw_prep.prepare_raw_landmark_dataset(
        input_root=raw_images,
        v1_data_dir=v1_root,
        output_dir=output,
        hand_model_path=tmp_path / "fake.task",
    )
    assert report["samples_by_split"] == {"train": 1, "val": 1}
    assert report["omitted_splits"] == ["test"]
    assert report["test_split_processed"] is False
    assert landmarker.closed is True
    assert (output / "train_raw.npz").is_file()
    assert (output / "val_raw.npz").is_file()
    assert not (output / "test_raw.npz").exists()
    with np.load(output / "train_raw.npz", allow_pickle=False) as data:
        assert data["x"].shape == (1, 126)
        assert data["left_present"].tolist() == [True]
        assert data["right_present"].tolist() == [True]
        assert data["y"].tolist() == [0]


def test_raw_prep_rejects_non_v1_contract_before_creating_landmarker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "images"
    source.mkdir()
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / "labels.json").write_text(
        json.dumps({"labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "preprocessing_version": "unknown"}),
        encoding="utf-8",
    )
    called = False
    def fail_if_created(path):
        nonlocal called
        called = True
        raise AssertionError("must not create landmarker for incompatible contract")
    monkeypatch.setattr(raw_prep, "create_hand_landmarker", fail_if_created)
    with pytest.raises(ValueError, match="documented V1"):
        raw_prep.prepare_raw_landmark_dataset(
            input_root=source,
            v1_data_dir=v1,
            output_dir=tmp_path / "out",
        )
    assert not called
