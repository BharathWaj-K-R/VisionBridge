"""Extract synchronized pose, face, and hand skeletons from raw videos.

Feature contract per frame:
  pose:       33 landmarks * (x,y,z,visibility) = 132
  face:       468 landmarks * (x,y,z)            = 1404
  left_hand:   21 landmarks * (x,y,z)            = 63
  right_hand:  21 landmarks * (x,y,z)            = 63

The legacy ``extract_clip_keypoints`` wrapper still returns pose+face for old
callers. New training/validation workflows should use
``extract_clip_keypoints_with_hands`` and persist all four streams.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import types

import numpy as np

POSE_FEATURE_DIM = 33 * 4
FACE_FEATURE_DIM = 468 * 3
HAND_FEATURE_DIM = 21 * 3
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _stub_tensorflow_for_mediapipe() -> None:
    if "tensorflow" in sys.modules:
        return
    try:
        import tensorflow  # noqa: F401
        return
    except Exception:
        pass

    def _noop_decorator(fn):
        return fn

    fake_tf = types.ModuleType("tensorflow")
    fake_tf_tools = types.ModuleType("tensorflow.tools")
    fake_tf_docs = types.ModuleType("tensorflow.tools.docs")
    fake_tf_docs.doc_controls = types.SimpleNamespace(
        do_not_generate_docs=_noop_decorator,
        for_subclass_implementers=_noop_decorator,
        do_not_doc_inheritable=_noop_decorator,
    )
    fake_tf.tools = fake_tf_tools
    fake_tf_tools.docs = fake_tf_docs
    sys.modules["tensorflow"] = fake_tf
    sys.modules["tensorflow.tools"] = fake_tf_tools
    sys.modules["tensorflow.tools.docs"] = fake_tf_docs


_stub_tensorflow_for_mediapipe()


def _flatten_landmarks(landmarks, dimensions: int, include_visibility: bool = False) -> np.ndarray:
    if landmarks is None:
        return np.zeros(
            dimensions * (4 if include_visibility else 3),
            dtype=np.float32,
        )
    if len(landmarks.landmark) != dimensions:
        raise ValueError(
            f"Unexpected landmark count: {len(landmarks.landmark)}, expected {dimensions}"
        )
    if include_visibility:
        values = [
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in landmarks.landmark
        ]
    else:
        values = [
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ]
    return np.asarray(values, dtype=np.float32).reshape(-1)


def extract_clip_keypoints_with_hands(video_path: str, holistic) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run MediaPipe Holistic and return synchronized pose/face/hand arrays."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    pose_frames: list[np.ndarray] = []
    face_frames: list[np.ndarray] = []
    left_frames: list[np.ndarray] = []
    right_frames: list[np.ndarray] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            pose = _flatten_landmarks(
                results.pose_landmarks,
                33,
                include_visibility=True,
            )
            face = _flatten_landmarks(
                results.face_landmarks,
                468,
            )
            left = _flatten_landmarks(
                results.left_hand_landmarks,
                21,
            )
            right = _flatten_landmarks(
                results.right_hand_landmarks,
                21,
            )

            if pose.shape[0] != POSE_FEATURE_DIM:
                raise ValueError(f"Pose dimension {pose.shape[0]} != {POSE_FEATURE_DIM}")
            if face.shape[0] != FACE_FEATURE_DIM:
                raise ValueError(f"Face dimension {face.shape[0]} != {FACE_FEATURE_DIM}")
            if left.shape[0] != HAND_FEATURE_DIM:
                raise ValueError(f"Left-hand dimension {left.shape[0]} != {HAND_FEATURE_DIM}")
            if right.shape[0] != HAND_FEATURE_DIM:
                raise ValueError(f"Right-hand dimension {right.shape[0]} != {HAND_FEATURE_DIM}")

            pose_frames.append(pose)
            face_frames.append(face)
            left_frames.append(left)
            right_frames.append(right)
    finally:
        cap.release()

    if not pose_frames:
        raise ValueError(f"No frames read from {video_path}")

    arrays = tuple(
        np.stack(frames).astype(np.float32, copy=False)
        for frames in (pose_frames, face_frames, left_frames, right_frames)
    )
    frame_counts = {array.shape[0] for array in arrays}
    if len(frame_counts) != 1:
        raise ValueError(f"Modality frame counts differ: {sorted(frame_counts)}")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"Non-finite landmark value detected in {video_path}")
    return arrays  # type: ignore[return-value]


def extract_clip_keypoints(video_path: str, holistic) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible pose+face extraction wrapper."""
    pose, face, _, _ = extract_clip_keypoints_with_hands(video_path, holistic)
    return pose, face


def resolve_video_path(videos_dir: str | os.PathLike[str], uid: str) -> str | None:
    directory = os.fspath(videos_dir)
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    uid_casefold = uid.casefold()
    for filename in entries:
        stem, suffix = os.path.splitext(filename)
        if stem == uid or stem.casefold() == uid_casefold:
            if suffix.casefold() in SUPPORTED_VIDEO_EXTENSIONS:
                return os.path.join(directory, filename)
    return None


def resolve_uid_column(columns: list[str]) -> str:
    lower = {column.lower(): column for column in columns}
    for candidate in ("uid", "video_uid", "id"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f"No uid-like column found in CSV. Columns: {columns}")


def resolve_text_column(columns: list[str]) -> str:
    lower = {column.lower(): column for column in columns}
    for candidate in ("text", "translation", "english"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f"No text-like column found in CSV. Columns: {columns}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    import mediapipe as mp
    import pandas as pd

    os.makedirs(args.out_dir, exist_ok=True)
    pose_dir = os.path.join(args.out_dir, "pose")
    face_dir = os.path.join(args.out_dir, "face")
    left_dir = os.path.join(args.out_dir, "left_hand")
    right_dir = os.path.join(args.out_dir, "right_hand")
    for path in (pose_dir, face_dir, left_dir, right_dir):
        os.makedirs(path, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    uid_col = resolve_uid_column(list(df.columns))
    text_col = resolve_text_column(list(df.columns))

    manifest_path = os.path.join(args.out_dir, "ISLTranslate.csv")
    failures_path = os.path.join(args.out_dir, "extraction_failures.csv")
    already = newly = missing_video = failed = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_fh, open(
        failures_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as failure_fh, mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
    ) as holistic:
        manifest_writer = csv.writer(manifest_fh)
        manifest_writer.writerow(["uid", "text"])
        failure_writer = csv.writer(failure_fh)
        failure_writer.writerow(["uid", "video_path", "exception_type", "exception_message"])

        for _, row in df.iterrows():
            uid = str(row[uid_col]).strip()
            text = str(row[text_col]).strip()
            if not uid or not text:
                continue

            paths = {
                "pose": os.path.join(pose_dir, f"{uid}.npy"),
                "face": os.path.join(face_dir, f"{uid}.npy"),
                "left_hand": os.path.join(left_dir, f"{uid}.npy"),
                "right_hand": os.path.join(right_dir, f"{uid}.npy"),
            }
            try:
                existing = {name: np.load(path) if os.path.exists(path) else None for name, path in paths.items()}
                if all(value is not None for value in existing.values()):
                    shapes = [value.shape for value in existing.values()]
                    if (
                        existing["pose"].ndim == 2 and existing["pose"].shape[1] == POSE_FEATURE_DIM
                        and existing["face"].shape[1] == FACE_FEATURE_DIM
                        and existing["left_hand"].shape[1] == HAND_FEATURE_DIM
                        and existing["right_hand"].shape[1] == HAND_FEATURE_DIM
                        and len({shape[0] for shape in shapes}) == 1
                    ):
                        already += 1
                        manifest_writer.writerow([uid, text])
                        manifest_fh.flush()
                        continue

                video_path = resolve_video_path(args.videos_dir, uid)
                if video_path is None:
                    print(f"  Skipping {uid}: no supported video file matching stem")
                    missing_video += 1
                    continue

                pose, face, left, right = extract_clip_keypoints_with_hands(video_path, holistic)
                np.save(paths["pose"], pose)
                np.save(paths["face"], face)
                np.save(paths["left_hand"], left)
                np.save(paths["right_hand"], right)
                manifest_writer.writerow([uid, text])
                manifest_fh.flush()
                newly += 1

            except Exception as exc:
                failed += 1
                video_path = resolve_video_path(args.videos_dir, uid) or ""
                failure_writer.writerow([uid, video_path, type(exc).__name__, str(exc)])
                failure_fh.flush()

    total_valid = already + newly
    print(f"Already processed: {already}")
    print(f"Newly processed: {newly}")
    print(f"Failed: {failed}")
    print(f"Skipped (missing video): {missing_video}")
    print(f"Total valid outputs: {total_valid}")
    print(f"Failure report: {failures_path}")
    print(f"Validated manifest written: {manifest_path} ({total_valid} rows)")


if __name__ == "__main__":
    main()
