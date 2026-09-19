"""
Convert iSign/.pose-format keypoint files into the exact layout
app/training/isltranslate.py's ISLTranslateKeypointDataset expects:

    data/processed/isltranslate/
    ├── ISLTranslate.csv
    ├── pose/<uid>.npy   (frames, 132)
    └── face/<uid>.npy   (frames, 1404)

Use this if the dataset already ships pre-extracted MediaPipe Holistic
keypoints in .pose-format. If you only have raw video, use
extract_keypoints.py instead.

IMPORTANT:
This converter is intentionally strict. VisionBridge's current model uses
468 face landmarks = 1404 values/frame (468 * 3). A .pose file exposing 478
face landmarks (478 * 3 values/frame) is NOT compatible and must not be
written into the training directory. The previous version only warned and
still saved incompatible arrays, creating a delayed dimensional failure.

Before converting a new iSign release:
  1. Run: python scripts/convert_isign_pose.py --inspect_only <one_file.pose>
  2. Confirm the component names and point counts.
  3. Only proceed when FACE_LANDMARKS resolves to 468 points (1404 values).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

POSE_COMPONENT = "POSE_LANDMARKS"
FACE_COMPONENT = "FACE_LANDMARKS"
EXPECTED_POSE_DIM = 132   # 33 landmarks * 4 (x,y,z,confidence)
EXPECTED_FACE_DIM = 1404  # 468 landmarks * 3 (x,y,z)


def inspect_pose_file(pose_path: str):
    from pose_format import Pose

    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    print(f"File: {pose_path}")
    print(f"Total frames: {pose.body.data.shape[0]}")
    for component in pose.header.components:
        print(f"  Component: {component.name}, points: {len(component.points)}")


def extract_component(pose, component_name: str) -> np.ndarray:
    matches = [
        (i, component)
        for i, component in enumerate(pose.header.components)
        if component.name == component_name
    ]
    if not matches:
        raise ValueError(
            f"Missing required pose-format component {component_name!r}; "
            f"available={[component.name for component in pose.header.components]}"
        )
    component_idx, component = matches[0]
    start = sum(len(c.points) for c in pose.header.components[:component_idx])
    n_points = len(component.points)
    data = pose.body.data[:, 0, start:start + n_points, :]  # single person
    return data.reshape(data.shape[0], -1).astype(np.float32)


def convert_file(pose_path: str, uid: str, pose_dir: str, face_dir: str):
    from pose_format import Pose

    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    pose_arr = extract_component(pose, POSE_COMPONENT)
    face_arr = extract_component(pose, FACE_COMPONENT)

    if pose_arr.ndim != 2 or pose_arr.shape[1] != EXPECTED_POSE_DIM:
        raise ValueError(
            f"{uid}: incompatible pose shape {tuple(pose_arr.shape)}; "
            f"expected (frames, {EXPECTED_POSE_DIM})"
        )
    if face_arr.ndim != 2 or face_arr.shape[1] != EXPECTED_FACE_DIM:
        raise ValueError(
            f"{uid}: incompatible face shape {tuple(face_arr.shape)}; "
            f"expected (frames, {EXPECTED_FACE_DIM}). "
            "Do not feed a 478-landmark (478 * 3 values/frame) face stream into this model."
        )
    if pose_arr.shape[0] != face_arr.shape[0] or pose_arr.shape[0] == 0:
        raise ValueError(
            f"{uid}: pose/face frame mismatch or empty clip: "
            f"pose_frames={pose_arr.shape[0]}, face_frames={face_arr.shape[0]}"
        )
    if not np.isfinite(pose_arr).all() or not np.isfinite(face_arr).all():
        raise ValueError(f"{uid}: non-finite values detected in converted keypoints")

    np.save(os.path.join(pose_dir, f"{uid}.npy"), pose_arr)
    np.save(os.path.join(face_dir, f"{uid}.npy"), face_arr)


def resolve_uid_column(columns: list[str]) -> str:
    lower = {c.lower(): c for c in columns}
    for candidate in ("uid", "video_uid", "id"):
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f"No uid-like column found in CSV. Columns: {columns}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_dir", help="Folder of .pose files")
    parser.add_argument("--labels_csv", help="iSign metadata CSV")
    parser.add_argument("--out_dir", help="e.g. data/processed/isltranslate")
    parser.add_argument("--inspect_only", metavar="POSE_FILE",
                        help="Just inspect one file's component layout and exit")
    args = parser.parse_args()

    if args.inspect_only:
        inspect_pose_file(args.inspect_only)
        return

    if not (args.pose_dir and args.labels_csv and args.out_dir):
        raise SystemExit("--pose_dir, --labels_csv, and --out_dir are all required (or use --inspect_only alone)")

    pose_dir_out = os.path.join(args.out_dir, "pose")
    face_dir_out = os.path.join(args.out_dir, "face")
    os.makedirs(pose_dir_out, exist_ok=True)
    os.makedirs(face_dir_out, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    uid_col = resolve_uid_column(list(df.columns))
    valid_uids = set(df[uid_col].astype(str))

    pose_files = sorted(glob.glob(os.path.join(args.pose_dir, "*.pose")))
    if not pose_files:
        raise SystemExit(f"No .pose files found in {args.pose_dir}")

    converted = 0
    failed = 0
    for pose_path in pose_files:
        uid = os.path.splitext(os.path.basename(pose_path))[0]
        if uid not in valid_uids:
            print(f"  Skipping {uid}: not found in {args.labels_csv}")
            continue
        try:
            convert_file(pose_path, uid, pose_dir_out, face_dir_out)
            converted += 1
            print(f"[{converted}] {uid} converted")
        except Exception as exc:
            failed += 1
            print(f"  FAILED {uid}: {type(exc).__name__}: {exc}")
            continue

    if converted == 0:
        raise RuntimeError("No compatible iSign .pose files were converted.")

    print(f"Done. Converted {converted}/{len(pose_files)} files; failed={failed}. "
          f"Output is compatible with the current 132/1404 model contract.")


if __name__ == "__main__":
    main()
