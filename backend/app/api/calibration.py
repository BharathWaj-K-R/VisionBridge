import math
from pathlib import Path

import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.models import SignerAdapter, User
from app.db.session import get_db
from app.models.base_model import FACE_INPUT_DIM, HAND_INPUT_DIM, POSE_INPUT_DIM
from app.schemas.schemas import CalibrationRequest, CalibrationResult
from app.services.calibration_service import calibrate_new_adapter
from app.services.inference_service import ModelUnavailableError, get_base_model
from app.training.isltranslate import SimpleCharTokenizer

router = APIRouter(prefix="/calibration", tags=["calibration"])
settings = get_settings()


def _ctc_min_input_length(target_labels: list[int]) -> int:
    if not target_labels:
        return 0
    repeats = sum(a == b for a, b in zip(target_labels, target_labels[1:]))
    return len(target_labels) + repeats


def _resolve_target_labels(payload: CalibrationRequest) -> list[int]:
    if payload.target_labels is not None:
        return payload.target_labels
    vocab_path = Path(settings.BASE_MODEL_PATH).with_suffix(".vocab.json")
    if not vocab_path.is_file():
        raise HTTPException(status_code=503, detail="Calibration model vocabulary is unavailable")
    try:
        tokenizer = SimpleCharTokenizer.load(vocab_path)
        return tokenizer.encode(payload.target_text or "")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _validate_calibration_payload(
    payload: CalibrationRequest,
    target_labels: list[int],
) -> None:
    if payload.calibration_seconds < settings.CALIBRATION_MIN_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"calibration_seconds must be at least {settings.CALIBRATION_MIN_SECONDS}; "
                f"received {payload.calibration_seconds}"
            ),
        )
    if not payload.pose_keypoints or not payload.face_keypoints:
        raise HTTPException(status_code=422, detail="pose_keypoints and face_keypoints must not be empty")
    if payload.left_hand_keypoints is None or payload.right_hand_keypoints is None:
        raise HTTPException(
            status_code=422,
            detail="left_hand_keypoints and right_hand_keypoints are required by the hand-aware model",
        )
    frame_count = len(payload.pose_keypoints)
    for name, frames in (
        ("face_keypoints", payload.face_keypoints),
        ("left_hand_keypoints", payload.left_hand_keypoints),
        ("right_hand_keypoints", payload.right_hand_keypoints),
    ):
        if len(frames) != frame_count:
            raise HTTPException(
                status_code=422,
                detail=f"{name} frame count mismatch: {len(frames)} vs {frame_count}",
            )
    if frame_count > settings.MAX_INFERENCE_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"too many frames; maximum is {settings.MAX_INFERENCE_FRAMES}",
        )

    required_frames = _ctc_min_input_length(target_labels)
    if not target_labels:
        raise HTTPException(
            status_code=422,
            detail="Calibration target is empty or contains no supported vocabulary characters",
        )
    if required_frames > min(frame_count, settings.CALIBRATION_MAX_FRAMES):
        repeats = required_frames - len(target_labels)
        raise HTTPException(
            status_code=422,
            detail=(
                "target cannot align under CTC after calibration downsampling: "
                f"target_length={len(target_labels)}, repeated_labels={repeats}, "
                f"minimum_required_frames={required_frames}, "
                f"calibration_max_frames={settings.CALIBRATION_MAX_FRAMES}"
            ),
        )

    expected = {
        "pose_keypoints": POSE_INPUT_DIM,
        "face_keypoints": FACE_INPUT_DIM,
        "left_hand_keypoints": HAND_INPUT_DIM,
        "right_hand_keypoints": HAND_INPUT_DIM,
    }
    for name, frames in (
        ("pose_keypoints", payload.pose_keypoints),
        ("face_keypoints", payload.face_keypoints),
        ("left_hand_keypoints", payload.left_hand_keypoints),
        ("right_hand_keypoints", payload.right_hand_keypoints),
    ):
        dimension = expected[name]
        for idx, frame in enumerate(frames):
            if len(frame) != dimension:
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} frame {idx} has {len(frame)} dims, expected {dimension}",
                )
            if not all(math.isfinite(value) for value in frame):
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} frame {idx} contains a non-finite value",
                )


def _downsample(
    pose: list[list[float]],
    face: list[list[float]],
    left_hand: list[list[float]],
    right_hand: list[list[float]],
) -> tuple[list[list[float]], ...]:
    if len(pose) <= settings.CALIBRATION_MAX_FRAMES:
        return pose, face, left_hand, right_hand
    indices = torch.linspace(
        0,
        len(pose) - 1,
        steps=settings.CALIBRATION_MAX_FRAMES,
    ).long().tolist()
    return (
        [pose[i] for i in indices],
        [face[i] for i in indices],
        [left_hand[i] for i in indices],
        [right_hand[i] for i in indices],
    )


@router.post("", response_model=CalibrationResult)
def calibrate(
    payload: CalibrationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="user_id does not match the authenticated user")

    target_labels = _resolve_target_labels(payload)
    _validate_calibration_payload(payload, target_labels)

    pose_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints = _downsample(
        payload.pose_keypoints,
        payload.face_keypoints,
        payload.left_hand_keypoints,  # type: ignore[arg-type]
        payload.right_hand_keypoints,  # type: ignore[arg-type]
    )

    try:
        base_model = get_base_model()
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Calibration model is unavailable") from exc

    if any(label <= 0 or label >= base_model.output_head.out_features for label in target_labels):
        raise HTTPException(status_code=422, detail="target labels contain an invalid vocabulary id")

    pose = torch.tensor(pose_keypoints, dtype=torch.float32).unsqueeze(0)
    face = torch.tensor(face_keypoints, dtype=torch.float32).unsqueeze(0)
    left_hand = torch.tensor(left_hand_keypoints, dtype=torch.float32).unsqueeze(0)
    right_hand = torch.tensor(right_hand_keypoints, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(target_labels, dtype=torch.long).unsqueeze(0)
    target_lengths = torch.tensor([len(target_labels)], dtype=torch.long)

    try:
        result = calibrate_new_adapter(
            pose,
            face,
            left_hand,
            right_hand,
            target_tensor,
            target_lengths,
            payload.calibration_seconds,
        )
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Calibration model is unavailable") from exc

    adapter_row = SignerAdapter(
        owner_id=current_user.id,
        weights_path=result["weights_path"],
        calibration_seconds=result["calibration_seconds"],
        param_count=result["param_count"],
    )
    db.add(adapter_row)
    db.commit()
    db.refresh(adapter_row)

    return CalibrationResult(
        adapter_id=adapter_row.id,
        calibration_seconds=adapter_row.calibration_seconds,
        param_count=adapter_row.param_count,
    )
