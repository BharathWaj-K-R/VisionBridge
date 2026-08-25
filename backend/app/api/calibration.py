import math

import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.models import SignerAdapter, User
from app.db.session import get_db
from app.models.base_model import FACE_INPUT_DIM, POSE_INPUT_DIM
from app.schemas.schemas import CalibrationRequest, CalibrationResult
from app.services.calibration_service import calibrate_new_adapter
from app.services.inference_service import ModelUnavailableError, get_base_model

router = APIRouter(prefix="/calibration", tags=["calibration"])
settings = get_settings()


def _ctc_min_input_length(target_labels: list[int]) -> int:
    if not target_labels:
        return 0
    repeats = sum(a == b for a, b in zip(target_labels, target_labels[1:]))
    return len(target_labels) + repeats


def _validate_calibration_payload(payload: CalibrationRequest) -> None:
    if payload.calibration_seconds < settings.CALIBRATION_MIN_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"calibration_seconds must be at least {settings.CALIBRATION_MIN_SECONDS}; "
                f"received {payload.calibration_seconds}"
            ),
        )
    if len(payload.pose_keypoints) == 0 or len(payload.face_keypoints) == 0:
        raise HTTPException(status_code=422, detail="pose_keypoints and face_keypoints must not be empty")
    if len(payload.pose_keypoints) != len(payload.face_keypoints):
        raise HTTPException(status_code=422, detail="pose/face frame count mismatch")
    if len(payload.pose_keypoints) > settings.MAX_INFERENCE_FRAMES:
        raise HTTPException(status_code=422, detail=f"too many frames; maximum is {settings.MAX_INFERENCE_FRAMES}")
    required_frames = _ctc_min_input_length(payload.target_labels)
    if required_frames > len(payload.pose_keypoints):
        repeats = required_frames - len(payload.target_labels)
        raise HTTPException(
            status_code=422,
            detail=(
                "target_labels cannot align under CTC for this clip: "
                f"target_length={len(payload.target_labels)}, repeated_labels={repeats}, "
                f"minimum_required_frames={required_frames}, input_frames={len(payload.pose_keypoints)}"
            ),
        )
    if any(len(frame) != POSE_INPUT_DIM for frame in payload.pose_keypoints):
        raise HTTPException(status_code=422, detail=f"pose frames must have {POSE_INPUT_DIM} dimensions")
    if any(len(frame) != FACE_INPUT_DIM for frame in payload.face_keypoints):
        raise HTTPException(status_code=422, detail=f"face frames must have {FACE_INPUT_DIM} dimensions")
    if any(not math.isfinite(value) for frame in payload.pose_keypoints for value in frame):
        raise HTTPException(status_code=422, detail="pose_keypoints contain a non-finite value")
    if any(not math.isfinite(value) for frame in payload.face_keypoints for value in frame):
        raise HTTPException(status_code=422, detail="face_keypoints contain a non-finite value")


@router.post("", response_model=CalibrationResult)
def calibrate(
    payload: CalibrationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Train a new BridgeAdapter for the authenticated signer.

    The caller's token is authoritative for ownership. `payload.user_id` is
    retained for backwards-compatible request shape, but it must match the
    authenticated user and can never be used to impersonate another signer.
    """
    _validate_calibration_payload(payload)
    if payload.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="user_id does not match the authenticated user")

    try:
        base_model = get_base_model()
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Calibration model is unavailable") from exc
    if any(label <= 0 or label >= base_model.output_head.out_features for label in payload.target_labels):
        raise HTTPException(status_code=422, detail="target_labels contain an invalid vocabulary id")

    pose = torch.tensor(payload.pose_keypoints, dtype=torch.float32).unsqueeze(0)
    face = torch.tensor(payload.face_keypoints, dtype=torch.float32).unsqueeze(0)
    target_labels = torch.tensor(payload.target_labels, dtype=torch.long).unsqueeze(0)
    target_lengths = torch.tensor([len(payload.target_labels)], dtype=torch.long)

    try:
        result = calibrate_new_adapter(pose, face, target_labels, target_lengths, payload.calibration_seconds)
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
