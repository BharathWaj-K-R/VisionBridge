import math

import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_optional_current_user
from app.core.config import get_settings
from app.db.models import SignerAdapter, TranslationLog, User
from app.db.session import get_db
from app.models.base_model import FACE_INPUT_DIM, HAND_INPUT_DIM, POSE_INPUT_DIM
from app.schemas.schemas import TranslationRequest, TranslationResult
from app.services.calibration_service import load_adapter_for_signer
from app.services.inference_service import ModelUnavailableError, get_base_model, run_inference

router = APIRouter(prefix="/translate", tags=["translate"])
settings = get_settings()


def _validate_keypoints(
    pose_keypoints: list[list[float]],
    face_keypoints: list[list[float]],
    left_hand_keypoints: list[list[float]] | None,
    right_hand_keypoints: list[list[float]] | None,
) -> None:
    """Validate synchronized multimodal landmark payloads before tensor conversion."""
    if not pose_keypoints or not face_keypoints:
        raise HTTPException(status_code=422, detail="pose_keypoints and face_keypoints must not be empty")
    if len(pose_keypoints) != len(face_keypoints):
        raise HTTPException(
            status_code=422,
            detail=f"pose/face frame count mismatch: {len(pose_keypoints)} vs {len(face_keypoints)}",
        )
    if left_hand_keypoints is None or right_hand_keypoints is None:
        raise HTTPException(
            status_code=422,
            detail="left_hand_keypoints and right_hand_keypoints are required by the hand-aware model",
        )
    for name, frames in (
        ("left_hand_keypoints", left_hand_keypoints),
        ("right_hand_keypoints", right_hand_keypoints),
    ):
        if len(frames) != len(pose_keypoints):
            raise HTTPException(
                status_code=422,
                detail=f"{name} frame count mismatch: {len(frames)} vs {len(pose_keypoints)}",
            )
    if len(pose_keypoints) > settings.MAX_INFERENCE_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"too many frames: {len(pose_keypoints)}, maximum is {settings.MAX_INFERENCE_FRAMES}",
        )

    for idx, frame in enumerate(pose_keypoints):
        if len(frame) != POSE_INPUT_DIM:
            raise HTTPException(status_code=422, detail=f"pose frame {idx} has {len(frame)} dims, expected {POSE_INPUT_DIM}")
        if not all(math.isfinite(value) for value in frame):
            raise HTTPException(status_code=422, detail=f"pose frame {idx} contains a non-finite value")
    for idx, frame in enumerate(face_keypoints):
        if len(frame) != FACE_INPUT_DIM:
            raise HTTPException(status_code=422, detail=f"face frame {idx} has {len(frame)} dims, expected {FACE_INPUT_DIM}")
        if not all(math.isfinite(value) for value in frame):
            raise HTTPException(status_code=422, detail=f"face frame {idx} contains a non-finite value")
    for name, frames in (("left_hand_keypoints", left_hand_keypoints), ("right_hand_keypoints", right_hand_keypoints)):
        for idx, frame in enumerate(frames):
            if len(frame) != HAND_INPUT_DIM:
                raise HTTPException(status_code=422, detail=f"{name} frame {idx} has {len(frame)} dims, expected {HAND_INPUT_DIM}")
            if not all(math.isfinite(value) for value in frame):
                raise HTTPException(status_code=422, detail=f"{name} frame {idx} contains a non-finite value")


@router.post("", response_model=TranslationResult)
def translate(
    payload: TranslationRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
):
    """Translate synchronized pose, face, left-hand, and right-hand keypoints."""
    _validate_keypoints(
        payload.pose_keypoints,
        payload.face_keypoints,
        payload.left_hand_keypoints,
        payload.right_hand_keypoints,
    )

    if payload.user_id is not None:
        if current_user is None:
            raise HTTPException(status_code=401, detail="Authentication required for user-scoped translation")
        if current_user.id != payload.user_id:
            raise HTTPException(status_code=403, detail="user_id does not match the authenticated user")

    pose = torch.tensor(payload.pose_keypoints, dtype=torch.float32).unsqueeze(0)
    face = torch.tensor(payload.face_keypoints, dtype=torch.float32).unsqueeze(0)
    left_hand = torch.tensor(payload.left_hand_keypoints, dtype=torch.float32).unsqueeze(0)
    right_hand = torch.tensor(payload.right_hand_keypoints, dtype=torch.float32).unsqueeze(0)

    adapter = None
    if payload.adapter_id is not None:
        if current_user is None:
            raise HTTPException(status_code=401, detail="Authentication required to use a signer adapter")
        row = db.query(SignerAdapter).filter(SignerAdapter.id == payload.adapter_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Adapter not found")
        if row.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Adapter does not belong to the authenticated user")
        try:
            base_model = get_base_model()
            adapter = load_adapter_for_signer(
                row.weights_path,
                d_model=base_model.d_model,
                n_layers=len(base_model.shared_encoder.layers),
            )
        except (ModelUnavailableError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail="Translation model is unavailable") from exc

    try:
        result = run_inference(
            pose,
            face,
            left_hand,
            right_hand,
            adapter=adapter,
        )
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Translation model is unavailable") from exc

    log = TranslationLog(
        user_id=current_user.id if current_user is not None and payload.user_id is not None else None,
        adapter_id=payload.adapter_id,
        predicted_text=result["predicted_text"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        used_adapter=int(result["used_adapter"]),
    )
    db.add(log)
    db.commit()

    return TranslationResult(**result)
