import math

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.rate_limit import SlidingWindowRateLimiter, make_rate_limit_dependency
from app.db.models import SignerAdapter, TranslationLog, User
from app.db.session import get_db
from app.schemas.schemas import (
    LetterCalibrationRequest,
    LetterCalibrationResult,
    LetterPredictionRequest,
    LetterPredictionResult,
)
from app.services.letter_fewshot import (
    COMBINED_HAND_DIM,
    fit_prototype_adapter,
    load_prototype_adapter,
    predict_letter,
    save_prototype_adapter,
)

router = APIRouter(prefix="/letter", tags=["letter recognition"])
settings = get_settings()
_letter_limiter = SlidingWindowRateLimiter(limit=settings.TRANSLATE_RATE_LIMIT_PER_MINUTE)
_rate_limit = make_rate_limit_dependency(_letter_limiter)


def _validate_vector(values: list[float]) -> None:
    if len(values) != COMBINED_HAND_DIM:
        raise HTTPException(
            status_code=422,
            detail="hand_keypoints must contain {} floats".format(COMBINED_HAND_DIM),
        )
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(status_code=422, detail="hand_keypoints contain a non-finite value")


@router.post("/calibrate", response_model=LetterCalibrationResult, dependencies=[Depends(_rate_limit)])
def calibrate_letters(
    payload: LetterCalibrationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="user_id does not match the authenticated user")
    for sample in payload.samples:
        _validate_vector(sample.hand_keypoints)

    try:
        fitted = fit_prototype_adapter(
            [(sample.letter, sample.hand_keypoints) for sample in payload.samples]
        )
        weights_path = save_prototype_adapter(fitted["payload"])
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    row = SignerAdapter(
        owner_id=current_user.id,
        weights_path=weights_path,
        calibration_seconds=payload.calibration_seconds,
        param_count=fitted["param_count"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return LetterCalibrationResult(
        adapter_id=row.id,
        letters=fitted["letters"],
        shots=fitted["shots"],
        param_count=fitted["param_count"],
    )


@router.post("/predict", response_model=LetterPredictionResult, dependencies=[Depends(_rate_limit)])
def predict_letter_endpoint(
    payload: LetterPredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="user_id does not match the authenticated user")
    _validate_vector(payload.hand_keypoints)

    row = db.query(SignerAdapter).filter(SignerAdapter.id == payload.adapter_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Adapter not found")
    if row.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Adapter does not belong to the authenticated user")

    try:
        adapter = load_prototype_adapter(row.weights_path)
        letter, confidence, _ = predict_letter(adapter, payload.hand_keypoints)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Letter adapter is unavailable") from exc

    result = LetterPredictionResult(
        predicted_letter=letter,
        confidence=confidence,
        latency_ms=0.0,
        adapter_id=row.id,
    )
    db.add(
        TranslationLog(
            user_id=current_user.id,
            adapter_id=row.id,
            predicted_text=letter,
            confidence=confidence,
            latency_ms=0.0,
            used_adapter=1,
        )
    )
    db.commit()
    return result
