from pathlib import Path
import math
import time
from pathlib import Path

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
    LetterRecognitionEvent,
)
from app.models.letter_model import build_browser_payload
from app.services.letter_fewshot import (
    COMBINED_HAND_DIM,
    _sha256,
    fit_prototype_adapter,
    get_letter_base_model,
    letter_model_status,
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
    if all(value == 0 for value in values):
        raise HTTPException(status_code=422, detail="At least one hand landmark must be visible")


@router.get("/status")
def status():
    return letter_model_status()



@router.get("/model")
def browser_model(
    current_user: User = Depends(get_current_user),
):
    target = Path(settings.LETTER_BASE_MODEL_PATH)
    try:
        base_model = get_letter_base_model()
        return build_browser_payload(base_model, _sha256(target))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Letter base model is not trained yet") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Letter base model is unavailable") from exc

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
        base_model = get_letter_base_model()
        fitted = fit_prototype_adapter(
            base_model,
            [(sample.letter, sample.hand_keypoints) for sample in payload.samples],
        )
        weights_path = save_prototype_adapter(fitted["payload"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Letter base model is not trained yet") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Letter base model is unavailable") from exc

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




@router.get("/adapters/{adapter_id}")
def get_letter_adapter(
    adapter_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = (
        db.query(SignerAdapter)
        .filter(SignerAdapter.id == adapter_id, SignerAdapter.owner_id == current_user.id)
        .first()
    )
    if not row or not Path(row.weights_path).name.startswith("letter_adapter_"):
        raise HTTPException(status_code=404, detail="Adapter not found")

    try:
        return load_prototype_adapter(row.weights_path, settings.LETTER_BASE_MODEL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Letter model or adapter is unavailable") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail="Adapter requires recalibration for the current model") from exc



@router.post("/event")
def log_letter_event(
    payload: LetterRecognitionEvent,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="user_id does not match the authenticated user")

    if payload.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="user_id does not match the authenticated user")

    adapter_id = payload.adapter_id
    predicted_letter = payload.predicted_letter.upper()
    confidence = payload.confidence
    latency_ms = payload.latency_ms
    if predicted_letter != "?" and predicted_letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raise HTTPException(status_code=422, detail="Invalid predicted letter")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise HTTPException(status_code=422, detail="Invalid confidence")
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise HTTPException(status_code=422, detail="Invalid latency")

    adapter = (
        db.query(SignerAdapter)
        .filter(SignerAdapter.id == adapter_id, SignerAdapter.owner_id == current_user.id)
        .first()
    )
    if not adapter or not Path(adapter.weights_path).name.startswith("letter_adapter_"):
        raise HTTPException(status_code=404, detail="Adapter not found")

    db.add(
        TranslationLog(
            user_id=current_user.id,
            adapter_id=adapter_id,
            predicted_text=predicted_letter,
            confidence=confidence,
            latency_ms=latency_ms,
            used_adapter=1,
        )
    )
    db.commit()
    return {"logged": True}

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

    started = time.perf_counter()
    try:
        base_model = get_letter_base_model()
        adapter = load_prototype_adapter(row.weights_path, settings.LETTER_BASE_MODEL_PATH)
        letter, confidence, _ = predict_letter(
            base_model,
            adapter,
            payload.hand_keypoints,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Letter base model or adapter is unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Letter base model or adapter is incompatible") from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Letter recognition is unavailable") from exc

    latency_ms = (time.perf_counter() - started) * 1000
    result = LetterPredictionResult(
        predicted_letter=letter,
        confidence=confidence,
        latency_ms=latency_ms,
        adapter_id=row.id,
    )
    db.add(
        TranslationLog(
            user_id=current_user.id,
            adapter_id=row.id,
            predicted_text=letter,
            confidence=confidence,
            latency_ms=latency_ms,
            used_adapter=1,
        )
    )
    db.commit()
    return result
