from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import SignerAdapter, TranslationLog, User
from app.db.session import get_db
from app.services.inference_service import model_status

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("")
def dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    logs = (
        db.query(TranslationLog)
        .filter(TranslationLog.user_id == current_user.id)
        .order_by(TranslationLog.created_at.desc())
        .limit(10)
        .all()
    )
    total_sessions = db.query(func.count(TranslationLog.id)).filter(
        TranslationLog.user_id == current_user.id
    ).scalar() or 0
    avg_confidence = db.query(func.avg(TranslationLog.confidence)).filter(
        TranslationLog.user_id == current_user.id,
        TranslationLog.confidence.isnot(None),
    ).scalar()
    avg_latency = db.query(func.avg(TranslationLog.latency_ms)).filter(
        TranslationLog.user_id == current_user.id,
        TranslationLog.latency_ms.isnot(None),
    ).scalar()
    adapter = (
        db.query(SignerAdapter)
        .filter(SignerAdapter.owner_id == current_user.id)
        .order_by(SignerAdapter.created_at.desc())
        .first()
    )

    return {
        "user": {"id": current_user.id, "username": current_user.username},
        "model": model_status(),
        "adapter": (
            {
                "id": adapter.id,
                "calibration_seconds": adapter.calibration_seconds,
                "param_count": adapter.param_count,
                "accuracy_gain_pct": adapter.accuracy_gain_pct,
                "created_at": adapter.created_at,
            }
            if adapter
            else None
        ),
        "usage": {
            "translation_events": int(total_sessions),
            "average_confidence": float(avg_confidence) if avg_confidence is not None else None,
            "average_latency_ms": float(avg_latency) if avg_latency is not None else None,
        },
        "recent_activity": [
            {
                "id": item.id,
                "predicted_text": item.predicted_text,
                "confidence": item.confidence,
                "latency_ms": item.latency_ms,
                "used_adapter": bool(item.used_adapter),
                "created_at": item.created_at,
            }
            for item in logs
        ],
    }
