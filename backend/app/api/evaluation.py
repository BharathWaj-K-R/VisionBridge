from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.models import SignerAdapter, User
from app.db.session import get_db
from app.services.inference_service import model_status

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
settings = get_settings()


@router.get("")
def evaluation(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    adapters = (
        db.query(SignerAdapter)
        .filter(SignerAdapter.owner_id == current_user.id)
        .order_by(SignerAdapter.created_at.desc())
        .all()
    )
    measured = [a for a in adapters if a.accuracy_gain_pct is not None]
    return {
        "model": model_status(),
        "base_model_path_configured": bool(settings.BASE_MODEL_PATH),
        "evaluation_data_available": False,
        "message": "No persisted benchmark run is available. Accuracy, BLEU, WER, and memory remain unmeasured.",
        "adapters": [
            {
                "id": a.id,
                "calibration_seconds": a.calibration_seconds,
                "param_count": a.param_count,
                "accuracy_gain_pct": a.accuracy_gain_pct,
                "measured_gain_available": a.accuracy_gain_pct is not None,
            }
            for a in adapters
        ],
        "measured_adapter_count": len(measured),
    }
