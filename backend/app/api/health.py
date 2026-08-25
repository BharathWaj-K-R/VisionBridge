from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.inference_service import model_status

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Process liveness check. Does not require the model checkpoint to load."""
    return {"status": "ok", "project": "VisionBridge"}


@router.get("/ready")
def ready():
    """Readiness check used by deployment infrastructure.

    A process can be alive while the model is unavailable. Readiness therefore
    returns HTTP 503 until the checkpoint and vocabulary are compatible.
    """
    status = model_status()
    payload = {
        "status": "ok" if status["available"] else "degraded",
        "project": "VisionBridge",
        "model": status,
    }
    return JSONResponse(
        status_code=200 if status["available"] else 503,
        content=payload,
    )
