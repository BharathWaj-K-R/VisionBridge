from fastapi import APIRouter

from app.services.inference_service import model_status

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    status = model_status()
    return {
        "status": "ok" if status["available"] else "degraded",
        "project": "VisionBridge",
        "model": status,
    }
