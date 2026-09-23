from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services.letter_fewshot import letter_model_status

router=APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {"status":"ok","project":"VisionBridge"}

@router.get("/ready")
def ready():
    status=letter_model_status()
    payload={"status":"ok" if status["available"] else "degraded","project":"VisionBridge","model":status}
    return JSONResponse(status_code=200 if status["available"] else 503,content=payload)
