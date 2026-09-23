"""
Application configuration.
Reads from environment variables so the same code runs locally and on Render.
"""
import os
from functools import lru_cache
from pathlib import Path

BACKEND_DIR=Path(__file__).resolve().parents[2]

class Settings:
    PROJECT_NAME="VisionBridge"
    API_V1_PREFIX="/api/v1"
    ENV=os.getenv("ENV","development")
    DATABASE_URL=os.getenv("DATABASE_URL",f"sqlite:///{BACKEND_DIR / 'visionbridge.db'}")
    SECRET_KEY=os.getenv("SECRET_KEY","dev-secret-change-me")
    ACCESS_TOKEN_EXPIRE_MINUTES=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES","60"))
    JWT_ALGORITHM="HS256"
    ALLOWED_ORIGINS=[origin.strip() for origin in os.getenv("ALLOWED_ORIGINS","http://localhost:5500,http://127.0.0.1:5500").split(",") if origin.strip()]
    BASE_MODEL_PATH=os.getenv("BASE_MODEL_PATH",str(BACKEND_DIR / "app/models/weights/base_model.pt"))
    LETTER_BASE_MODEL_PATH=os.getenv("LETTER_BASE_MODEL_PATH",str(BACKEND_DIR / "app/models/weights/letter_base_model.pt"))
    ADAPTER_WEIGHTS_DIR=os.getenv("ADAPTER_WEIGHTS_DIR",str(BACKEND_DIR / "app/models/weights/adapters"))
    CALIBRATION_MIN_SECONDS=int(os.getenv("CALIBRATION_MIN_SECONDS","300"))
    CALIBRATION_MAX_FRAMES=int(os.getenv("CALIBRATION_MAX_FRAMES","256"))
    MAX_INFERENCE_FRAMES=int(os.getenv("MAX_INFERENCE_FRAMES","1024"))
    TRANSLATE_RATE_LIMIT_PER_MINUTE=int(os.getenv("TRANSLATE_RATE_LIMIT_PER_MINUTE","60"))
    CALIBRATION_RATE_LIMIT_PER_MINUTE=int(os.getenv("CALIBRATION_RATE_LIMIT_PER_MINUTE","5"))

    def validate_for_runtime(self)->None:
        if self.ENV.lower()=="production" and self.SECRET_KEY=="dev-secret-change-me": raise RuntimeError("SECRET_KEY must be set to a non-default value in production")
        if self.MAX_INFERENCE_FRAMES<1: raise RuntimeError("MAX_INFERENCE_FRAMES must be at least 1")
        if self.CALIBRATION_MAX_FRAMES<1: raise RuntimeError("CALIBRATION_MAX_FRAMES must be at least 1")
        if self.CALIBRATION_MAX_FRAMES>self.MAX_INFERENCE_FRAMES: raise RuntimeError("CALIBRATION_MAX_FRAMES cannot exceed MAX_INFERENCE_FRAMES")
        if self.TRANSLATE_RATE_LIMIT_PER_MINUTE<1: raise RuntimeError("TRANSLATE_RATE_LIMIT_PER_MINUTE must be at least 1")
        if self.CALIBRATION_RATE_LIMIT_PER_MINUTE<1: raise RuntimeError("CALIBRATION_RATE_LIMIT_PER_MINUTE must be at least 1")

@lru_cache
def get_settings()->Settings:
    return Settings()
