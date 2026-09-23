"""Environment-backed application settings."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings:
    project_name = "VisionBridge"
    api_v1_prefix = "/api/v1"

    env = os.getenv("ENV", "development")
    database_url = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BACKEND_DIR / 'visionbridge.db'}",
    )
    secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
    access_token_expire_minutes = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )
    jwt_algorithm = "HS256"

    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            ",".join(
                [
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:5500",
                    "http://127.0.0.1:5500",
                ]
            ),
        ).split(",")
        if origin.strip()
    ]

    letter_base_model_path = os.getenv(
        "LETTER_BASE_MODEL_PATH",
        str(BACKEND_DIR / "app/models/weights/letter_base_model.pt"),
    )
    adapter_weights_dir = os.getenv(
        "ADAPTER_WEIGHTS_DIR",
        str(BACKEND_DIR / "app/models/weights/adapters"),
    )

    translate_rate_limit_per_minute = int(
        os.getenv("TRANSLATE_RATE_LIMIT_PER_MINUTE", "60")
    )
    calibration_rate_limit_per_minute = int(
        os.getenv("CALIBRATION_RATE_LIMIT_PER_MINUTE", "5")
    )

    def validate_for_runtime(self) -> None:
        if self.env.lower() == "production" and self.secret_key == "dev-secret-change-me":
            raise RuntimeError("SECRET_KEY must be set to a non-default value in production")

        if self.translate_rate_limit_per_minute < 1:
            raise RuntimeError("TRANSLATE_RATE_LIMIT_PER_MINUTE must be at least 1")

        if self.calibration_rate_limit_per_minute < 1:
            raise RuntimeError("CALIBRATION_RATE_LIMIT_PER_MINUTE must be at least 1")


@lru_cache
def get_settings() -> Settings:
    return Settings()
