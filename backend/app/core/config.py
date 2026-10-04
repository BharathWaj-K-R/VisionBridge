"""Environment-backed application settings."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings:
    PROJECT_NAME = "VisionBridge"
    API_V1_PREFIX = "/api/v1"

    ENV = os.getenv("ENV", "development")
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BACKEND_DIR / 'visionbridge.db'}",
    )
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    JWT_ALGORITHM = "HS256"

    AUTH_COOKIE_NAME = "__Host-visionbridge_session"
    CSRF_COOKIE_NAME = "__Host-visionbridge_csrf"
    AUTH_COOKIE_SECURE = os.getenv(
        "AUTH_COOKIE_SECURE",
        "true" if ENV.lower() == "production" else "false",
    ).lower() == "true"
    AUTH_COOKIE_SAMESITE = os.getenv(
        "AUTH_COOKIE_SAMESITE",
        "none" if ENV.lower() == "production" else "lax",
    ).lower()

    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173").rstrip("/")

    ALLOWED_ORIGINS = [
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

    LETTER_BASE_MODEL_PATH = os.getenv(
        "LETTER_BASE_MODEL_PATH",
        str(BACKEND_DIR / "app/models/weights/letter_base_model.pt"),
    )
    ADAPTER_WEIGHTS_DIR = os.getenv(
        "ADAPTER_WEIGHTS_DIR",
        str(BACKEND_DIR / "app/models/weights/adapters"),
    )

    TRANSLATE_RATE_LIMIT_PER_MINUTE = int(os.getenv("TRANSLATE_RATE_LIMIT_PER_MINUTE", "60"))
    CALIBRATION_RATE_LIMIT_PER_MINUTE = int(os.getenv("CALIBRATION_RATE_LIMIT_PER_MINUTE", "5"))

    REGISTER_RATE_LIMIT_PER_MINUTE = int(os.getenv("REGISTER_RATE_LIMIT_PER_MINUTE", "5"))
    LOGIN_RATE_LIMIT_PER_MINUTE = int(os.getenv("LOGIN_RATE_LIMIT_PER_MINUTE", "5"))

    def validate_for_runtime(self) -> None:
        if self.ENV.lower() == "production":
            if self.SECRET_KEY == "dev-secret-change-me":
                raise RuntimeError("SECRET_KEY must be set to a non-default value in production")
            if self.DATABASE_URL.lower().startswith("sqlite"):
                raise RuntimeError(
                    "DATABASE_URL must point to durable PostgreSQL in production; SQLite fallback is disabled"
                )

        if self.AUTH_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
            raise RuntimeError("AUTH_COOKIE_SAMESITE must be lax, strict, or none")

        if self.AUTH_COOKIE_SAMESITE == "none" and not self.AUTH_COOKIE_SECURE:
            raise RuntimeError("AUTH_COOKIE_SECURE must be true when AUTH_COOKIE_SAMESITE is none")

        if self.TRANSLATE_RATE_LIMIT_PER_MINUTE < 1:
            raise RuntimeError("TRANSLATE_RATE_LIMIT_PER_MINUTE must be at least 1")

        if self.CALIBRATION_RATE_LIMIT_PER_MINUTE < 1:
            raise RuntimeError("CALIBRATION_RATE_LIMIT_PER_MINUTE must be at least 1")

        if self.REGISTER_RATE_LIMIT_PER_MINUTE < 1:
            raise RuntimeError("REGISTER_RATE_LIMIT_PER_MINUTE must be at least 1")

        if self.LOGIN_RATE_LIMIT_PER_MINUTE < 1:
            raise RuntimeError("LOGIN_RATE_LIMIT_PER_MINUTE must be at least 1")



@lru_cache
def get_settings() -> Settings:
    return Settings()
