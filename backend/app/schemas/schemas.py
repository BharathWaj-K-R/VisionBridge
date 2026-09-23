"""Pydantic API contracts for VisionBridge."""
import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=72)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    created_at: dt.datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AdapterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    owner_id: int
    calibration_seconds: float
    param_count: int | None
    accuracy_gain_pct: float | None
    created_at: dt.datetime


class LetterCalibrationSample(BaseModel):
    letter: str = Field(min_length=1, max_length=1, pattern=r"^[A-Za-z]$")
    hand_keypoints: list[float] = Field(min_length=126, max_length=126)


class LetterCalibrationRequest(BaseModel):
    user_id: int = Field(gt=0)
    calibration_seconds: float = Field(default=1, ge=0)
    samples: list[LetterCalibrationSample] = Field(min_length=2)


class LetterCalibrationResult(BaseModel):
    adapter_id: int
    letters: list[str]
    shots: dict[str, int]
    param_count: int


class LetterPredictionRequest(BaseModel):
    user_id: int = Field(gt=0)
    adapter_id: int = Field(gt=0)
    hand_keypoints: list[float] = Field(min_length=126, max_length=126)


class LetterPredictionResult(BaseModel):
    predicted_letter: str
    confidence: float
    latency_ms: float
    adapter_id: int


class LetterRecognitionEvent(BaseModel):
    user_id: int = Field(gt=0)
    adapter_id: int = Field(gt=0)
    predicted_letter: str = Field(min_length=1, max_length=1, pattern=r"^(?:[A-Za-z]|\?)$")
    confidence: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
