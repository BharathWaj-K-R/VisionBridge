"""Pydantic API contracts for VisionBridge."""
import datetime as dt
from collections import Counter

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

    @model_validator(mode="after")
    def validate_shot_counts(self) -> "LetterCalibrationRequest":
        counts = Counter(sample.letter.upper() for sample in self.samples)
        insufficient = sorted(letter for letter, count in counts.items() if count < 3)
        if insufficient:
            raise ValueError(
                "Each calibrated letter requires at least 3 examples: {}".format(
                    ", ".join(insufficient)
                )
            )
        return self


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
