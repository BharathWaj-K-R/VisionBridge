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


class CalibrationStartRequest(BaseModel):
    user_id: int


class CalibrationRequest(BaseModel):
    user_id: int = Field(gt=0)
    calibration_seconds: float = Field(gt=0)
    pose_keypoints: list[list[float]]
    face_keypoints: list[list[float]]
    left_hand_keypoints: list[list[float]] | None = None
    right_hand_keypoints: list[list[float]] | None = None
    target_labels: list[int] | None = Field(default=None, min_length=1)
    target_text: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def require_target(self):
        if self.target_labels is None and not self.target_text:
            raise ValueError("Provide target_labels or target_text")
        if self.target_labels is not None and self.target_text:
            raise ValueError("Provide only one of target_labels or target_text")
        return self


class CalibrationResult(BaseModel):
    adapter_id: int
    calibration_seconds: float
    param_count: int
    accuracy_gain_pct: float | None = None


class AdapterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    owner_id: int
    calibration_seconds: float
    param_count: int | None
    accuracy_gain_pct: float | None
    created_at: dt.datetime


class TranslationRequest(BaseModel):
    user_id: int | None = Field(default=None, gt=0)
    adapter_id: int | None = Field(default=None, gt=0)
    pose_keypoints: list[list[float]]
    face_keypoints: list[list[float]]
    left_hand_keypoints: list[list[float]] | None = None
    right_hand_keypoints: list[list[float]] | None = None


class TranslationResult(BaseModel):
    predicted_text: str
    confidence: float
    latency_ms: float
    used_adapter: bool


class AblationRow(BaseModel):
    config_name: str
    accuracy: float
    calibration_seconds: float | None = None


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
