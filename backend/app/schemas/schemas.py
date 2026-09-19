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
