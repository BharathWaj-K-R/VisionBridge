"""
Pydantic request/response models. Kept separate from ORM models so the API
contract can evolve independently of the DB schema.
"""
import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


# ---------- Users / auth ----------

class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    # bcrypt only processes the first 72 bytes; reject longer passwords rather
    # than silently weakening the effective credential.
    password: str = Field(min_length=8, max_length=72)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    created_at: dt.datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- Calibration / adapter ----------

class CalibrationStartRequest(BaseModel):
    user_id: int


class CalibrationRequest(BaseModel):
    user_id: int = Field(gt=0)
    calibration_seconds: float = Field(gt=0)
    pose_keypoints: list[list[float]]  # (frames, feature_dim) — one clip
    face_keypoints: list[list[float]]  # (frames, feature_dim) — one clip
    target_labels: list[int] = Field(min_length=1)  # sentence-level token ids for this clip, no blank token


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


# ---------- Translation ----------

class TranslationRequest(BaseModel):
    user_id: int | None = Field(default=None, gt=0)
    adapter_id: int | None = Field(default=None, gt=0)  # if None, base model only
    pose_keypoints: list[list[float]]  # (frames, feature_dim)
    face_keypoints: list[list[float]]  # (frames, feature_dim)


class TranslationResult(BaseModel):
    predicted_text: str
    confidence: float
    latency_ms: float
    used_adapter: bool


# ---------- Ablation / eval ----------

class AblationRow(BaseModel):
    config_name: str  # e.g. "base_only", "base+face", "base+adapter", "base+face+adapter"
    accuracy: float
    calibration_seconds: float | None = None
