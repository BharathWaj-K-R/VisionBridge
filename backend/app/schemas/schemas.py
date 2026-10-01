"""Pydantic API contracts for VisionBridge."""
import datetime as dt
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegisterRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=72)


class RegisterResponse(BaseModel):
    message: str
    email: str
    verification_required: bool = True
    resend_after_seconds: int


class VerifyOtpRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class ResendOtpRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)


class LoginRequest(BaseModel):
    identifier: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=8, max_length=72)



class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: str | None = None
    created_at: dt.datetime
    is_verified: bool


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AdapterOut(BaseModel):
    id: int
    owner_id: int
    calibration_seconds: float
    param_count: int | None
    accuracy_gain_pct: float | None
    letters: list[str] = Field(default_factory=list)
    shots: dict[str, int] = Field(default_factory=dict)
    created_at: dt.datetime


class LetterCalibrationSample(BaseModel):
    letter: str = Field(min_length=1, max_length=1, pattern=r"^[A-Za-z]$")
    hand_keypoints: list[float] = Field(min_length=126, max_length=126)


class LetterCalibrationRequest(BaseModel):
    user_id: int = Field(gt=0)
    calibration_seconds: float = Field(default=1, ge=0)
    samples: list[LetterCalibrationSample] = Field(min_length=2, max_length=130)

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
    adapter_id: int | None = Field(default=None, gt=0)
    hand_keypoints: list[float] = Field(min_length=126, max_length=126)


class LetterPredictionResult(BaseModel):
    predicted_letter: str
    confidence: float
    latency_ms: float
    adapter_id: int | None
    mode: Literal["base", "adapter"]


class LetterRecognitionEvent(BaseModel):
    user_id: int = Field(gt=0)
    adapter_id: int | None = Field(default=None, gt=0)
    predicted_letter: str = Field(min_length=1, max_length=1, pattern=r"^(?:[A-Za-z]|\?)$")
    confidence: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)


class CommunicationWordCreate(BaseModel):
    phrase: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=80)


class CommunicationWordOut(BaseModel):
    id: int
    phrase: str
    category: str
    created_at: dt.datetime
    updated_at: dt.datetime


class QuickAccessPayload(BaseModel):
    slots: list[str | None] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def validate_slots(self) -> "QuickAccessPayload":
        for value in self.slots:
            if value is not None and not value.strip():
                raise ValueError("Quick-access phrases must be non-empty or null.")
            if value is not None and len(value.strip()) > 200:
                raise ValueError("Quick-access phrases must be 200 characters or fewer.")
        return self


class AvatarProfile(BaseModel):
    skinTone: str = Field(min_length=4, max_length=20)
    hair: Literal["short", "curly", "long"]
    hairColor: str = Field(min_length=4, max_length=20)
    shirtColor: str = Field(min_length=4, max_length=20)
    bodyShape: Literal["slim", "average", "athletic"]
    apparel: Literal["tee", "vest", "button-down"]
    highContrast: bool = False


class PersonalizationProfileConfig(BaseModel):
    avatar: AvatarProfile
    quickAccess: list[str | None] = Field(min_length=10, max_length=10)
    favorites: list[str] = Field(default_factory=list, max_length=50)
    signingSpeed: Literal[0.5, 0.75, 1] = 1
    ttsVoice: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_profile_config(self) -> "PersonalizationProfileConfig":
        self.favorites = [value.strip() for value in self.favorites if value and value.strip()][:50]
        for index, value in enumerate(self.quickAccess):
            if value is not None:
                self.quickAccess[index] = value.strip()[:200] or None
        return self


class PersonalizationProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    config: PersonalizationProfileConfig


class PersonalizationProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    config: PersonalizationProfileConfig


class PersonalizationProfileOut(BaseModel):
    id: int
    name: str
    config: PersonalizationProfileConfig
    created_at: dt.datetime
    updated_at: dt.datetime


class CommunicationUsageCreate(BaseModel):
    phrase: str = Field(min_length=1, max_length=200)
    profileId: int | None = Field(default=None, gt=0)
