"""Database models for users, signer adapters, and prediction history."""
import datetime as dt

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.session import Base


class User(Base):
    """A VisionBridge account used to store signer-specific adapters and history."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(
        DateTime,
        default=lambda: dt.datetime.now(dt.timezone.utc),
    )

    adapters = relationship("SignerAdapter", back_populates="owner")


class SignerAdapter(Base):
    """Metadata for one signer's calibrated letter prototypes."""

    __tablename__ = "signer_adapters"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    weights_path = Column(String, nullable=False)
    payload_json = Column(Text, nullable=True)
    calibration_seconds = Column(Float, nullable=False)
    param_count = Column(Integer, nullable=True)
    accuracy_gain_pct = Column(Float, nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: dt.datetime.now(dt.timezone.utc),
    )

    owner = relationship("User", back_populates="adapters")


class TranslationLog(Base):
    """A persisted letter prediction used by dashboard and history views."""

    __tablename__ = "translation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    adapter_id = Column(Integer, ForeignKey("signer_adapters.id"), nullable=True)
    predicted_text = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    used_adapter = Column(Integer, default=0)
    created_at = Column(
        DateTime,
        default=lambda: dt.datetime.now(dt.timezone.utc),
    )


class CommunicationWord(Base):
    """A user-created word or phrase for the daily communication word bank."""

    __tablename__ = "communication_words"
    __table_args__ = (UniqueConstraint("user_id", "phrase", name="uq_communication_word_user_phrase"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    phrase = Column(String(200), nullable=False)
    category = Column(String(80), nullable=False, default="Custom")
    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: dt.datetime.now(dt.timezone.utc),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc),
    )


class QuickAccessSlot(Base):
    """One of exactly ten persistent quick-access slots for a user."""

    __tablename__ = "quick_access_slots"
    __table_args__ = (UniqueConstraint("user_id", "slot", name="uq_quick_access_user_slot"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    slot = Column(Integer, nullable=False)
    phrase = Column(String(200), nullable=True)
    updated_at = Column(
        DateTime,
        default=lambda: dt.datetime.now(dt.timezone.utc),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc),
    )


class PersonalizationProfile(Base):
    """A named, user-scoped daily communication profile."""

    __tablename__ = "personalization_profiles"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_personalization_profile_user_name"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(80), nullable=False)
    config_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: dt.datetime.now(dt.timezone.utc),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc),
    )


class CommunicationUsage(Base):
    """Per-user phrase usage counters for the Most Used section."""

    __tablename__ = "communication_usage"
    __table_args__ = (UniqueConstraint("user_id", "profile_id", "phrase", name="uq_communication_usage_user_profile_phrase"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    profile_id = Column(Integer, ForeignKey("personalization_profiles.id"), nullable=True, index=True)
    phrase = Column(String(200), nullable=False)
    usage_count = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))
