"""Database models for users, signer adapters, and prediction history."""
import datetime as dt

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.session import Base


class User(Base):
    """A VisionBridge account used to store signer-specific adapters and history."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
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
