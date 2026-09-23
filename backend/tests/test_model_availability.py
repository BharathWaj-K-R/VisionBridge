"""Model readiness tests for the active ISL letter pipeline."""
import pytest

from app.services import letter_fewshot


def test_missing_letter_checkpoint_is_reported_as_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        letter_fewshot.settings,
        "LETTER_BASE_MODEL_PATH",
        str(tmp_path / "missing.pt"),
    )

    status = letter_fewshot.letter_model_status()

    assert status["available"] is False
    assert status["status"] == "letter_base_model_missing"
    assert "letter base" in status["modality"]


def test_invalid_letter_checkpoint_is_reported_as_unavailable(monkeypatch, tmp_path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not-a-checkpoint")
    monkeypatch.setattr(letter_fewshot.settings, "LETTER_BASE_MODEL_PATH", str(path))

    status = letter_fewshot.letter_model_status()

    assert status["available"] is False
    assert status["status"] == "letter_base_model_invalid"
