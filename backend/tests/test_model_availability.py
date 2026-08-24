"""Model-readiness behavior must be explicit rather than serving random logits."""

import pytest

from app.services import inference_service


def test_missing_checkpoint_is_reported_as_unavailable(monkeypatch, tmp_path):
    previous_model = inference_service._base_model
    previous_vocab = inference_service._id_to_token
    monkeypatch.setattr(inference_service.settings, "BASE_MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setattr(inference_service, "_base_model", None)
    monkeypatch.setattr(inference_service, "_id_to_token", {})

    try:
        status = inference_service.model_status()
        assert status == {"available": False, "status": "unavailable"}
        with pytest.raises(inference_service.ModelUnavailableError):
            inference_service.get_base_model()
    finally:
        # The module keeps a process-wide model cache; leave it exactly as it
        # was so this focused test cannot affect later integration tests.
        inference_service._base_model = previous_model
        inference_service._id_to_token = previous_vocab
