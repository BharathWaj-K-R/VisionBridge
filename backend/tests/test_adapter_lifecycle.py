from pathlib import Path

import pytest

from app.models.base_model import VisionBridgeBaseModel
from app.models.bridge_adapter import BridgeAdapterStack
from app.services import calibration_service


def test_adapter_parameter_budget_matches_current_backbone():
    base = VisionBridgeBaseModel(vocab_size=49)
    adapter = BridgeAdapterStack(
        d_model=base.d_model,
        n_layers=len(base.shared_encoder.layers),
        bottleneck_dim=16,
    )
    base_params = sum(p.numel() for p in base.parameters())
    assert adapter.total_param_count() < base_params * 0.02
    assert adapter.param_budget_ok(base_params)


def test_delete_adapter_weights_removes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_service.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path))
    adapter_path = Path(tmp_path) / "adapter.pt"
    adapter_path.write_bytes(b"test")

    calibration_service.delete_adapter_weights(str(adapter_path))
    assert not adapter_path.exists()


def test_delete_adapter_weights_rejects_path_outside_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        calibration_service.settings,
        "ADAPTER_WEIGHTS_DIR",
        str(tmp_path / "adapters"),
    )
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"do-not-delete")

    with pytest.raises(ValueError, match="outside the adapter weight directory"):
        calibration_service.delete_adapter_weights(str(outside))

    assert outside.exists()
