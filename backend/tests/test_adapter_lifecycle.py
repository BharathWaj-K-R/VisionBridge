from pathlib import Path

import pytest
import torch

from app.models.bridge_adapter import BridgeAdapterStack
from app.services import calibration_service


def test_adapter_parameter_budget_matches_current_backbone():
    base = torch.nn.Sequential(torch.nn.Linear(256, 256))
    adapter = BridgeAdapterStack(d_model=256, n_layers=4, bottleneck_dim=16)
    assert adapter.param_budget_ok(sum(p.numel() for p in base.parameters()), budget_pct=100.0)


def test_delete_adapter_weights_removes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_service.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path))
    adapter_path = Path(tmp_path) / "adapter.pt"
    adapter_path.write_bytes(b"test")

    calibration_service.delete_adapter_weights(str(adapter_path))
    assert not adapter_path.exists()


def test_delete_adapter_weights_rejects_path_outside_store(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_service.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path / "adapters"))
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"do-not-delete")

    with pytest.raises(ValueError, match="outside the adapter weight directory"):
        calibration_service.delete_adapter_weights(str(outside))

    assert outside.exists()
