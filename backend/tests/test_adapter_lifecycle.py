from pathlib import Path

import pytest

from app.models.base_model import VisionBridgeBaseModel
from app.models.bridge_adapter import BridgeAdapterStack
from app.services import calibration_service


def _shared_layer_count(base: VisionBridgeBaseModel) -> int:
    encoder = base.shared_encoder
    if hasattr(encoder, "num_layers"):
        return int(encoder.num_layers)
    return len(encoder.layers)


def test_adapter_parameter_budget_matches_current_backbone():
    base = VisionBridgeBaseModel(vocab_size=49)
    adapter = BridgeAdapterStack(
        d_model=base.d_model,
        n_layers=_shared_layer_count(base),
        bottleneck_dim=16,
    )
    base_params = sum(p.numel() for p in base.parameters())
    assert adapter.total_param_count() < base_params * 0.02
    assert adapter.param_budget_ok(base_params)


def test_stage_and_finalize_adapter_weight_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_service.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path))
    adapter_path = Path(tmp_path) / "adapter.pt"
    adapter_path.write_bytes(b"test")

    tombstone = calibration_service.stage_adapter_weight_delete(str(adapter_path))
    assert not adapter_path.exists()
    assert tombstone is not None and tombstone.exists()

    calibration_service.finalize_staged_adapter_weight_delete(tombstone)
    assert not tombstone.exists()


def test_stage_adapter_weight_delete_can_be_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_service.settings, "ADAPTER_WEIGHTS_DIR", str(tmp_path))
    adapter_path = Path(tmp_path) / "adapter.pt"
    adapter_path.write_bytes(b"test")

    tombstone = calibration_service.stage_adapter_weight_delete(str(adapter_path))
    assert tombstone is not None
    calibration_service.restore_staged_adapter_weight(tombstone, str(adapter_path))

    assert adapter_path.read_bytes() == b"test"
    assert not tombstone.exists()


def test_stage_adapter_weight_delete_rejects_path_outside_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        calibration_service.settings,
        "ADAPTER_WEIGHTS_DIR",
        str(tmp_path / "adapters"),
    )
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"do-not-delete")

    with pytest.raises(ValueError, match="outside the adapter weight directory"):
        calibration_service.stage_adapter_weight_delete(str(outside))

    assert outside.exists()
