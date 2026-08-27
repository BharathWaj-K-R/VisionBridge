"""Calibration orchestration for signer-specific BridgeAdapters."""
from __future__ import annotations

import os
import pickle
import uuid
from pathlib import Path

import torch

from app.core.config import get_settings
from app.models.bridge_adapter import BridgeAdapterStack
from app.services.inference_service import get_base_model

settings = get_settings()


def calibrate_new_adapter(
    pose: torch.Tensor,
    face: torch.Tensor,
    left_hand: torch.Tensor,
    right_hand: torch.Tensor,
    target_labels: torch.Tensor,
    target_lengths: torch.Tensor,
    calibration_seconds: float,
) -> dict:
    """Train a fresh signer adapter against the multimodal base model."""
    base_model = get_base_model()
    n_layers = len(base_model.shared_encoder.layers)
    adapter = BridgeAdapterStack(d_model=base_model.d_model, n_layers=n_layers)

    base_param_count = sum(p.numel() for p in base_model.parameters())
    if not adapter.param_budget_ok(base_param_count):
        raise RuntimeError(
            "BridgeAdapter parameter budget exceeded: "
            f"adapter_params={adapter.total_param_count()}, base_params={base_param_count}"
        )

    stats = adapter.calibrate(
        base_model,
        pose,
        face,
        left_hand,
        right_hand,
        target_labels,
        target_lengths,
    )

    os.makedirs(settings.ADAPTER_WEIGHTS_DIR, exist_ok=True)
    weights_path = os.path.join(
        settings.ADAPTER_WEIGHTS_DIR,
        f"{uuid.uuid4().hex}.pt",
    )
    torch.save(adapter.state_dict(), weights_path)

    return {
        "weights_path": weights_path,
        "calibration_seconds": calibration_seconds,
        "param_count": stats["param_count"],
        "param_budget_ok": True,
        "final_loss": stats["final_loss"],
    }


def load_adapter_for_signer(weights_path: str, d_model: int, n_layers: int) -> BridgeAdapterStack:
    adapter_root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    candidate = Path(weights_path).resolve()
    if adapter_root not in candidate.parents or not candidate.is_file():
        raise FileNotFoundError("Adapter weights are unavailable")

    adapter = BridgeAdapterStack(d_model=d_model, n_layers=n_layers)
    try:
        state = torch.load(candidate, map_location="cpu", weights_only=True)
        adapter.load_state_dict(state)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise RuntimeError("Adapter weights could not be loaded") from exc
    adapter.eval()
    return adapter


def stage_adapter_weight_delete(weights_path: str) -> Path | None:
    adapter_root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    candidate = Path(weights_path).resolve()
    if adapter_root not in candidate.parents:
        raise ValueError("Refusing to delete a path outside the adapter weight directory")
    if not candidate.exists():
        return None
    tombstone = candidate.with_name(f".{candidate.name}.{uuid.uuid4().hex}.deleting")
    candidate.replace(tombstone)
    return tombstone


def restore_staged_adapter_weight(tombstone: Path, original_path: str) -> None:
    adapter_root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    candidate = Path(original_path).resolve()
    tombstone = tombstone.resolve()
    if adapter_root not in candidate.parents or adapter_root not in tombstone.parents:
        raise ValueError("Refusing to restore adapter weights outside the configured directory")
    if tombstone.exists():
        tombstone.replace(candidate)


def finalize_staged_adapter_weight_delete(tombstone: Path | None) -> None:
    if tombstone is not None and tombstone.exists():
        tombstone.unlink()
