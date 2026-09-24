"""Configurable, replaceable base model for isolated ISL letter recognition."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

INPUT_DIM = 126
HIDDEN_DIM = 128
EMBEDDING_DIM = 64
NUM_CLASSES = 26
LETTER_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MODEL_VERSION = "visionbridge-letter-base-v3"
PREPROCESSING_VERSION = "two-hand-wrist-scale-v1"
LANDMARK_RUNTIME = "mediapipe-hand-landmarker-0.10.35"


class VisionBridgeLetterBaseModel(nn.Module):
    """Scalable MLP whose dimensions and label vocabulary are checkpoint-defined."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        labels: Sequence[str] = LETTER_LABELS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        labels_tuple = tuple(str(label) for label in labels)
        if input_dim <= 0 or hidden_dim <= 0 or embedding_dim <= 0:
            raise ValueError("Model dimensions must be positive")
        if not labels_tuple:
            raise ValueError("Model must contain at least one output label")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.labels = labels_tuple
        self.num_classes = len(labels_tuple)
        self.dropout = float(dropout)

        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(approximate="tanh"),
        )
        self.output_head = nn.Linear(self.embedding_dim, self.num_classes)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, {self.input_dim}] hand features, got {tuple(inputs.shape)}"
            )
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.embed(inputs))


def build_checkpoint(model: VisionBridgeLetterBaseModel) -> dict:
    return {
        "model_version": MODEL_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "landmark_runtime": LANDMARK_RUNTIME,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "embedding_dim": model.embedding_dim,
        "num_classes": model.num_classes,
        "labels": list(model.labels),
        "dropout": model.dropout,
        "state_dict": model.state_dict(),
    }


def save_checkpoint(model: VisionBridgeLetterBaseModel, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint(model), target)


def load_checkpoint(path: str | Path) -> VisionBridgeLetterBaseModel:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Letter base-model checkpoint is missing: {target}")

    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except (OSError, EOFError, pickle.UnpicklingError, UnicodeError) as exc:
        raise ValueError("Letter base-model checkpoint could not be decoded") from exc

    if not isinstance(payload, dict):
        raise ValueError("Letter base-model checkpoint must be a dictionary")
    if payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported letter base-model version")
    if payload.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Unsupported letter base-model preprocessing version")
    if payload.get("landmark_runtime") != LANDMARK_RUNTIME:
        raise ValueError("Unsupported letter base-model landmark runtime")

    try:
        labels = tuple(payload.get("labels", ()))
        input_dim = int(payload.get("input_dim", 0))
        hidden_dim = int(payload.get("hidden_dim", 0))
        embedding_dim = int(payload.get("embedding_dim", 0))
        num_classes = int(payload.get("num_classes", 0))
        dropout = float(payload.get("dropout", 0.10))
    except (TypeError, ValueError) as exc:
        raise ValueError("Letter base-model architecture metadata is invalid") from exc

    if not labels or len(labels) != num_classes:
        raise ValueError("Letter base-model label metadata is incompatible")

    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Letter base-model state_dict is missing")

    model = VisionBridgeLetterBaseModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        labels=labels,
        dropout=dropout,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def build_browser_payload(model: VisionBridgeLetterBaseModel, model_sha256: str) -> dict:
    """Return only inference weights needed by the browser fast path."""
    state = model.state_dict()

    def array(name: str) -> list:
        return state[name].detach().cpu().tolist()

    return {
        "model_version": MODEL_VERSION,
        "model_sha256": model_sha256,
        "preprocessing_version": PREPROCESSING_VERSION,
        "landmark_runtime": LANDMARK_RUNTIME,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "embedding_dim": model.embedding_dim,
        "num_classes": model.num_classes,
        "labels": list(model.labels),
        "layers": {
            "input_norm": {"weight": array("encoder.0.weight"), "bias": array("encoder.0.bias")},
            "hidden": {"weight": array("encoder.1.weight"), "bias": array("encoder.1.bias")},
            "embedding": {"weight": array("encoder.4.weight"), "bias": array("encoder.4.bias")},
            "embedding_norm": {"weight": array("encoder.5.weight"), "bias": array("encoder.5.bias")},
            "head": {"weight": array("output_head.weight"), "bias": array("output_head.bias")},
        },
    }


def checkpoint_status(path: str | Path) -> dict[str, str | bool]:
    try:
        model = load_checkpoint(path)
        return {
            "available": True,
            "status": "ready",
            "modality": "hand-only letter base + dynamic few-shot adapter",
            "model_version": MODEL_VERSION,
            "input_dim": str(model.input_dim),
            "embedding_dim": str(model.embedding_dim),
            "num_classes": str(model.num_classes),
        }
    except FileNotFoundError:
        return {
            "available": False,
            "status": "letter_base_model_missing",
            "modality": "hand-only letter base + dynamic few-shot adapter",
        }
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError, UnicodeError):
        return {
            "available": False,
            "status": "letter_base_model_invalid",
            "modality": "hand-only letter base + dynamic few-shot adapter",
        }
