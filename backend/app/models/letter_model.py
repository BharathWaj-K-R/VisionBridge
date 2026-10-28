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

    if input_dim != INPUT_DIM:
        raise ValueError("Letter base-model input dimension is incompatible")
    if labels != LETTER_LABELS or num_classes != NUM_CLASSES:
        raise ValueError("Letter base-model label vocabulary is incompatible")

    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Letter base-model state_dict is missing")
    if any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("Letter base-model state_dict contains invalid values")

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




V4_MODEL_VERSION = "visionbridge-letter-base-v4"
V4_PREPROCESSING_VERSION = "two-hand-geometry-v2"
V4_VARIANTS = (
    "full_v4",
    "v4_features_old_mlp",
    "v1_features_shared_encoder",
)
V4_VARIANT_INPUT_DIMS = {
    "full_v4": 141,
    "v4_features_old_mlp": 141,
    "v1_features_shared_encoder": INPUT_DIM,
}
V4_VARIANT_PREPROCESSING = {
    "full_v4": V4_PREPROCESSING_VERSION,
    "v4_features_old_mlp": V4_PREPROCESSING_VERSION,
    "v1_features_shared_encoder": PREPROCESSING_VERSION,
}
V4_FEATURE_LAYOUTS = {
    "full_v4": "two-hand-geometry-v2:141:left64,right64,relative13",
    "v4_features_old_mlp": "two-hand-geometry-v2:141:left64,right64,relative13",
    "v1_features_shared_encoder": "two-hand-wrist-scale-v1:126:left63,right63",
}


class VisionBridgeLetterBaseV4Model(nn.Module):
    """Versioned V4 architecture and feature ablations; never replaces the V3 MLP."""

    def __init__(
        self,
        variant: str = "full_v4",
        hidden_dim: int = HIDDEN_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        labels: Sequence[str] = LETTER_LABELS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if variant not in V4_VARIANTS:
            raise ValueError(f"Unsupported V4 architecture variant: {variant}")
        labels_tuple = tuple(str(label) for label in labels)
        if hidden_dim <= 0 or embedding_dim != EMBEDDING_DIM:
            raise ValueError("V4 requires a positive hidden dimension and 64D embeddings")
        if labels_tuple != LETTER_LABELS:
            raise ValueError("V4 requires the fixed A-Z label vocabulary")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")

        self.model_version = V4_MODEL_VERSION
        self.preprocessing_version = V4_VARIANT_PREPROCESSING[variant]
        self.landmark_runtime = LANDMARK_RUNTIME
        self.architecture_variant = variant
        self.feature_layout = V4_FEATURE_LAYOUTS[variant]
        self.input_dim = V4_VARIANT_INPUT_DIMS[variant]
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = EMBEDDING_DIM
        self.num_classes = len(labels_tuple)
        self.labels = labels_tuple
        self.dropout = float(dropout)
        self.hand_embedding_dim = 32
        self.geometry_dim = 13 if variant != "v1_features_shared_encoder" else 0
        self.fusion_input_dim = 2 * self.hand_embedding_dim + 13

        if variant == "v4_features_old_mlp":
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
            self.hand_embedding_dim = 0
            self.geometry_dim = 0
            self.fusion_input_dim = 0
            return

        self.hand_encoder = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 48),
            nn.GELU(approximate="tanh"),
            nn.Dropout(self.dropout),
            nn.Linear(48, self.hand_embedding_dim),
            nn.LayerNorm(self.hand_embedding_dim),
            nn.GELU(approximate="tanh"),
        )
        self.left_role_embedding = nn.Parameter(torch.zeros(self.hand_embedding_dim))
        self.right_role_embedding = nn.Parameter(torch.zeros(self.hand_embedding_dim))
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.fusion_input_dim),
            nn.Linear(self.fusion_input_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(approximate="tanh"),
        )
        self.output_head = nn.Linear(self.embedding_dim, self.num_classes)

    def _hand_slots(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.architecture_variant == "v1_features_shared_encoder":
            left = inputs[:, :63]
            right = inputs[:, 63:126]
            left_presence = (left.abs().amax(dim=1, keepdim=True) > 1e-8).to(inputs.dtype)
            right_presence = (right.abs().amax(dim=1, keepdim=True) > 1e-8).to(inputs.dtype)
            left_slot = torch.cat((left, left_presence), dim=1)
            right_slot = torch.cat((right, right_presence), dim=1)
            geometry = inputs.new_zeros((inputs.shape[0], 13))
            return left_slot, right_slot, geometry

        left_slot = inputs[:, :64]
        right_slot = inputs[:, 64:128]
        geometry = inputs[:, 128:141]
        return left_slot, right_slot, geometry

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, {self.input_dim}] V4 features, got {tuple(inputs.shape)}"
            )
        if self.architecture_variant == "v4_features_old_mlp":
            return self.encoder(inputs)

        left_slot, right_slot, geometry = self._hand_slots(inputs)
        left_presence = left_slot[:, 63:64]
        right_presence = right_slot[:, 63:64]
        left_encoded = (
            self.hand_encoder(left_slot) + self.left_role_embedding.unsqueeze(0)
        ) * left_presence
        right_encoded = (
            self.hand_encoder(right_slot) + self.right_role_embedding.unsqueeze(0)
        ) * right_presence
        fused = torch.cat((left_encoded, right_encoded, geometry), dim=1)
        return self.fusion(fused)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.embed(inputs))


def build_v4_checkpoint(model: VisionBridgeLetterBaseV4Model) -> dict:
    """Serialize V4 with explicit variant and preprocessing metadata."""
    return {
        "model_version": V4_MODEL_VERSION,
        "architecture_variant": model.architecture_variant,
        "feature_layout": model.feature_layout,
        "preprocessing_version": model.preprocessing_version,
        "landmark_runtime": model.landmark_runtime,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "hand_embedding_dim": model.hand_embedding_dim,
        "fusion_input_dim": model.fusion_input_dim,
        "embedding_dim": model.embedding_dim,
        "num_classes": model.num_classes,
        "labels": list(model.labels),
        "dropout": model.dropout,
        "state_dict": model.state_dict(),
    }


def save_v4_checkpoint(model: VisionBridgeLetterBaseV4Model, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_v4_checkpoint(model), target)


def load_v4_checkpoint(
    path: str | Path,
    expected_variant: str | None = None,
) -> VisionBridgeLetterBaseV4Model:
    """Load only a metadata-compatible V4 checkpoint; the V3 loader stays V3-only."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"V4 letter base-model checkpoint is missing: {target}")
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except (OSError, EOFError, pickle.UnpicklingError, UnicodeError) as exc:
        raise ValueError("V4 letter checkpoint could not be decoded") from exc
    if not isinstance(payload, dict):
        raise ValueError("V4 letter checkpoint must be a dictionary")
    if payload.get("model_version") != V4_MODEL_VERSION:
        raise ValueError("Unsupported V4 letter base-model version")

    variant = payload.get("architecture_variant")
    if variant not in V4_VARIANTS:
        raise ValueError("V4 architecture variant metadata is invalid")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError("V4 checkpoint architecture variant does not match the requested variant")
    if payload.get("preprocessing_version") != V4_VARIANT_PREPROCESSING[variant]:
        raise ValueError("V4 checkpoint preprocessing metadata does not match its architecture variant")
    if payload.get("feature_layout") != V4_FEATURE_LAYOUTS[variant]:
        raise ValueError("V4 checkpoint feature layout metadata is incompatible")
    if payload.get("landmark_runtime") != LANDMARK_RUNTIME:
        raise ValueError("V4 checkpoint landmark runtime is incompatible")

    try:
        input_dim = int(payload.get("input_dim", 0))
        hidden_dim = int(payload.get("hidden_dim", 0))
        embedding_dim = int(payload.get("embedding_dim", 0))
        num_classes = int(payload.get("num_classes", 0))
        dropout = float(payload.get("dropout", 0.10))
    except (TypeError, ValueError) as exc:
        raise ValueError("V4 checkpoint architecture metadata is invalid") from exc

    if input_dim != V4_VARIANT_INPUT_DIMS[variant]:
        raise ValueError("V4 checkpoint input dimension is incompatible with its feature variant")
    if embedding_dim != EMBEDDING_DIM or num_classes != NUM_CLASSES:
        raise ValueError("V4 checkpoint output dimensions are incompatible")
    if tuple(payload.get("labels", ())) != LETTER_LABELS:
        raise ValueError("V4 checkpoint label vocabulary is incompatible")

    state = payload.get("state_dict")
    if not isinstance(state, dict) or any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("V4 checkpoint state_dict is missing or contains invalid values")

    model = VisionBridgeLetterBaseV4Model(
        variant=variant,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        labels=tuple(payload["labels"]),
        dropout=dropout,
    )
    if int(payload.get("hand_embedding_dim", model.hand_embedding_dim)) != model.hand_embedding_dim:
        raise ValueError("V4 checkpoint shared-hand embedding dimension is incompatible")
    if int(payload.get("fusion_input_dim", model.fusion_input_dim)) != model.fusion_input_dim:
        raise ValueError("V4 checkpoint fusion input dimension is incompatible")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_model_checkpoint(
    path: str | Path,
) -> VisionBridgeLetterBaseModel | VisionBridgeLetterBaseV4Model:
    """Dispatch by model version while retaining strict, separate V3/V4 loaders."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Letter base-model checkpoint is missing: {target}")
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except (OSError, EOFError, pickle.UnpicklingError, UnicodeError) as exc:
        raise ValueError("Letter checkpoint could not be decoded") from exc
    if not isinstance(payload, dict):
        raise ValueError("Letter checkpoint must be a dictionary")
    version = payload.get("model_version")
    if version == MODEL_VERSION:
        return load_checkpoint(target)
    if version == V4_MODEL_VERSION:
        return load_v4_checkpoint(target)
    raise ValueError("Unsupported letter base-model version")


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
