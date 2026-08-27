"""VisionBridge multimodal temporal base model.

The model consumes pose, face, and both hand skeleton streams. Hands are kept as
separate modalities because hand geometry and motion carry much of the lexical
signal in continuous sign language. The inference loader still freezes the
whole trained backbone; training constructs the model directly so parameters
remain trainable.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

POSE_INPUT_DIM = 33 * 4
FACE_INPUT_DIM = 468 * 3
HAND_INPUT_DIM = 21 * 3
MAX_SEQUENCE_LENGTH = 1024


class StreamEncoder(nn.Module):
    """Project one landmark stream into a shared temporal representation."""

    def __init__(self, input_dim: int, d_model: int = 256, n_layers: int = 1, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, MAX_SEQUENCE_LENGTH, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.1,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("stream input must have shape (batch, frames, features)")
        if x.shape[1] > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"sequence has {x.shape[1]} frames; maximum is {MAX_SEQUENCE_LENGTH}")
        hidden = self.input_proj(x) + self.pos_embedding[:, : x.shape[1]]
        return self.encoder(hidden, src_key_padding_mask=padding_mask)


class MultimodalFusion(nn.Module):
    """Fuse four synchronized modalities per frame with learned gating."""

    def __init__(self, d_model: int = 256, modalities: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * modalities, d_model),
            nn.GELU(),
            nn.Linear(d_model, modalities),
        )
        self.output = nn.Sequential(
            nn.Linear(d_model * modalities, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, streams: tuple[torch.Tensor, ...], padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        joined = torch.cat(streams, dim=-1)
        weights = torch.softmax(self.gate(joined), dim=-1)
        stacked = torch.stack(streams, dim=2)
        weighted = stacked * weights.unsqueeze(-1)
        gated = weighted.reshape(joined.shape)
        fused = self.output(gated)
        if padding_mask is not None:
            fused = fused.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return fused


class TemporalConvBlock(nn.Module):
    """Local temporal motion extractor placed before global attention."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.depthwise(y)
        y = self.pointwise(y).transpose(1, 2)
        return self.norm(residual + self.act(y))


class VisionBridgeBaseModel(nn.Module):
    """Pose + face + left/right hands -> temporal encoder -> character CTC logits."""

    def __init__(
        self,
        pose_input_dim: int = POSE_INPUT_DIM,
        face_input_dim: int = FACE_INPUT_DIM,
        hand_input_dim: int = HAND_INPUT_DIM,
        d_model: int = 256,
        vocab_size: int = 3000,
        shared_layers: int = 4,
        n_heads: int = 4,
        use_hands: bool = True,
    ):
        super().__init__()
        self.use_hands = use_hands
        self.pose_encoder = StreamEncoder(pose_input_dim, d_model, n_layers=1, n_heads=n_heads)
        self.face_encoder = StreamEncoder(face_input_dim, d_model, n_layers=1, n_heads=n_heads)
        self.left_hand_encoder = StreamEncoder(hand_input_dim, d_model, n_layers=1, n_heads=n_heads) if use_hands else None
        self.right_hand_encoder = StreamEncoder(hand_input_dim, d_model, n_layers=1, n_heads=n_heads) if use_hands else None

        modality_count = 4 if use_hands else 2
        self.fusion = MultimodalFusion(d_model=d_model, modalities=modality_count)
        self.temporal_conv = TemporalConvBlock(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
            norm_first=True,
        )
        self.shared_encoder = nn.TransformerEncoder(
            layer,
            num_layers=shared_layers,
            enable_nested_tensor=False,
        )
        self.output_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    @staticmethod
    def _padding_mask(lengths: torch.Tensor, max_frames: int) -> torch.Tensor:
        positions = torch.arange(max_frames, device=lengths.device).unsqueeze(0)
        return positions >= lengths.to(dtype=torch.long).unsqueeze(1)

    def _validate_stream(self, x: torch.Tensor, name: str) -> None:
        if x.ndim != 3:
            raise ValueError(f"{name} must be 3-D: (batch, frames, features)")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite values")

    def forward(
        self,
        pose: torch.Tensor,
        face: torch.Tensor,
        left_hand: torch.Tensor | None = None,
        right_hand: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_stream(pose, "pose")
        self._validate_stream(face, "face")
        if pose.shape[:2] != face.shape[:2]:
            raise ValueError("pose and face must have matching batch/frame dimensions")
        if pose.shape[-1] != POSE_INPUT_DIM:
            raise ValueError(f"pose feature dimension must be {POSE_INPUT_DIM}")
        if face.shape[-1] != FACE_INPUT_DIM:
            raise ValueError(f"face feature dimension must be {FACE_INPUT_DIM}")

        if self.use_hands:
            if left_hand is None or right_hand is None:
                raise ValueError("left_hand and right_hand are required by the hand-aware model")
            self._validate_stream(left_hand, "left_hand")
            self._validate_stream(right_hand, "right_hand")
            if left_hand.shape[:2] != pose.shape[:2] or right_hand.shape[:2] != pose.shape[:2]:
                raise ValueError("all modalities must have matching batch/frame dimensions")
            if left_hand.shape[-1] != HAND_INPUT_DIM or right_hand.shape[-1] != HAND_INPUT_DIM:
                raise ValueError(f"hand feature dimension must be {HAND_INPUT_DIM}")

        if pose.shape[1] > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"sequence has {pose.shape[1]} frames; maximum is {MAX_SEQUENCE_LENGTH}")

        padding_mask = None
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != pose.shape[0]:
                raise ValueError("lengths must have shape (batch,)")
            lengths = lengths.to(device=pose.device, dtype=torch.long)
            if torch.any(lengths <= 0) or torch.any(lengths > pose.shape[1]):
                raise ValueError("lengths must be in the range [1, frames]")
            padding_mask = self._padding_mask(lengths, pose.shape[1])

        pose_emb = self.pose_encoder(pose, padding_mask)
        face_emb = self.face_encoder(face, padding_mask)
        streams: tuple[torch.Tensor, ...]
        if self.use_hands:
            left_emb = self.left_hand_encoder(left_hand, padding_mask)  # type: ignore[arg-type]
            right_emb = self.right_hand_encoder(right_hand, padding_mask)  # type: ignore[arg-type]
            streams = (pose_emb, face_emb, left_emb, right_emb)
        else:
            streams = (pose_emb, face_emb)

        fused = self.fusion(streams, padding_mask)
        fused = self.temporal_conv(fused)
        if padding_mask is not None:
            fused = fused.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden = self.shared_encoder(fused, src_key_padding_mask=padding_mask)
        return self.output_head(hidden)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False


def load_frozen_base_model(weights_path: str | None = None, **kwargs) -> VisionBridgeBaseModel:
    """Load a trained checkpoint and freeze it for inference."""
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Base-model checkpoint was not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        output_head_weight = state.get("output_head.weight")
        if output_head_weight is not None:
            kwargs["vocab_size"] = int(output_head_weight.shape[0])
        # New checkpoints contain hand encoder weights. Legacy checkpoints stay loadable
        # with the old two-stream architecture until a hand-aware model is trained.
        if "left_hand_encoder.input_proj.0.weight" in state:
            kwargs["use_hands"] = True
        elif "left_hand_encoder.input_proj.weight" in state:
            kwargs["use_hands"] = True
        else:
            kwargs["use_hands"] = False
        model = VisionBridgeBaseModel(**kwargs)
        model.load_state_dict(state, strict=True)
    else:
        model = VisionBridgeBaseModel(**kwargs)
    model.freeze()
    model.eval()
    return model
