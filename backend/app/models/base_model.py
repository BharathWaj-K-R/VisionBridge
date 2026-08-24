"""
Base pose + facial-expression fusion transformer.

This is the FROZEN backbone: once pretrained on ISLTranslate / ISL-CSLTR /
iSign / INCLUDE, its weights never change at inference time. Signer-specific
personalization happens entirely in BridgeAdapter (see bridge_adapter.py),
which is the actual novel contribution of this project.
"""
import torch
import torch.nn as nn

POSE_INPUT_DIM = 33 * 4
FACE_INPUT_DIM = 468 * 3
MAX_SEQUENCE_LENGTH = 1024


class StreamEncoder(nn.Module):
    """Encode one keypoint stream while ignoring padded frames when lengths are supplied."""

    def __init__(self, input_dim: int, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, MAX_SEQUENCE_LENGTH, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        h = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        return self.encoder(h, src_key_padding_mask=padding_mask)


class CrossModalFusion(nn.Module):
    """Fuse pose and face while preventing attention to padded face frames."""

    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        pose_emb: torch.Tensor,
        face_emb: torch.Tensor,
        face_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused, _ = self.cross_attn(
            query=pose_emb,
            key=face_emb,
            value=face_emb,
            key_padding_mask=face_padding_mask,
        )
        return self.norm(pose_emb + fused)


class VisionBridgeBaseModel(nn.Module):
    """Pose + face -> fusion -> shared Transformer -> character logits.

    ``lengths`` is optional for backward compatibility. During batched training
    it is required to prevent padded frames introduced by ``collate_ctc_batch``
    from influencing real frames through self/cross attention. Single-clip
    inference can omit it because no padding is present.
    """

    def __init__(
        self,
        pose_input_dim: int = POSE_INPUT_DIM,
        face_input_dim: int = FACE_INPUT_DIM,
        d_model: int = 256,
        vocab_size: int = 3000,
        shared_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.pose_encoder = StreamEncoder(pose_input_dim, d_model, n_layers=2, n_heads=n_heads)
        self.face_encoder = StreamEncoder(face_input_dim, d_model, n_layers=2, n_heads=n_heads)
        self.fusion = CrossModalFusion(d_model, n_heads)

        shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.shared_encoder = nn.TransformerEncoder(shared_layer, num_layers=shared_layers)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def _padding_mask(lengths: torch.Tensor, max_frames: int) -> torch.Tensor:
        lengths = lengths.to(dtype=torch.long)
        positions = torch.arange(max_frames, device=lengths.device).unsqueeze(0)
        return positions >= lengths.unsqueeze(1)

    def forward(
        self,
        pose: torch.Tensor,
        face: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pose.ndim != 3 or face.ndim != 3:
            raise ValueError("pose and face must be 3-D tensors: (batch, frames, features)")
        if pose.shape[:2] != face.shape[:2]:
            raise ValueError("pose and face must have matching batch/frame dimensions")

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
        fused = self.fusion(pose_emb, face_emb, padding_mask)
        hidden = self.shared_encoder(fused, src_key_padding_mask=padding_mask)
        return self.output_head(hidden)


def load_frozen_base_model(
    weights_path: str | None = None,
    **kwargs,
) -> VisionBridgeBaseModel:
    """Instantiate and freeze the base model."""
    if weights_path:
        import os

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            output_head_weight = state.get("output_head.weight")
            if output_head_weight is not None:
                kwargs["vocab_size"] = int(output_head_weight.shape[0])
            model = VisionBridgeBaseModel(**kwargs)
            model.load_state_dict(state)
        else:
            model = VisionBridgeBaseModel(**kwargs)
    else:
        model = VisionBridgeBaseModel(**kwargs)

    model.freeze()
    model.eval()
    return model
