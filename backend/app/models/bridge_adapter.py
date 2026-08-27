"""Few-shot signer personalization adapters for VisionBridge."""
from __future__ import annotations

import time

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Small residual adapter trained while the base model remains frozen."""

    def __init__(self, d_model: int = 256, bottleneck_dim: int = 16):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up_proj(self.activation(self.down_proj(x)))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BridgeAdapterStack(nn.Module):
    """One lightweight adapter after each shared temporal Transformer layer."""

    def __init__(self, d_model: int = 256, n_layers: int = 4, bottleneck_dim: int = 16):
        super().__init__()
        self.adapters = nn.ModuleList(
            [BottleneckAdapter(d_model, bottleneck_dim) for _ in range(n_layers)]
        )

    def total_param_count(self) -> int:
        return sum(adapter.param_count() for adapter in self.adapters)

    def param_budget_ok(self, base_model_param_count: int, budget_pct: float = 2.0) -> bool:
        return 100.0 * self.total_param_count() / max(base_model_param_count, 1) <= budget_pct

    def forward_with_base(
        self,
        base_model,
        pose: torch.Tensor,
        face: torch.Tensor,
        left_hand: torch.Tensor | None = None,
        right_hand: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reuse the base model's modality encoders and splice adapters into its temporal stack."""
        padding_mask = None
        if pose.shape[1] != face.shape[1]:
            raise ValueError("pose/face frame counts must match for adapter inference")
        if left_hand is None:
            left_hand = torch.zeros(
                pose.shape[0], pose.shape[1], 63, dtype=pose.dtype, device=pose.device
            )
        if right_hand is None:
            right_hand = torch.zeros_like(left_hand)

        # The base model performs all shape and modality validation here.
        pose_emb = base_model.pose_encoder(
            pose,
            padding_mask,
        )
        face_emb = base_model.face_encoder(
            face,
            padding_mask,
        )
        if not getattr(base_model, "use_hands", False):
            streams = (pose_emb, face_emb)
        else:
            left_emb = base_model.left_hand_encoder(left_hand, padding_mask)
            right_emb = base_model.right_hand_encoder(right_hand, padding_mask)
            streams = (pose_emb, face_emb, left_emb, right_emb)

        hidden = base_model.fusion(streams, padding_mask)
        hidden = base_model.temporal_conv(hidden)
        for layer, adapter in zip(base_model.shared_encoder.layers, self.adapters):
            hidden = layer(hidden)
            hidden = adapter(hidden)
        return base_model.output_head(hidden)

    def calibrate(
        self,
        base_model,
        calibration_pose: torch.Tensor,
        calibration_face: torch.Tensor,
        calibration_left_hand: torch.Tensor,
        calibration_right_hand: torch.Tensor,
        target_labels: torch.Tensor,
        target_lengths: torch.Tensor,
        epochs: int = 20,
        lr: float = 1e-3,
        blank_id: int = 0,
    ) -> dict:
        start = time.time()
        for parameter in base_model.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        input_lengths = torch.full(
            (calibration_pose.shape[0],),
            calibration_pose.shape[1],
            dtype=torch.long,
            device=calibration_pose.device,
        )

        self.train()
        loss = torch.tensor(0.0, device=calibration_pose.device)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = self.forward_with_base(
                base_model,
                calibration_pose,
                calibration_face,
                calibration_left_hand,
                calibration_right_hand,
            )
            log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = ctc_loss(log_probs, target_labels, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("Adapter calibration produced a non-finite CTC loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
        self.eval()
        return {
            "calibration_wall_seconds": time.time() - start,
            "final_loss": float(loss.item()),
            "param_count": self.total_param_count(),
        }
