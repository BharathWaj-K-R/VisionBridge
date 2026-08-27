import pytest
import torch

from app.models.base_model import (
    FACE_INPUT_DIM,
    HAND_INPUT_DIM,
    POSE_INPUT_DIM,
    VisionBridgeBaseModel,
)


def test_hand_aware_forward_shape():
    model = VisionBridgeBaseModel(vocab_size=49, use_hands=True).eval()
    pose = torch.randn(1, 8, POSE_INPUT_DIM)
    face = torch.randn(1, 8, FACE_INPUT_DIM)
    left = torch.randn(1, 8, HAND_INPUT_DIM)
    right = torch.randn(1, 8, HAND_INPUT_DIM)
    with torch.inference_mode():
        logits = model(pose, face, left, right, torch.tensor([8]))
    assert logits.shape == (1, 8, 49)
    assert torch.isfinite(logits).all()


def test_hand_aware_forward_requires_both_hands():
    model = VisionBridgeBaseModel(vocab_size=49, use_hands=True).eval()
    pose = torch.zeros(1, 8, POSE_INPUT_DIM)
    face = torch.zeros(1, 8, FACE_INPUT_DIM)
    with pytest.raises(ValueError, match="hand-aware model requires"):
        model(pose, face, lengths=torch.tensor([8]))
