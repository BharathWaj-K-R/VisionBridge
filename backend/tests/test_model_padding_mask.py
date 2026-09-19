import torch

from app.models.base_model import FACE_INPUT_DIM, HAND_INPUT_DIM, POSE_INPUT_DIM, VisionBridgeBaseModel


def _streams(batch: int, frames: int):
    return (
        torch.randn(batch, frames, POSE_INPUT_DIM),
        torch.randn(batch, frames, FACE_INPUT_DIM),
        torch.randn(batch, frames, HAND_INPUT_DIM),
        torch.randn(batch, frames, HAND_INPUT_DIM),
    )


def test_padding_mask_blocks_padded_frames_from_valid_outputs():
    torch.manual_seed(7)
    model = VisionBridgeBaseModel(vocab_size=49, use_hands=True).eval()
    batch, frames, valid = 2, 8, 5
    pose, face, left, right = _streams(batch, frames)
    lengths = torch.tensor([valid, frames], dtype=torch.long)

    with torch.inference_mode():
        baseline = model(pose, face, left, right, lengths)

    modified_pose = pose.clone()
    modified_face = face.clone()
    modified_left = left.clone()
    modified_right = right.clone()
    modified_pose[0, valid:] = torch.randn_like(modified_pose[0, valid:]) * 1000.0
    modified_face[0, valid:] = torch.randn_like(modified_face[0, valid:]) * 1000.0
    modified_left[0, valid:] = torch.randn_like(modified_left[0, valid:]) * 1000.0
    modified_right[0, valid:] = torch.randn_like(modified_right[0, valid:]) * 1000.0

    with torch.inference_mode():
        masked = model(modified_pose, modified_face, modified_left, modified_right, lengths)

    assert torch.isfinite(masked).all()
    assert torch.allclose(baseline[0, :valid], masked[0, :valid], atol=1e-5, rtol=1e-5)


def test_forward_without_lengths_remains_usable_for_complete_hand_aware_inputs():
    model = VisionBridgeBaseModel(vocab_size=49, use_hands=True).eval()
    pose, face, left, right = _streams(1, 4)

    with torch.inference_mode():
        logits = model(pose, face, left, right)

    assert logits.shape == (1, 4, 49)
    assert torch.isfinite(logits).all()
