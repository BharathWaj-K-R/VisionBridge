import torch

from app.models.base_model import FACE_INPUT_DIM, POSE_INPUT_DIM, VisionBridgeBaseModel


def test_padding_mask_blocks_padded_frames_from_valid_outputs():
    torch.manual_seed(7)
    model = VisionBridgeBaseModel(vocab_size=49).eval()
    batch, frames, valid = 2, 8, 5
    pose = torch.randn(batch, frames, POSE_INPUT_DIM)
    face = torch.randn(batch, frames, FACE_INPUT_DIM)
    lengths = torch.tensor([valid, frames], dtype=torch.long)

    with torch.inference_mode():
        baseline = model(pose, face, lengths)

    modified_pose = pose.clone()
    modified_face = face.clone()
    modified_pose[0, valid:] = torch.randn_like(modified_pose[0, valid:]) * 1000.0
    modified_face[0, valid:] = torch.randn_like(modified_face[0, valid:]) * 1000.0

    with torch.inference_mode():
        masked = model(modified_pose, modified_face, lengths)

    assert torch.isfinite(masked).all()
    assert torch.allclose(baseline[0, :valid], masked[0, :valid], atol=1e-5, rtol=1e-5)


def test_forward_without_lengths_remains_backward_compatible():
    model = VisionBridgeBaseModel(vocab_size=49).eval()
    pose = torch.randn(1, 4, POSE_INPUT_DIM)
    face = torch.randn(1, 4, FACE_INPUT_DIM)

    with torch.inference_mode():
        logits = model(pose, face)

    assert logits.shape == (1, 4, 49)
    assert torch.isfinite(logits).all()
