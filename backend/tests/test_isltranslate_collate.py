"""Regression tests for long clips and synchronized multimodal collation."""
import torch

from app.models.base_model import (
    FACE_INPUT_DIM,
    HAND_INPUT_DIM,
    MAX_SEQUENCE_LENGTH,
    POSE_INPUT_DIM,
    VisionBridgeBaseModel,
)
from app.training.isltranslate import collate_ctc_batch


def _make_item(uid: str, frames: int, n_labels: int = 5) -> dict:
    return {
        "uid": uid,
        "text": "placeholder",
        "pose": torch.randn(frames, POSE_INPUT_DIM),
        "face": torch.randn(frames, FACE_INPUT_DIM),
        "left_hand": torch.randn(frames, HAND_INPUT_DIM),
        "right_hand": torch.randn(frames, HAND_INPUT_DIM),
        "labels": torch.randint(1, 20, (n_labels,), dtype=torch.long),
    }


def test_oversized_clip_is_downsampled_to_max_sequence_length():
    batch = [_make_item("clip0076", frames=4500), _make_item("clip0001", frames=119)]
    out = collate_ctc_batch(batch)

    assert out["pose"].shape == (2, MAX_SEQUENCE_LENGTH, POSE_INPUT_DIM)
    assert out["face"].shape == (2, MAX_SEQUENCE_LENGTH, FACE_INPUT_DIM)
    assert out["left_hand"].shape == (2, MAX_SEQUENCE_LENGTH, HAND_INPUT_DIM)
    assert out["right_hand"].shape == (2, MAX_SEQUENCE_LENGTH, HAND_INPUT_DIM)
    assert out["input_lengths"].tolist() == [MAX_SEQUENCE_LENGTH, 119]
    assert (out["input_lengths"] <= MAX_SEQUENCE_LENGTH).all()


def test_normal_length_clips_are_left_unchanged():
    for frames in (25, 50, 100, 119, 500, MAX_SEQUENCE_LENGTH):
        batch = [_make_item("clip", frames=frames)]
        out = collate_ctc_batch(batch)
        assert out["pose"].shape[1] == frames
        assert out["left_hand"].shape[1] == frames
        assert out["right_hand"].shape[1] == frames
        assert out["input_lengths"].tolist() == [frames]


def test_modality_frame_mismatch_raises_with_uid():
    bad_item = _make_item("clip_bad", frames=100)
    bad_item["right_hand"] = torch.randn(99, HAND_INPUT_DIM)
    try:
        collate_ctc_batch([bad_item])
    except ValueError as exc:
        assert "clip_bad" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched modality frame counts")


def test_downsampled_batch_passes_through_hand_aware_model_with_finite_ctc_loss():
    batch = [_make_item("clip0076", frames=4500), _make_item("clip0001", frames=119)]
    out = collate_ctc_batch(batch)

    model = VisionBridgeBaseModel(vocab_size=49, use_hands=True)
    logits = model(
        out["pose"],
        out["face"],
        out["left_hand"],
        out["right_hand"],
        out["input_lengths"],
    )

    assert logits.shape[:2] == (2, MAX_SEQUENCE_LENGTH)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    loss = loss_fn(
        log_probs,
        out["labels"],
        out["input_lengths"],
        out["label_lengths"],
    )
    assert torch.isfinite(loss)
