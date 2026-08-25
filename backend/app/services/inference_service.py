"""
Inference service: loads the frozen base model once (module-level singleton),
optionally applies a signer's BridgeAdapter, and returns decoded text +
confidence + latency.

Keypoint extraction (e.g. MediaPipe Holistic on raw video/webcam frames) is
not implemented here. This service expects pre-extracted pose and face tensors.
"""
import pickle
from pathlib import Path
import time

import torch

from app.core.config import get_settings
from app.models.base_model import load_frozen_base_model
from app.models.bridge_adapter import BridgeAdapterStack

settings = get_settings()

_base_model = None
_id_to_token: dict[int, str] = {}


class ModelUnavailableError(RuntimeError):
    """Raised when the configured trained model cannot safely serve inference."""


def _load_vocab(base_model_path: str) -> dict[int, str]:
    """Load and validate the id->token map saved beside the base checkpoint."""
    import json

    vocab_path = Path(base_model_path).with_suffix(".vocab.json")
    if not vocab_path.exists():
        raise ModelUnavailableError(f"Missing model vocabulary: {vocab_path}")

    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    id_to_token = payload.get("id_to_token")
    if not isinstance(id_to_token, list) or not id_to_token:
        raise ModelUnavailableError(f"Invalid model vocabulary file: {vocab_path}")
    if id_to_token[0] != "<blank>":
        raise ModelUnavailableError("Model vocabulary must reserve token 0 for CTC blank")
    return {idx: token for idx, token in enumerate(id_to_token)}


def get_base_model():
    global _base_model, _id_to_token
    if _base_model is None:
        model_path = Path(settings.BASE_MODEL_PATH)
        vocab_path = model_path.with_suffix(".vocab.json")
        if not model_path.is_file() or not vocab_path.is_file():
            raise ModelUnavailableError(
                "The VisionBridge base-model checkpoint and vocabulary must both be installed."
            )
        try:
            _base_model = load_frozen_base_model(str(model_path))
            _id_to_token = _load_vocab(str(model_path))
            if len(_id_to_token) != _base_model.output_head.out_features:
                raise ModelUnavailableError(
                    "Base-model output head size does not match the saved vocabulary size: "
                    f"head={_base_model.output_head.out_features}, vocab={len(_id_to_token)}"
                )
        except ModelUnavailableError:
            _base_model = None
            _id_to_token = {}
            raise
        except (OSError, RuntimeError, ValueError, KeyError, EOFError, pickle.UnpicklingError, UnicodeError) as exc:
            _base_model = None
            _id_to_token = {}
            raise ModelUnavailableError(
                "The VisionBridge base-model checkpoint could not be loaded safely."
            ) from exc
    return _base_model


def model_status() -> dict[str, str | bool]:
    """Validate checkpoint/vocabulary presence and output-head compatibility.

    This readiness probe intentionally loads the checkpoint state once because
    PyTorch's state-dict format does not provide a cheap header-only shape API.
    The live model is still cached separately by ``get_base_model``.
    """
    model_path = Path(settings.BASE_MODEL_PATH)
    vocab_path = model_path.with_suffix(".vocab.json")
    if not model_path.is_file() or not vocab_path.is_file():
        return {"available": False, "status": "unavailable"}

    try:
        id_to_token = _load_vocab(str(model_path))
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        output_head_weight = state.get("output_head.weight")
        if output_head_weight is None:
            return {"available": False, "status": "invalid_checkpoint"}
        if len(id_to_token) != int(output_head_weight.shape[0]):
            return {"available": False, "status": "vocabulary_mismatch"}
    except Exception:
        return {"available": False, "status": "invalid_checkpoint"}

    return {"available": True, "status": "ready"}


CTC_BLANK_ID = 0


def decode_logits(logits: torch.Tensor) -> tuple[str, float]:
    """Greedy CTC decode: collapse consecutive repeats, then drop blanks."""
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape [1, T, V], got {tuple(logits.shape)}")
    if logits.shape[-1] <= CTC_BLANK_ID:
        raise ValueError("Logits do not contain the configured CTC blank class")
    if _id_to_token and len(_id_to_token) != logits.shape[-1]:
        raise ValueError(
            f"Decoder vocabulary/logit mismatch: vocab={len(_id_to_token)}, logits={logits.shape[-1]}"
        )

    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = probs.max(dim=-1)
    ids = top_ids[0].tolist()
    frame_probs = top_probs[0].tolist()

    collapsed: list[tuple[int, float]] = []
    previous_id = None
    for token_id, probability in zip(ids, frame_probs):
        if token_id != previous_id:
            collapsed.append((int(token_id), float(probability)))
            previous_id = token_id

    non_blank = [
        (token_id, probability)
        for token_id, probability in collapsed
        if token_id != CTC_BLANK_ID
    ]
    tokens = [_id_to_token.get(i, f"<{i}>") for i, _ in non_blank]
    text = "".join(tokens)
    confidence = float(sum(p for _, p in non_blank) / len(non_blank)) if non_blank else 0.0
    return text or "(no sign detected)", confidence


def run_inference(
    pose: torch.Tensor,
    face: torch.Tensor,
    adapter: BridgeAdapterStack | None = None,
) -> dict:
    model = get_base_model()
    start = time.perf_counter()

    with torch.no_grad():
        if adapter is not None:
            logits = adapter.forward_with_base(model, pose, face)
        else:
            logits = model(pose, face)

    latency_ms = (time.perf_counter() - start) * 1000
    text, confidence = decode_logits(logits)

    return {
        "predicted_text": text,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "used_adapter": adapter is not None,
    }
