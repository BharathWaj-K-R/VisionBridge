"""Dynamic few-shot signer adaptation for the letter recognition model."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from app.core.config import get_settings
from app.models.letter_model import LANDMARK_RUNTIME, MODEL_VERSION, PREPROCESSING_VERSION, VisionBridgeLetterBaseModel, load_checkpoint

HAND_LANDMARKS = 21
HAND_COORDS = 3
HAND_DIM = HAND_LANDMARKS * HAND_COORDS
COMBINED_HAND_DIM = HAND_DIM * 2
MIN_SIMILARITY = 0.35
settings = get_settings()
_base_model: VisionBridgeLetterBaseModel | None = None
_base_model_signature: tuple[int, int, int, str] | None = None


def _normalize_single_hand(values: np.ndarray) -> np.ndarray:
    if values.shape != (HAND_LANDMARKS, HAND_COORDS):
        raise ValueError("Expected one hand as (21, 3), got {}".format(values.shape))
    if not np.isfinite(values).all():
        raise ValueError("Hand keypoints contain a non-finite value")
    if np.allclose(values, 0.0):
        return np.zeros(HAND_DIM, dtype=np.float32)
    centered = values - values[0]
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not math.isfinite(scale) or scale < 1e-6:
        return np.zeros(HAND_DIM, dtype=np.float32)
    return (centered / scale).astype(np.float32).reshape(-1)


def normalize_hand_pair(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.float32)
    if vector.size != COMBINED_HAND_DIM:
        raise ValueError(
            "Expected {} hand features, got {}".format(COMBINED_HAND_DIM, vector.size)
        )
    if not np.isfinite(vector).all():
        raise ValueError("Hand keypoints contain a non-finite value")
    left = _normalize_single_hand(vector[:HAND_DIM].reshape(HAND_LANDMARKS, HAND_COORDS))
    right = _normalize_single_hand(vector[HAND_DIM:].reshape(HAND_LANDMARKS, HAND_COORDS))
    if np.allclose(left, 0.0) and np.allclose(right, 0.0):
        raise ValueError("No visible hand landmarks were provided")
    return np.concatenate([left, right]).astype(np.float32)




V4_COMBINED_HAND_DIM = 141
V4_PREPROCESSING_VERSION = "two-hand-geometry-v2"
V4_FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def normalize_hand_pair_v2(
    values: Iterable[float],
    left_present: bool,
    right_present: bool,
) -> np.ndarray:
    """Create the documented 141D v2 hand/presence/relative-geometry vector."""
    if not isinstance(left_present, (bool, np.bool_)) or not isinstance(
        right_present, (bool, np.bool_)
    ):
        raise ValueError("V4 hand-presence flags must be booleans")

    try:
        vector = np.asarray(list(values), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("V4 raw hand coordinates must be numeric") from exc
    if vector.size != COMBINED_HAND_DIM:
        raise ValueError(
            "Expected {} raw hand coordinates, got {}".format(COMBINED_HAND_DIM, vector.size)
        )
    if not np.isfinite(vector).all():
        raise ValueError("V4 raw hand coordinates contain a non-finite value")

    raw_left = vector[:HAND_DIM].reshape(HAND_LANDMARKS, HAND_COORDS)
    raw_right = vector[HAND_DIM:].reshape(HAND_LANDMARKS, HAND_COORDS)
    left_scale = (
        float(np.linalg.norm(raw_left - raw_left[0], axis=1).max())
        if left_present else 0.0
    )
    right_scale = (
        float(np.linalg.norm(raw_right - raw_right[0], axis=1).max())
        if right_present else 0.0
    )

    left = _normalize_single_hand(raw_left) if left_present else np.zeros(HAND_DIM, dtype=np.float32)
    right = _normalize_single_hand(raw_right) if right_present else np.zeros(HAND_DIM, dtype=np.float32)
    left_flag = np.asarray([1.0 if left_present else 0.0], dtype=np.float32)
    right_flag = np.asarray([1.0 if right_present else 0.0], dtype=np.float32)
    geometry = np.zeros(13, dtype=np.float32)

    if left_present and right_present:
        geometry_scale = max(left_scale, right_scale)
        if math.isfinite(geometry_scale) and geometry_scale >= 1e-6:
            geometry[0:3] = (
                (raw_right[0].astype(np.float64) - raw_left[0].astype(np.float64))
                / geometry_scale
            ).astype(np.float32)
            for offset, fingertip in enumerate(V4_FINGERTIP_INDICES, start=3):
                delta = raw_left[fingertip].astype(np.float64) - raw_right[0].astype(np.float64)
                geometry[offset] = np.float32(np.linalg.norm(delta) / geometry_scale)
            for offset, fingertip in enumerate(V4_FINGERTIP_INDICES, start=8):
                delta = raw_right[fingertip].astype(np.float64) - raw_left[0].astype(np.float64)
                geometry[offset] = np.float32(np.linalg.norm(delta) / geometry_scale)

    result = np.concatenate((left, left_flag, right, right_flag, geometry)).astype(np.float32)
    if result.shape != (V4_COMBINED_HAND_DIM,) or not np.isfinite(result).all():
        raise ValueError("V4 preprocessing produced an invalid feature vector")
    return result


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a degenerate embedding")
    return (vector / norm).astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_letter_base_model() -> VisionBridgeLetterBaseModel:
    """Hot-reload the latest compatible checkpoint whenever the file changes."""
    global _base_model, _base_model_signature
    target = Path(settings.LETTER_BASE_MODEL_PATH)
    if not target.is_file():
        raise FileNotFoundError("Letter base-model checkpoint is missing")
    stat = target.stat()
    current_hash = _sha256(target)
    signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino, current_hash)
    if _base_model is None or _base_model_signature != signature:
        _base_model = load_checkpoint(target)
        _base_model_signature = signature
    return _base_model


def letter_model_status() -> dict[str, str | bool]:
    from app.models.letter_model import checkpoint_status

    return checkpoint_status(settings.LETTER_BASE_MODEL_PATH)


def embed_hand_vector(base_model: VisionBridgeLetterBaseModel, raw: list[float]) -> np.ndarray:
    normalized = normalize_hand_pair(raw)
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    with torch.inference_mode():
        embedding = base_model.embed(tensor)[0].cpu().numpy()
    return _unit(embedding)


def predict_base_letter(
    base_model: VisionBridgeLetterBaseModel,
    raw: list[float],
) -> tuple[str, float]:
    """Run the generic V3 26-class softmax path without signer adaptation."""
    normalized = normalize_hand_pair(raw)
    tensor = torch.from_numpy(normalized).unsqueeze(0)
    with torch.inference_mode():
        logits = base_model(tensor)[0]
        probabilities = torch.softmax(logits, dim=0)
        index = int(torch.argmax(probabilities).item())
    return base_model.labels[index], float(probabilities[index].item())


def fit_prototype_adapter(
    base_model: VisionBridgeLetterBaseModel,
    samples: list[tuple[str, list[float]]],
) -> dict:
    grouped: dict[str, list[np.ndarray]] = {}
    for letter, raw in samples:
        label = letter.strip().upper()
        if len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise ValueError("Unsupported letter label: {!r}".format(letter))
        grouped.setdefault(label, []).append(embed_hand_vector(base_model, raw))

    if len(grouped) < 2:
        raise ValueError("Calibrate at least two different letters before fitting an adapter")

    insufficient = sorted(
        letter
        for letter, items in grouped.items()
        if len(items) < 3
    )
    if insufficient:
        raise ValueError(
            "Each calibrated letter requires at least 3 examples: {}".format(
                ", ".join(insufficient)
            )
        )

    prototypes = {
        letter: _unit(np.mean(np.stack(items, axis=0), axis=0))
        for letter, items in sorted(grouped.items())
    }

    base_path = Path(settings.LETTER_BASE_MODEL_PATH)
    payload = {
        "version": 4,
        "method": "dynamic-base-embedding-prototype",
        "base_model_version": MODEL_VERSION,
        "base_model_sha256": _sha256(base_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "landmark_runtime": LANDMARK_RUNTIME,
        "feature_dim": COMBINED_HAND_DIM,
        "embedding_dim": base_model.embedding_dim,
        "prototypes": {key: value.tolist() for key, value in prototypes.items()},
        "shots": {key: len(value) for key, value in sorted(grouped.items())},
        "base_model_labels": list(base_model.labels),
    }
    return {
        "payload": payload,
        "letters": list(prototypes),
        "shots": payload["shots"],
        "param_count": sum(len(value) for value in payload["prototypes"].values()),
    }


def save_prototype_adapter(payload: dict) -> str:
    """Return a durable database locator; adapter payload is persisted in SignerAdapter.payload_json."""
    if not isinstance(payload, dict) or not payload.get("prototypes"):
        raise ValueError("Cannot persist an empty letter adapter")
    return "database://signer_adapters/{}".format(uuid.uuid4().hex)


def validate_prototype_adapter_payload(payload: dict, base_model_path: str | Path) -> dict:
    if (
        payload.get("version") != 4
        or payload.get("method") != "dynamic-base-embedding-prototype"
        or payload.get("feature_dim") != COMBINED_HAND_DIM
    ):
        raise ValueError("Invalid VisionBridge letter adapter")
    if payload.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Letter adapter preprocessing version is incompatible")
    if payload.get("landmark_runtime") != LANDMARK_RUNTIME:
        raise ValueError("Letter adapter landmark runtime is incompatible")

    base_path = Path(base_model_path)
    if not base_path.is_file():
        raise FileNotFoundError("Letter base-model checkpoint is missing")

    base_model = load_checkpoint(base_path)
    current_hash = _sha256(base_path)

    if payload.get("base_model_version") != MODEL_VERSION:
        raise ValueError("Letter adapter requires recalibration for the current model version")
    if payload.get("base_model_sha256") != current_hash:
        raise ValueError("Letter adapter requires recalibration for the current model")
    if payload.get("embedding_dim") != base_model.embedding_dim:
        raise ValueError("Letter adapter requires recalibration for the current embedding dimension")
    if payload.get("base_model_labels") != list(base_model.labels):
        raise ValueError("Letter adapter requires recalibration for the current label vocabulary")

    prototypes = payload.get("prototypes")
    if not isinstance(prototypes, dict) or len(prototypes) < 2:
        raise ValueError("Letter adapter must contain at least two prototypes")

    for letter, values in prototypes.items():
        if str(letter) not in base_model.labels:
            raise ValueError("Letter adapter contains an unsupported prototype label")
        try:
            vector = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Letter adapter prototype is invalid") from exc
        if vector.shape != (base_model.embedding_dim,) or not np.isfinite(vector).all():
            raise ValueError("Letter adapter prototype is incompatible with the current model")

    return payload


def load_prototype_adapter(weights_path: str, base_model_path: str | Path) -> dict:
    """Legacy filesystem loader retained only for old records; new adapters use payload_json."""
    if weights_path.startswith("database://"):
        raise FileNotFoundError("Database-backed adapter payload must be loaded from SignerAdapter.payload_json")
    root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    candidate = Path(weights_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise FileNotFoundError("Letter adapter is unavailable")

    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Letter adapter could not be decoded") from exc
    return validate_prototype_adapter_payload(payload, base_model_path)

def predict_letter(
    base_model: VisionBridgeLetterBaseModel,
    payload: dict,
    raw: list[float],
) -> tuple[str, float, list[tuple[str, float]]]:
    query = embed_hand_vector(base_model, raw)
    scores = []
    for letter, values in payload["prototypes"].items():
        scores.append(
            (
                str(letter),
                float(np.dot(query, _unit(np.asarray(values, dtype=np.float32)))),
            )
        )
    scores.sort(key=lambda item: item[1], reverse=True)
    best_letter, best_score = scores[0]
    logits = np.asarray([score for _, score in scores], dtype=np.float64) * 10.0
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    confidence = float(probs[0])
    if best_score < MIN_SIMILARITY:
        return "?", confidence, scores
    return best_letter, confidence, scores


def stage_adapter_delete(weights_path: str) -> Path | None:
    adapter_root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    candidate = Path(weights_path).resolve()
    if adapter_root not in candidate.parents:
        raise ValueError("Refusing to delete a path outside the adapter weight directory")
    if not candidate.exists():
        return None
    tombstone = candidate.with_name(".{}.{}.deleting".format(candidate.name, uuid.uuid4().hex))
    candidate.replace(tombstone)
    return tombstone


def restore_adapter_delete(tombstone: Path, original_path: str) -> None:
    adapter_root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    candidate = Path(original_path).resolve()
    staged = tombstone.resolve()
    if adapter_root not in candidate.parents or adapter_root not in staged.parents:
        raise ValueError("Refusing to restore adapter weights outside the configured directory")
    if staged.exists():
        staged.replace(candidate)


def finalize_adapter_delete(tombstone: Path | None) -> None:
    if tombstone is not None and tombstone.exists():
        tombstone.unlink()
