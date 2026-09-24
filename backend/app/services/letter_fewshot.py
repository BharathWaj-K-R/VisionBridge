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
_base_model_signature: tuple[int, int, int] | None = None


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
    signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
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
        "calibration_samples": [
            {"letter": letter, "hand_keypoints": normalize_hand_pair(raw).tolist()}
            for letter, raw in samples
        ],
    }
    return {
        "payload": payload,
        "letters": list(prototypes),
        "shots": payload["shots"],
        "param_count": sum(len(value) for value in payload["prototypes"].values()),
    }


def save_prototype_adapter(payload: dict) -> str:
    root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "letter_adapter_{}.json".format(uuid.uuid4().hex)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(target)


def load_prototype_adapter(weights_path: str, base_model_path: str | Path) -> dict:
    root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    candidate = Path(weights_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise FileNotFoundError("Letter adapter is unavailable")

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if (
        payload.get("version") != 4
        or payload.get("method") != "dynamic-base-embedding-prototype"
        or payload.get("feature_dim") != COMBINED_HAND_DIM
    ):
        raise ValueError("Invalid VisionBridge letter adapter")
    if payload.get("preprocessing_version") not in (None, PREPROCESSING_VERSION):
        raise ValueError("Letter adapter preprocessing version is incompatible")
    if payload.get("landmark_runtime") not in (None, LANDMARK_RUNTIME):
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
