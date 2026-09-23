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
from app.models.letter_model import MODEL_VERSION, VisionBridgeLetterBaseModel, load_checkpoint

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


def _build_prototypes_from_calibration(
    base_model: VisionBridgeLetterBaseModel,
    samples: list[dict],
) -> tuple[dict[str, list[float]], dict[str, int]]:
    grouped: dict[str, list[np.ndarray]] = {}
    for sample in samples:
        label = str(sample.get("letter", "")).strip().upper()
        raw = sample.get("hand_keypoints")
        if len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or not isinstance(raw, list):
            raise ValueError("Stored calibration sample is invalid")
        grouped.setdefault(label, []).append(embed_hand_vector(base_model, raw))

    if len(grouped) < 2:
        raise ValueError("Stored calibration data does not contain enough letters")

    prototypes = {
        letter: _unit(np.mean(np.stack(items, axis=0), axis=0)).tolist()
        for letter, items in sorted(grouped.items())
    }
    shots = {letter: len(items) for letter, items in sorted(grouped.items())}
    return prototypes, shots


def load_prototype_adapter(weights_path: str, base_model_path: str | Path) -> dict:
    root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    candidate = Path(weights_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise FileNotFoundError("Letter adapter is unavailable")

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("version") != 4 or payload.get("feature_dim") != COMBINED_HAND_DIM:
        raise ValueError("Invalid VisionBridge letter adapter")

    base_path = Path(base_model_path)
    if not base_path.is_file():
        raise FileNotFoundError("Letter base-model checkpoint is missing")

    base_model = load_checkpoint(base_path)
    if payload.get("embedding_dim") != base_model.embedding_dim:
        calibration_samples = payload.get("calibration_samples")
        if not isinstance(calibration_samples, list):
            raise ValueError("Letter adapter requires recalibration for the current model")
    current_hash = _sha256(base_path)
    needs_refresh = (
        payload.get("base_model_version") != MODEL_VERSION
        or payload.get("base_model_sha256") != current_hash
        or payload.get("embedding_dim") != base_model.embedding_dim
        or payload.get("base_model_labels") != list(base_model.labels)
    )

    if needs_refresh:
        calibration_samples = payload.get("calibration_samples")
        if not isinstance(calibration_samples, list) or len(calibration_samples) < 2:
            raise ValueError("Letter adapter requires recalibration for the current model")
        prototypes, shots = _build_prototypes_from_calibration(base_model, calibration_samples)
        payload = dict(payload)
        payload["base_model_version"] = MODEL_VERSION
        payload["base_model_sha256"] = current_hash
        payload["embedding_dim"] = base_model.embedding_dim
        payload["base_model_labels"] = list(base_model.labels)
        payload["prototypes"] = prototypes
        payload["shots"] = shots
        candidate.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    prototypes = payload.get("prototypes")
    if not isinstance(prototypes, dict) or len(prototypes) < 2:
        raise ValueError("Letter adapter must contain at least two prototypes")
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
