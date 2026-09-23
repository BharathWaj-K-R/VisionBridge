"""Few-shot signer adaptation for isolated hand-letter recognition.

This module intentionally avoids a trained checkpoint. A signer calibrates a
small number of examples per letter, and the adapter stores normalized hand
prototypes. Prediction is cosine similarity against those prototypes.
"""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np

from app.core.config import get_settings

HAND_LANDMARKS = 21
HAND_COORDS = 3
HAND_DIM = HAND_LANDMARKS * HAND_COORDS
COMBINED_HAND_DIM = HAND_DIM * 2
MIN_SIMILARITY = 0.35

settings = get_settings()


def _normalize_single_hand(values: np.ndarray) -> np.ndarray:
    if values.shape != (HAND_LANDMARKS, HAND_COORDS):
        raise ValueError(
            "Expected one hand as ({}, {}), got {}".format(
                HAND_LANDMARKS, HAND_COORDS, values.shape
            )
        )
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
    """Translate-scale normalize left and right hand landmarks independently."""
    vector = np.asarray(list(values), dtype=np.float32)
    if vector.size != COMBINED_HAND_DIM:
        raise ValueError(
            "Expected {} hand features (two 21-point XYZ hands), got {}".format(
                COMBINED_HAND_DIM, vector.size
            )
        )
    if not np.isfinite(vector).all():
        raise ValueError("Hand keypoints contain a non-finite value")
    left = _normalize_single_hand(vector[:HAND_DIM].reshape(HAND_LANDMARKS, HAND_COORDS))
    right = _normalize_single_hand(vector[HAND_DIM:].reshape(HAND_LANDMARKS, HAND_COORDS))
    return np.concatenate([left, right]).astype(np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot build an adapter from a degenerate hand sample")
    return (vector / norm).astype(np.float32)


def fit_prototype_adapter(samples: list[tuple[str, list[float]]]) -> dict:
    """Fit one normalized prototype per letter from real signer examples."""
    grouped: dict[str, list[np.ndarray]] = {}
    for letter, raw in samples:
        normalized_letter = letter.strip().upper()
        if len(normalized_letter) != 1 or normalized_letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise ValueError("Unsupported letter label: {!r}".format(letter))
        grouped.setdefault(normalized_letter, []).append(normalize_hand_pair(raw))

    if len(grouped) < 2:
        raise ValueError("Calibrate at least two different letters before fitting an adapter")

    prototypes = {
        letter: _unit(np.mean(np.stack(items, axis=0), axis=0))
        for letter, items in sorted(grouped.items())
    }
    payload = {
        "version": 1,
        "feature_dim": COMBINED_HAND_DIM,
        "method": "normalized-prototype-cosine",
        "prototypes": {letter: proto.tolist() for letter, proto in prototypes.items()},
        "shots": {letter: len(items) for letter, items in sorted(grouped.items())},
    }
    return {
        "payload": payload,
        "letters": list(prototypes),
        "shots": payload["shots"],
        "param_count": sum(len(values) for values in payload["prototypes"].values()),
    }


def save_prototype_adapter(payload: dict) -> str:
    root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "letter_adapter_{}.json".format(uuid.uuid4().hex)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(target)


def load_prototype_adapter(weights_path: str) -> dict:
    root = Path(settings.ADAPTER_WEIGHTS_DIR).resolve()
    candidate = Path(weights_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise FileNotFoundError("Letter adapter is unavailable")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("feature_dim") != COMBINED_HAND_DIM:
        raise ValueError("Invalid VisionBridge letter adapter")
    prototypes = payload.get("prototypes")
    if not isinstance(prototypes, dict) or len(prototypes) < 2:
        raise ValueError("Letter adapter must contain at least two prototypes")
    return payload


def predict_letter(payload: dict, raw: list[float]) -> tuple[str, float, list[tuple[str, float]]]:
    query = _unit(normalize_hand_pair(raw))
    prototypes = payload["prototypes"]
    scores: list[tuple[str, float]] = []
    for letter, values in prototypes.items():
        prototype = _unit(np.asarray(values, dtype=np.float32))
        scores.append((str(letter), float(np.dot(query, prototype))))
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
