"""Verify browser-model math against the PyTorch V3 model contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from app.models.letter_model import build_browser_payload, load_checkpoint


def gelu(value: np.ndarray) -> np.ndarray:
    return 0.5 * value * (
        1.0
        + np.tanh(
            0.7978845608028654
            * (value + 0.044715 * np.power(value, 3))
        )
    )


def layer_norm(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + 1e-5) * weight + bias


def dense(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return values @ weight.T + bias


def browser_forward(payload: dict, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    layers = payload["layers"]
    x = layer_norm(
        raw,
        np.asarray(layers["input_norm"]["weight"], dtype=np.float32),
        np.asarray(layers["input_norm"]["bias"], dtype=np.float32),
    )
    hidden = gelu(
        dense(
            x,
            np.asarray(layers["hidden"]["weight"], dtype=np.float32),
            np.asarray(layers["hidden"]["bias"], dtype=np.float32),
        )
    )
    embedding = layer_norm(
        dense(
            hidden,
            np.asarray(layers["embedding"]["weight"], dtype=np.float32),
            np.asarray(layers["embedding"]["bias"], dtype=np.float32),
        ),
        np.asarray(layers["embedding_norm"]["weight"], dtype=np.float32),
        np.asarray(layers["embedding_norm"]["bias"], dtype=np.float32),
    )
    embedding = gelu(embedding)
    logits = dense(
        embedding,
        np.asarray(layers["head"]["weight"], dtype=np.float32),
        np.asarray(layers["head"]["bias"], dtype=np.float32),
    )
    return embedding, logits


def verify(checkpoint: Path, samples: int, tolerance: float) -> dict:
    model = load_checkpoint(checkpoint)
    payload = build_browser_payload(model, "parity-check")
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(samples, model.input_dim)).astype(np.float32)

    with torch.inference_mode():
        tensor = torch.from_numpy(raw)
        torch_embedding = model.embed(tensor).numpy()
        torch_logits = model(tensor).numpy()

    browser_embedding, browser_logits = browser_forward(payload, raw)

    embedding_error = float(np.max(np.abs(torch_embedding - browser_embedding)))
    logits_error = float(np.max(np.abs(torch_logits - browser_logits)))

    return {
        "samples": samples,
        "embedding_max_abs_error": embedding_error,
        "logits_max_abs_error": logits_error,
        "tolerance": tolerance,
        "passed": embedding_error <= tolerance and logits_error <= tolerance,
        "contract": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "embedding_dim": model.embedding_dim,
            "num_classes": model.num_classes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    report = verify(Path(args.checkpoint), args.samples, args.tolerance)
    print(json.dumps(report, indent=2))
    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
