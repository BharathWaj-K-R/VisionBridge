"""Migrate a legacy V3 checkpoint into the current metadata-bound envelope."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.models.letter_model import (
    INPUT_DIM,
    LANDMARK_RUNTIME,
    LETTER_LABELS,
    MODEL_VERSION,
    NUM_CLASSES,
    PREPROCESSING_VERSION,
)


EXPECTED_STATE_KEYS = {
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.1.weight",
    "encoder.1.bias",
    "encoder.4.weight",
    "encoder.4.bias",
    "encoder.5.weight",
    "encoder.5.bias",
    "output_head.weight",
    "output_head.bias",
}


def migrate_checkpoint(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(f"Legacy checkpoint is missing: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("Migration output must differ from the source checkpoint")

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Legacy checkpoint must be a dictionary")

    if payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Legacy checkpoint is not VisionBridge V3")
    if int(payload.get("input_dim", 0)) != INPUT_DIM:
        raise ValueError("Legacy checkpoint input dimension is incompatible")
    if int(payload.get("num_classes", 0)) != NUM_CLASSES:
        raise ValueError("Legacy checkpoint class count is incompatible")
    if tuple(payload.get("labels", ())) != LETTER_LABELS:
        raise ValueError("Legacy checkpoint label vocabulary is incompatible")
    if int(payload.get("hidden_dim", 0)) != 128:
        raise ValueError("Legacy V3 hidden dimension is unexpected")
    if int(payload.get("embedding_dim", 0)) != 64:
        raise ValueError("Legacy V3 embedding dimension is unexpected")

    state = payload.get("state_dict")
    if not isinstance(state, dict) or set(state) != EXPECTED_STATE_KEYS:
        raise ValueError("Legacy checkpoint state_dict keys are incompatible")
    if any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("Legacy checkpoint state_dict contains invalid values")

    migrated = dict(payload)
    migrated["preprocessing_version"] = PREPROCESSING_VERSION
    migrated["landmark_runtime"] = LANDMARK_RUNTIME

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "model_version": migrated["model_version"],
        "preprocessing_version": migrated["preprocessing_version"],
        "landmark_runtime": migrated["landmark_runtime"],
        "input_dim": migrated["input_dim"],
        "hidden_dim": migrated["hidden_dim"],
        "embedding_dim": migrated["embedding_dim"],
        "num_classes": migrated["num_classes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the current preprocessing/runtime metadata to a legacy V3 checkpoint."
    )
    parser.add_argument("--input", required=True, help="Legacy V3 checkpoint")
    parser.add_argument("--output", required=True, help="Migrated checkpoint")
    args = parser.parse_args()

    report = migrate_checkpoint(Path(args.input), Path(args.output))
    for key, value in report.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
