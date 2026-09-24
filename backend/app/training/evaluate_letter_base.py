"""Evaluate a VisionBridge A-Z letter checkpoint on a prepared split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.models.letter_model import INPUT_DIM, load_checkpoint


def load_split(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    path = root / (name + ".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} split: {path}")

    data = np.load(path, allow_pickle=False)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)

    if x.ndim != 2 or x.shape[1] != INPUT_DIM:
        raise ValueError(f"{name} features must have shape [N,{INPUT_DIM}], got {x.shape}")
    if y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError(f"{name} labels are invalid")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} features contain NaN or Inf")

    return torch.from_numpy(x), torch.from_numpy(y)


def evaluate(
    checkpoint: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
) -> dict[str, object]:
    model = load_checkpoint(checkpoint)
    x, y = load_split(data_dir, split)

    if y.min().item() < 0 or y.max().item() >= model.num_classes:
        raise ValueError(f"{split} labels exceed the model vocabulary")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
    )

    confusion = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    correct = 0
    total = 0

    with torch.inference_mode():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()
            actual = batch_y.numpy()

            correct += int((predictions == actual).sum())
            total += len(actual)

            for truth, predicted in zip(actual, predictions):
                confusion[int(truth), int(predicted)] += 1

    per_letter: dict[str, float] = {}
    class_counts: dict[str, int] = {}
    for index, letter in enumerate(model.labels):
        count = int(confusion[index].sum())
        class_counts[letter] = count
        per_letter[letter] = (
            float(confusion[index, index] / count) if count else 0.0
        )

    accuracy = correct / max(total, 1)
    macro_accuracy = sum(per_letter.values()) / len(per_letter)
    worst_letter = min(per_letter, key=per_letter.get)

    # Warm up before measuring the model-only batch latency.
    warmup = x[: min(len(x), batch_size)].to(device)
    with torch.inference_mode():
        for _ in range(5):
            model(warmup)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings: list[float] = []
        sample = x[: min(len(x), batch_size)].to(device)
        for _ in range(20):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

    mean_batch_ms = float(np.mean(timings) * 1000.0)
    mean_sample_ms = mean_batch_ms / max(len(sample), 1)

    report = {
        "checkpoint": str(checkpoint),
        "split": split,
        "device": str(device),
        "samples": total,
        "overall_accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "worst_letter": worst_letter,
        "worst_letter_accuracy": per_letter[worst_letter],
        "per_letter_accuracy": per_letter,
        "class_counts": class_counts,
        "confusion_matrix": confusion.tolist(),
        "labels": list(model.labels),
        "model": {
            "model_version": "visionbridge-letter-base-v3",
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "embedding_dim": model.embedding_dim,
            "num_classes": model.num_classes,
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_bytes": checkpoint.stat().st_size,
        },
        "latency": {
            "warmup_runs": 5,
            "measured_runs": 20,
            "mean_batch_ms": mean_batch_ms,
            "mean_sample_ms": mean_sample_ms,
        },
    }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        checkpoint=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        split=args.split,
        batch_size=args.batch_size,
    )

    print(f"split={report['split']}")
    print(f"samples={report['samples']}")
    print(f"overall_accuracy={report['overall_accuracy']:.4f}")
    print(f"macro_accuracy={report['macro_accuracy']:.4f}")
    print(
        f"worst_letter={report['worst_letter']} "
        f"accuracy={report['worst_letter_accuracy']:.4f}"
    )
    print(f"parameters={report['model']['parameters']}")
    print(f"checkpoint_bytes={report['model']['checkpoint_bytes']}")
    print(
        f"mean_batch_ms={report['latency']['mean_batch_ms']:.3f} "
        f"mean_sample_ms={report['latency']['mean_sample_ms']:.5f}"
    )
    print("per_letter:")
    for letter, accuracy in report["per_letter_accuracy"].items():
        print(f"  {letter}={accuracy:.4f}")

    if args.output_json:
        target = Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report={target}")


if __name__ == "__main__":
    main()
