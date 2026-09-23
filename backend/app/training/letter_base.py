"""Train the configurable VisionBridge letter base model on prepared hand landmarks."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.models.letter_model import INPUT_DIM, VisionBridgeLetterBaseModel, save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(root: Path, name: str):
    path = root / (name + ".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} split: {path}")

    data = np.load(path, allow_pickle=False)
    x = torch.from_numpy(data["x"]).float()
    y = torch.from_numpy(data["y"]).long()

    if (
        x.ndim != 2
        or x.shape[1] != INPUT_DIM
        or y.ndim != 1
        or len(x) != len(y)
        or len(x) == 0
    ):
        raise ValueError(f"Invalid {name} split")
    if not torch.isfinite(x).all():
        raise ValueError(f"Invalid {name} split: features contain NaN/Inf")
    return x, y


def class_accuracy(
    model: VisionBridgeLetterBaseModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, int]]:
    model.eval()
    correct = torch.zeros(model.num_classes, dtype=torch.long)
    totals = torch.zeros(model.num_classes, dtype=torch.long)

    with torch.inference_mode():
        for x, y in loader:
            predictions = model(x.to(device)).argmax(1).cpu()
            y = y.cpu()

            for label in range(model.num_classes):
                mask = y == label
                totals[label] += int(mask.sum())
                correct[label] += int((predictions[mask] == label).sum())

    if (totals == 0).any():
        missing = [
            model.labels[i]
            for i, count in enumerate(totals.tolist())
            if count == 0
        ]
        raise ValueError(
            f"Validation split has no samples for: {', '.join(missing)}"
        )

    per_letter = {
        model.labels[i]: float(correct[i].item() / totals[i].item())
        for i in range(model.num_classes)
    }
    counts = {
        model.labels[i]: int(totals[i].item())
        for i in range(model.num_classes)
    }

    total_correct = int(correct.sum())
    total_samples = int(totals.sum())
    overall = total_correct / max(total_samples, 1)
    return overall, per_letter, counts


def format_class_scores(scores: dict[str, float]) -> str:
    return " ".join(
        f"{letter}={score:.3f}"
        for letter, score in scores.items()
    )


def train_model(
    *,
    root: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    target_class_accuracy: float,
    seed: int,
    hidden_dim: int,
    embedding_dim: int,
    dropout: float,
) -> dict[str, object]:
    seed_everything(seed)

    labels = tuple(
        json.loads(
            (root / "labels.json").read_text(encoding="utf-8")
        )["labels"]
    )
    if not labels:
        raise ValueError("Dataset must define at least one class label")

    train = DataLoader(
        TensorDataset(*load_split(root, "train")),
        batch_size=batch_size,
        shuffle=True,
    )
    val = DataLoader(
        TensorDataset(*load_split(root, "val")),
        batch_size=batch_size,
    )
    test = DataLoader(
        TensorDataset(*load_split(root, "test")),
        batch_size=batch_size,
    )

    train_labels = train.dataset.tensors[1]
    missing_train = [
        label
        for index, label in enumerate(labels)
        if not (train_labels == index).any()
    ]
    if missing_train:
        raise ValueError(
            f"Training split has no samples for: {', '.join(missing_train)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionBridgeLetterBaseModel(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        labels=labels,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, min(8, epochs // 10)),
        min_lr=1e-5,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_key = (-1.0, -1.0)
    best_state = None
    best_epoch = 0
    best_per_letter: dict[str, float] = {}
    reached_target = False

    print(
        f"device={device} "
        f"target_per_letter_accuracy={target_class_accuracy:.3f} "
        f"max_epochs={epochs}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train:
            if int(y.min()) < 0 or int(y.max()) >= model.num_classes:
                raise ValueError(
                    "Training labels exceed the checkpoint label vocabulary"
                )

            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = loss_fn(logits, y.to(device))

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite base-model loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if not all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("Non-finite base-model gradient")

            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total += len(y)

        val_acc, per_letter, _ = class_accuracy(model, val, device)
        worst_letter = min(per_letter, key=per_letter.get)
        worst_score = per_letter[worst_letter]
        scheduler.step(worst_score)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} "
            f"train_loss={total_loss / max(total, 1):.4f} "
            f"val_accuracy={val_acc:.4f} "
            f"worst={worst_letter}:{worst_score:.4f} "
            f"lr={current_lr:.6f}"
        )
        print(f"per_letter: {format_class_scores(per_letter)}")

        score_key = (worst_score, val_acc)
        if score_key > best_key:
            best_key = score_key
            best_epoch = epoch
            best_per_letter = dict(per_letter)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        reached_target = all(
            score >= target_class_accuracy
            for score in per_letter.values()
        )
        if reached_target:
            print(
                "TARGET REACHED: every A-Z letter met "
                "the validation accuracy target."
            )
            break

    if best_state is None:
        raise RuntimeError("Base-model training produced no checkpoint")

    model.load_state_dict(best_state)
    test_acc, test_per_letter, test_counts = class_accuracy(
        model, test, device
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, output_path)

    print(f"best_epoch={best_epoch}")
    print(f"best_validation_accuracy={best_key[1]:.4f}")
    print(f"best_worst_letter_accuracy={best_key[0]:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"test_per_letter: {format_class_scores(test_per_letter)}")
    print(f"checkpoint={output_path}")

    if not reached_target:
        missing = [
            letter
            for letter, score in best_per_letter.items()
            if score < target_class_accuracy
        ]
        print(
            "TARGET NOT REACHED within the epoch cap. "
            f"Best checkpoint was saved. "
            f"Validation letters below target: {', '.join(missing)}"
        )

    return {
        "reached_target": reached_target,
        "best_epoch": best_epoch,
        "validation_accuracy": best_key[1],
        "worst_validation_accuracy": best_key[0],
        "validation_per_letter": best_per_letter,
        "test_accuracy": test_acc,
        "test_per_letter": test_per_letter,
        "test_counts": test_counts,
        "checkpoint": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--target-class-accuracy",
        type=float,
        default=1.0,
        help="Stop when every validation letter reaches this accuracy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    args = parser.parse_args()

    if not 0 < args.target_class_accuracy <= 1:
        raise ValueError(
            "--target-class-accuracy must be > 0 and <= 1"
        )

    train_model(
        root=Path(args.data_dir),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        target_class_accuracy=args.target_class_accuracy,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
