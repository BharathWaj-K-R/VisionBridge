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

    if x.ndim != 2 or x.shape[1] != INPUT_DIM or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError(f"Invalid {name} split")
    if not torch.isfinite(x).all():
        raise ValueError(f"Invalid {name} split: features contain NaN/Inf")
    return x, y


def accuracy(model: VisionBridgeLetterBaseModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1)
            correct += int((pred == y.to(device)).sum())
            total += len(y)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    args = parser.parse_args()

    seed_everything(args.seed)
    root = Path(args.data_dir)
    labels = tuple(json.loads((root / "labels.json").read_text(encoding="utf-8"))["labels"])
    if not labels:
        raise ValueError("Dataset must define at least one class label")

    train = DataLoader(
        TensorDataset(*load_split(root, "train")),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val = DataLoader(TensorDataset(*load_split(root, "val")), batch_size=args.batch_size)
    test = DataLoader(TensorDataset(*load_split(root, "test")), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionBridgeLetterBaseModel(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        labels=labels,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    best = -1.0
    best_state = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train:
            if int(y.min()) < 0 or int(y.max()) >= model.num_classes:
                raise ValueError("Training labels exceed the checkpoint label vocabulary")

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite base-model loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("Non-finite base-model gradient")

            opt.step()
            total_loss += float(loss.item()) * len(y)
            total += len(y)

        val_acc = accuracy(model, val, device)
        print(
            f"epoch={epoch:02d} train_loss={total_loss / max(total, 1):.4f} "
            f"val_accuracy={val_acc:.4f}"
        )

        if val_acc > best:
            best = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Base-model training produced no checkpoint")

    model.load_state_dict(best_state)
    test_acc = accuracy(model, test, device)
    save_checkpoint(model, args.output)

    print(f"best_val_accuracy={best:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"checkpoint={args.output}")
    print(
        f"model_config=hidden:{model.hidden_dim},embedding:{model.embedding_dim},"
        f"classes:{model.num_classes},dropout:{model.dropout}"
    )


if __name__ == "__main__":
    main()
