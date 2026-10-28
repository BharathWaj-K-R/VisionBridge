"""Train the configurable VisionBridge letter base model on prepared hand landmarks."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from app.models.letter_model import (
    EMBEDDING_DIM,
    INPUT_DIM,
    LETTER_LABELS,
    MODEL_VERSION,
    NUM_CLASSES,
    V4_MODEL_VERSION,
    V4_VARIANTS,
    V4_VARIANT_INPUT_DIMS,
    V4_VARIANT_PREPROCESSING,
    VisionBridgeLetterBaseModel,
    VisionBridgeLetterBaseV4Model,
    save_checkpoint,
    save_v4_checkpoint,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment_landmark_batch(
    features: torch.Tensor,
    *,
    jitter_std: float = 0.0,
    rotation_degrees: float = 0.0,
) -> torch.Tensor:
    """Apply mild label-preserving perturbations to normalized hand landmarks.

    Augmentation is training-only. Empty hand slots stay exactly zero, wrist
    anchors stay at the origin, and the 126D feature contract is unchanged.
    """
    if features.ndim != 2 or features.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected [batch, {INPUT_DIM}] features")
    if not np.isfinite(jitter_std) or jitter_std < 0:
        raise ValueError("jitter_std must be finite and non-negative")
    if not np.isfinite(rotation_degrees) or not 0 <= rotation_degrees <= 15:
        raise ValueError("rotation_degrees must be between 0 and 15")

    if jitter_std == 0 and rotation_degrees == 0:
        return features

    augmented = features.reshape(-1, 2, 21, 3).clone()
    radians = float(rotation_degrees) * np.pi / 180.0

    for hand_index in range(2):
        hand = augmented[:, hand_index]
        active = hand.abs().amax(dim=(1, 2)) > 1e-8

        if rotation_degrees > 0:
            angles = (torch.rand(len(hand), device=hand.device) * 2.0 - 1.0) * radians
            cosines = torch.cos(angles).unsqueeze(1)
            sines = torch.sin(angles).unsqueeze(1)
            x = hand[:, :, 0].clone()
            y = hand[:, :, 1].clone()
            rotated_x = x * cosines - y * sines
            rotated_y = x * sines + y * cosines
            active_points = active.unsqueeze(1)
            hand[:, :, 0] = torch.where(active_points, rotated_x, x)
            hand[:, :, 1] = torch.where(active_points, rotated_y, y)

        if jitter_std > 0:
            noise = torch.randn_like(hand) * float(jitter_std)
            noise[:, 0, :] = 0.0
            noise *= active[:, None, None]
            hand.add_(noise)

        augmented[:, hand_index] = hand

    return augmented.reshape(-1, INPUT_DIM)


def build_focus_weights(
    labels: tuple[str, ...],
    focus_letters: tuple[str, ...],
    focus_weight: float,
) -> torch.Tensor:
    """Return per-class loss weights; focused labels get a configurable boost."""
    if not np.isfinite(focus_weight) or focus_weight < 1.0:
        raise ValueError("focus_weight must be finite and at least 1.0")
    unknown = sorted(set(focus_letters) - set(labels))
    if unknown:
        raise ValueError("Unknown focus letter(s): " + ", ".join(unknown))

    weights = torch.ones(len(labels), dtype=torch.float32)
    for letter in focus_letters:
        weights[labels.index(letter)] = float(focus_weight)
    return weights


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_name: str,
    focal_gamma: float,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy or focal loss with optional hard-class weighting."""
    if loss_name not in {"cross_entropy", "focal"}:
        raise ValueError("loss_name must be 'cross_entropy' or 'focal'")
    if not np.isfinite(focal_gamma) or focal_gamma < 0:
        raise ValueError("focal_gamma must be finite and non-negative")

    per_example = F.cross_entropy(logits, targets, reduction="none")
    if loss_name == "focal":
        probability_of_truth = F.softmax(logits, dim=1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        per_example = per_example * (1.0 - probability_of_truth).clamp_min(0.0).pow(
            float(focal_gamma)
        )

    sample_weights = class_weights.to(logits.device)[targets]
    return (per_example * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def load_split(root: Path, name: str, input_dim: int = INPUT_DIM):
    path = root / (name + ".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} split: {path}")

    data = np.load(path, allow_pickle=False)
    x = torch.from_numpy(data["x"]).float()
    y = torch.from_numpy(data["y"]).long()

    if (
        x.ndim != 2
        or x.shape[1] != input_dim
        or y.ndim != 1
        or len(x) != len(y)
        or len(x) == 0
    ):
        raise ValueError(f"Invalid {name} split: expected features with width {input_dim}")
    if not torch.isfinite(x).all():
        raise ValueError(f"Invalid {name} split: features contain NaN/Inf")
    return x, y


def class_accuracy(
    model: VisionBridgeLetterBaseModel | VisionBridgeLetterBaseV4Model,
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
    loss_name: str = "cross_entropy",
    focal_gamma: float = 1.5,
    landmark_jitter_std: float = 0.0,
    rotation_degrees: float = 0.0,
    focus_letters: tuple[str, ...] = (),
    focus_weight: float = 1.0,
    validation_only: bool = False,
    model_version: str = "v3",
    v4_variant: str = "full_v4",
) -> dict[str, object]:
    seed_everything(seed)

    if model_version not in {"v3", "v4"}:
        raise ValueError("model_version must be 'v3' or 'v4'")
    if model_version == "v3":
        input_dim = INPUT_DIM
        model_variant = MODEL_VERSION
        preprocessing_version = "two-hand-wrist-scale-v1"
    else:
        if v4_variant not in V4_VARIANTS:
            raise ValueError("Unsupported V4 architecture variant")
        input_dim = V4_VARIANT_INPUT_DIMS[v4_variant]
        model_variant = V4_MODEL_VERSION
        preprocessing_version = V4_VARIANT_PREPROCESSING[v4_variant]
    if input_dim != INPUT_DIM and (landmark_jitter_std != 0 or rotation_degrees != 0):
        raise ValueError("Landmark augmentation is only implemented for 126D inputs; use zero jitter/rotation for V4 comparison")

    labels = tuple(
        json.loads(
            (root / "labels.json").read_text(encoding="utf-8")
        )["labels"]
    )
    if labels != LETTER_LABELS or len(labels) != NUM_CLASSES:
        raise ValueError("Dataset must define the active A-Z label vocabulary")

    train = DataLoader(
        TensorDataset(*load_split(root, "train", input_dim=input_dim)),
        batch_size=batch_size,
        shuffle=True,
    )
    val = DataLoader(
        TensorDataset(*load_split(root, "val", input_dim=input_dim)),
        batch_size=batch_size,
    )
    test = None if validation_only else DataLoader(
        TensorDataset(*load_split(root, "test", input_dim=input_dim)),
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
    if model_version == "v3":
        model = VisionBridgeLetterBaseModel(
            input_dim=INPUT_DIM,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            labels=labels,
            dropout=dropout,
        )
    else:
        model = VisionBridgeLetterBaseV4Model(
            variant=v4_variant,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            labels=labels,
            dropout=dropout,
        )
    model = model.to(device)

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
    class_weights = build_focus_weights(labels, focus_letters, focus_weight).to(device)
    if loss_name not in {"cross_entropy", "focal"}:
        raise ValueError("loss_name must be 'cross_entropy' or 'focal'")
    if not np.isfinite(focal_gamma) or focal_gamma < 0:
        raise ValueError("focal_gamma must be finite and non-negative")
    if not np.isfinite(landmark_jitter_std) or landmark_jitter_std < 0:
        raise ValueError("landmark_jitter_std must be finite and non-negative")
    if not np.isfinite(rotation_degrees) or not 0 <= rotation_degrees <= 15:
        raise ValueError("rotation_degrees must be between 0 and 15")

    best_key = (-1.0, -1.0, -1.0)
    best_state = None
    best_epoch = 0
    best_per_letter: dict[str, float] = {}
    reached_target = False

    print(
        f"device={device} "
        f"target_per_letter_accuracy={target_class_accuracy:.3f} "
        f"loss={loss_name} focal_gamma={focal_gamma:.2f} "
        f"jitter_std={landmark_jitter_std:.4f} rotation_degrees={rotation_degrees:.1f} "
        f"focus_letters={','.join(focus_letters) or 'none'} focus_weight={focus_weight:.2f} "
        f"validation_only={validation_only} max_epochs={epochs}"
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
            if model_version == "v3":
                batch_x = augment_landmark_batch(
                    x.to(device),
                    jitter_std=landmark_jitter_std,
                    rotation_degrees=rotation_degrees,
                )
            else:
                # V4 input includes presence and relative geometry; the current
                # augmentation implementation is defined only for V1 126D data.
                batch_x = x.to(device)
            batch_y = y.to(device)
            logits = model(batch_x)
            loss = classification_loss(
                logits,
                batch_y,
                loss_name=loss_name,
                focal_gamma=focal_gamma,
                class_weights=class_weights,
            )

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
        val_macro_accuracy = float(sum(per_letter.values()) / len(per_letter))
        scheduler.step(worst_score)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} "
            f"train_loss={total_loss / max(total, 1):.4f} "
            f"val_accuracy={val_acc:.4f} "
            f"val_macro_accuracy={val_macro_accuracy:.4f} "
            f"worst={worst_letter}:{worst_score:.4f} "
            f"lr={current_lr:.6f}"
        )
        print(f"per_letter: {format_class_scores(per_letter)}")

        score_key = (worst_score, val_macro_accuracy, val_acc)
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
    if test is not None:
        test_acc, test_per_letter, test_counts = class_accuracy(model, test, device)
    else:
        test_acc, test_per_letter, test_counts = None, None, None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if model_version == "v3":
        save_checkpoint(model, output_path)
    else:
        save_v4_checkpoint(model, output_path)

    print(f"best_epoch={best_epoch}")
    print(f"best_validation_accuracy={best_key[2]:.4f}")
    print(f"best_validation_macro_accuracy={best_key[1]:.4f}")
    print(f"best_worst_letter_accuracy={best_key[0]:.4f}")
    if test_acc is not None:
        print(f"test_accuracy={test_acc:.4f}")
        print(f"test_per_letter: {format_class_scores(test_per_letter)}")
    else:
        print("test split was not loaded (validation-only experiment)")
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
        "model_version": model_variant,
        "architecture_variant": "v3_mlp" if model_version == "v3" else v4_variant,
        "preprocessing_version": preprocessing_version,
        "input_dim": input_dim,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "reached_target": reached_target,
        "best_epoch": best_epoch,
        "validation_accuracy": best_key[2],
        "validation_macro_accuracy": best_key[1],
        "worst_validation_accuracy": best_key[0],
        "validation_per_letter": best_per_letter,
        "test_accuracy": test_acc,
        "test_per_letter": test_per_letter,
        "test_counts": test_counts,
        "training": {
            "loss": loss_name,
            "focal_gamma": float(focal_gamma),
            "landmark_jitter_std": float(landmark_jitter_std),
            "rotation_degrees": float(rotation_degrees),
            "focus_letters": list(focus_letters),
            "focus_weight": float(focus_weight),
            "seed": int(seed),
            "hidden_dim": int(hidden_dim),
            "embedding_dim": int(embedding_dim),
            "dropout": float(dropout),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
        },
        "checkpoint": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
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
    parser.add_argument("--loss", choices=("cross_entropy", "focal"), default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--landmark-jitter-std", type=float, default=0.0)
    parser.add_argument("--rotation-degrees", type=float, default=0.0)
    parser.add_argument("--focus-letters", nargs="*", default=[])
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--model-version", choices=("v3", "v4"), default="v3")
    parser.add_argument("--v4-variant", choices=V4_VARIANTS, default="full_v4")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if not 0 < args.target_class_accuracy <= 1:
        raise ValueError(
            "--target-class-accuracy must be > 0 and <= 1"
        )

    report = train_model(
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
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        landmark_jitter_std=args.landmark_jitter_std,
        rotation_degrees=args.rotation_degrees,
        focus_letters=tuple(letter.upper() for letter in args.focus_letters),
        focus_weight=args.focus_weight,
        validation_only=args.validation_only,
        model_version=args.model_version,
        v4_variant=args.v4_variant,
    )
    if args.output_json:
        report_path = Path(args.output_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report={report_path}")


if __name__ == "__main__":
    main()
