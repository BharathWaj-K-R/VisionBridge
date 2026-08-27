"""Train VisionBridgeBaseModel on preprocessed ISLTranslate keypoints."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from app.models.base_model import VisionBridgeBaseModel
from app.training.isltranslate import (
    ISLTranslateKeypointDataset,
    SimpleCharTokenizer,
    collate_ctc_batch,
)


def run_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    max_grad_norm: float,
    train: bool,
    verify_gradients: bool = False,
) -> float:
    """Run one epoch and optionally fail fast when gradients are absent."""
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0
    gradient_checked = False

    for batch in loader:
        pose = batch["pose"].to(device, non_blocking=device.type == "cuda")
        face = batch["face"].to(device, non_blocking=device.type == "cuda")
        labels = batch["labels"].to(device, non_blocking=device.type == "cuda")
        input_lengths = batch["input_lengths"].to(device, non_blocking=device.type == "cuda")
        label_lengths = batch["label_lengths"].to(device, non_blocking=device.type == "cuda")

        with torch.set_grad_enabled(train):
            logits = model(pose, face, input_lengths)
            log_probs = torch.nn.functional.log_softmax(
                logits,
                dim=-1,
            ).transpose(0, 1)
            loss = loss_fn(
                log_probs,
                labels,
                input_lengths,
                label_lengths,
            )

            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite CTC loss.")

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if verify_gradients and not gradient_checked:
                    trainable = [
                        (name, parameter)
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    ]
                    if not trainable:
                        raise RuntimeError(
                            "TRAINING BLOCKED: optimizer model has zero trainable parameters."
                        )
                    missing = [
                        name
                        for name, parameter in trainable
                        if parameter.grad is None
                    ]
                    zero = [
                        name
                        for name, parameter in trainable
                        if parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                    ]
                    if missing:
                        raise RuntimeError(
                            "TRAINING BLOCKED: trainable parameters received no gradient: "
                            + ", ".join(missing[:8])
                        )
                    if zero:
                        raise RuntimeError(
                            "TRAINING BLOCKED: non-finite gradients detected: "
                            + ", ".join(zero[:8])
                        )
                    gradient_checked = True

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )
                optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tokenizer = SimpleCharTokenizer()
    dataset = ISLTranslateKeypointDataset(
        args.data_dir,
        tokenizer=tokenizer,
    )

    val_size = (
        max(1, int(len(dataset) * args.val_fraction))
        if len(dataset) > 1
        else 0
    )
    train_size = len(dataset) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)

    train_dataset, val_dataset = (
        random_split(
            dataset,
            [train_size, val_size],
            generator=split_generator,
        )
        if val_size
        else (dataset, None)
    )

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    pin_memory = device.type == "cuda"
    worker_count = max(0, args.num_workers)

    loader_kwargs = {
        "collate_fn": collate_ctc_batch,
        "num_workers": worker_count,
        "pin_memory": pin_memory,
    }
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    model = VisionBridgeBaseModel(
        vocab_size=tokenizer.vocab_size
    ).to(device)

    trainable_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_params
    )
    if trainable_count <= 0:
        raise RuntimeError(
            "TRAINING BLOCKED: model initialized with zero trainable parameters."
        )

    print(
        f"device={device} total_params={sum(p.numel() for p in model.parameters()):,} "
        f"trainable_params={trainable_count:,} dataset={len(dataset)} "
        f"train={len(train_dataset)} val={len(val_dataset) if val_dataset is not None else 0}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.CTCLoss(
        blank=0,
        zero_infinity=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    vocabulary_path = output_path.with_suffix(".vocab.json")

    # Never let an old checkpoint masquerade as the result of a failed/new run.
    if not args.resume:
        output_path.unlink(missing_ok=True)
        vocabulary_path.unlink(missing_ok=True)

    best_val_loss = float("inf")
    start_epoch = 1

    checkpoint_path = (
        Path(args.checkpoint_dir) / "latest.pt"
        if args.checkpoint_dir
        else None
    )

    if args.resume and checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        print(
            f"resumed from {checkpoint_path}: "
            f"starting at epoch={start_epoch}, "
            f"best_val_loss={best_val_loss:.4f}"
        )
    elif args.resume:
        print(
            f"--resume passed but no checkpoint found at "
            f"{checkpoint_path} — starting from epoch 1"
        )

    if start_epoch > args.epochs:
        raise RuntimeError(
            "Resume checkpoint is already at/after requested epoch count; "
            "increase --epochs or omit --resume."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.max_grad_norm,
            train=True,
            verify_gradients=epoch == start_epoch,
        )

        if val_loader is not None:
            val_loss = run_epoch(
                model,
                val_loader,
                optimizer,
                loss_fn,
                device,
                args.max_grad_norm,
                train=False,
            )
            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    output_path,
                )
                print(
                    f"  -> new best (val_loss={val_loss:.4f}), "
                    f"saved checkpoint to {output_path}"
                )
        else:
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                "(no validation split — dataset too small)"
            )
            torch.save(
                model.state_dict(),
                output_path,
            )

        if checkpoint_path:
            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    if not output_path.exists():
        raise RuntimeError(
            "Training completed without producing a model checkpoint."
        )

    tokenizer.save(vocabulary_path)
    print(
        f"training done. best_val_loss="
        f"{best_val_loss if val_loader else 'n/a'}. "
        f"weights at {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/processed/isltranslate",
    )
    parser.add_argument(
        "--output",
        default="backend/app/models/weights/base_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers; 0 disables multiprocessing.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="if set, saves a full resumable checkpoint (model+optimizer+epoch) "
        "to <checkpoint-dir>/latest.pt after every epoch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from <checkpoint-dir>/latest.pt if it exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
