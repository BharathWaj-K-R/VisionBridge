"""Tiny real-data overfit check for the VisionBridge CTC training path.

Use this before a full retraining run:

PYTHONPATH=backend python -m app.training.overfit_sanity \
  --data-dir data/processed/isltranslate \
  --samples 4 \
  --steps 150

A healthy training pipeline should drive the loss down on the same few real
examples and should produce at least some non-blank predictions. This is not a
model-quality benchmark; it is a plumbing/optimization sanity check.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from app.models.base_model import VisionBridgeBaseModel
from app.services.inference_service import decode_logits
from app.training.isltranslate import (
    ISLTranslateKeypointDataset,
    SimpleCharTokenizer,
    collate_ctc_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = SimpleCharTokenizer()
    dataset = ISLTranslateKeypointDataset(args.data_dir, tokenizer=tokenizer)
    n = min(max(args.samples, 2), len(dataset))
    subset = Subset(dataset, list(range(n)))
    loader = DataLoader(subset, batch_size=n, shuffle=False, collate_fn=collate_ctc_batch)
    batch = next(iter(loader))

    model = VisionBridgeBaseModel(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    pose = batch["pose"].to(device)
    face = batch["face"].to(device)
    labels = batch["labels"].to(device)
    input_lengths = batch["input_lengths"].to(device)
    label_lengths = batch["label_lengths"].to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(pose, face, input_lengths)
        initial_loss = float(
            loss_fn(
                torch.log_softmax(logits, dim=-1).transpose(0, 1),
                labels,
                input_lengths,
                label_lengths,
            ).item()
        )

    model.train()
    for step in range(1, args.steps + 1):
        logits = model(pose, face, input_lengths)
        loss = loss_fn(
            torch.log_softmax(logits, dim=-1).transpose(0, 1),
            labels,
            input_lengths,
            label_lengths,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % max(1, args.steps // 5) == 0:
            print(f"step={step} loss={float(loss.item()):.4f}")

    model.eval()
    with torch.inference_mode():
        final_logits = model(pose, face, input_lengths)
        final_loss = float(
            loss_fn(
                torch.log_softmax(final_logits, dim=-1).transpose(0, 1),
                labels,
                input_lengths,
                label_lengths,
            ).item()
        )

    non_blank = 0
    total = 0
    print("\nPredictions on overfit samples:")
    for i in range(n):
        text, confidence = decode_logits(final_logits[i : i + 1])
        target = batch["text"][i]
        frame_ids = final_logits[i].argmax(dim=-1)
        valid_ids = frame_ids[: int(input_lengths[i].item())]
        sample_non_blank = int((valid_ids != 0).sum().item())
        non_blank += sample_non_blank
        total += int(valid_ids.numel())
        print(f"  {i}: truth={target!r} predicted={text!r} confidence={confidence:.3f}")

    blank_ratio = 1.0 - (non_blank / max(total, 1))
    loss_reduction = 0.0 if initial_loss == 0 else 1.0 - (final_loss / initial_loss)

    print("\n" + "=" * 60)
    print(f"Device:              {device}")
    print(f"Samples:             {n}")
    print(f"Initial CTC loss:    {initial_loss:.4f}")
    print(f"Final CTC loss:      {final_loss:.4f}")
    print(f"Loss reduction:      {loss_reduction * 100:.1f}%")
    print(f"Final blank ratio:   {blank_ratio:.4f}")
    print("=" * 60)

    if not torch.isfinite(torch.tensor(final_loss)):
        raise RuntimeError("OVERFIT SANITY FAILED: final loss is not finite.")
    if loss_reduction < 0.20:
        raise RuntimeError(
            "OVERFIT SANITY FAILED: loss did not fall by at least 20%. "
            "Fix the training/data path before full retraining."
        )
    if blank_ratio >= 0.99:
        raise RuntimeError(
            "OVERFIT SANITY FAILED: model still predicts effectively all CTC blanks."
        )

    print("OVERFIT SANITY: PASS")


if __name__ == "__main__":
    main()
