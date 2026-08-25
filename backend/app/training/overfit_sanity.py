"""Tiny real-data overfit check for the VisionBridge CTC training path.

Use this before a full retraining run. The default gate deliberately trains on
ONE of the shortest real sentence labels for a longer run so the test answers
the narrow question: can the current model/CTC pipeline memorize one real
example at all?

Example:

PYTHONPATH=backend python -m app.training.overfit_sanity \
  --data-dir data/processed/isltranslate \
  --samples 1 \
  --steps 2000

This is not a model-quality benchmark. A PASS means the optimization path can
escape the all-blank solution on a real example. Full training quality is
still evaluated by the training notebook's validation predictions/CER gate.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from app.models.base_model import VisionBridgeBaseModel
from app.training.isltranslate import (
    ISLTranslateKeypointDataset,
    SimpleCharTokenizer,
    collate_ctc_batch,
)

CTC_BLANK_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def greedy_ctc_decode(
    logits: torch.Tensor,
    tokenizer: SimpleCharTokenizer,
) -> tuple[str, float, int, int, int]:
    """Decode one sample and return text, confidence, frame counts, and tokens."""
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape [1, T, V], got {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = probs.max(dim=-1)

    frame_ids = top_ids[0].tolist()
    frame_probs = top_probs[0].tolist()

    collapsed_ids: list[tuple[int, float]] = []
    previous_id: int | None = None
    for token_id, probability in zip(frame_ids, frame_probs):
        token_id = int(token_id)
        if token_id != previous_id:
            collapsed_ids.append((token_id, float(probability)))
            previous_id = token_id

    non_blank_collapsed = [
        (token_id, probability)
        for token_id, probability in collapsed_ids
        if token_id != CTC_BLANK_ID
    ]

    tokens = [
        tokenizer.id_to_token[token_id]
        if 0 <= token_id < len(tokenizer.id_to_token)
        else f"<{token_id}>"
        for token_id, _ in non_blank_collapsed
    ]
    text = "".join(tokens) or "(no sign detected)"
    confidence = (
        sum(probability for _, probability in non_blank_collapsed)
        / len(non_blank_collapsed)
        if non_blank_collapsed
        else 0.0
    )
    frame_non_blank = sum(token_id != CTC_BLANK_ID for token_id in frame_ids)
    frame_total = len(frame_ids)
    unique_non_blank = len({token_id for token_id, _ in non_blank_collapsed})
    return text, confidence, frame_non_blank, frame_total, unique_non_blank


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = SimpleCharTokenizer()
    dataset = ISLTranslateKeypointDataset(args.data_dir, tokenizer=tokenizer)
    n = min(max(args.samples, 1), len(dataset))

    # For the default single-sample gate, deliberately choose the shortest
    # real label available. This removes unnecessary sequence-to-label
    # difficulty from the architectural/optimization sanity check.
    indexed = list(range(len(dataset)))
    indexed.sort(key=lambda idx: len(dataset.examples[idx].text))
    chosen_indices = indexed[:n]

    subset = Subset(dataset, chosen_indices)
    loader = DataLoader(subset, batch_size=n, shuffle=False, collate_fn=collate_ctc_batch)
    batch = next(iter(loader))

    model = VisionBridgeBaseModel(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CTCLoss(blank=CTC_BLANK_ID, zero_infinity=True)

    pose = batch["pose"].to(device)
    face = batch["face"].to(device)
    labels = batch["labels"].to(device)
    input_lengths = batch["input_lengths"].to(device)
    label_lengths = batch["label_lengths"].to(device)

    print(f"Sanity samples: {n} (shortest real labels)")
    print("Targets:")
    for i, text in enumerate(batch["text"]):
        print(f"  {i}: {text!r}")

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
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % max(1, args.steps // 10) == 0:
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
    decoded_non_blank = 0
    print("\nPredictions on overfit samples:")
    for i in range(n):
        valid_frames = int(input_lengths[i].item())
        sample_logits = final_logits[i : i + 1, :valid_frames]
        text, confidence, sample_non_blank, sample_total, unique_non_blank = greedy_ctc_decode(
            sample_logits,
            tokenizer,
        )
        target = batch["text"][i]
        decoded_non_blank += unique_non_blank
        non_blank += sample_non_blank
        total += sample_total
        print(
            f"  {i}: truth={target!r} predicted={text!r} "
            f"confidence={confidence:.3f} frame_non_blank={sample_non_blank} "
            f"unique_non_blank={unique_non_blank}"
        )

    frame_blank_ratio = 1.0 - (non_blank / max(total, 1))
    loss_reduction = 0.0 if initial_loss == 0 else 1.0 - (final_loss / initial_loss)

    print("\n" + "=" * 60)
    print(f"Device:              {device}")
    print(f"Samples:             {n}")
    print(f"Initial CTC loss:    {initial_loss:.4f}")
    print(f"Final CTC loss:      {final_loss:.4f}")
    print(f"Loss reduction:      {loss_reduction * 100:.1f}%")
    print(f"Frame blank ratio:   {frame_blank_ratio:.4f}")
    print(f"Frame non-blank:     {non_blank}/{total}")
    print(f"Unique decoded IDs:  {decoded_non_blank}")
    print("=" * 60)

    if not torch.isfinite(torch.tensor(final_loss)):
        raise RuntimeError("OVERFIT SANITY FAILED: final loss is not finite.")
    if loss_reduction < 0.20:
        raise RuntimeError(
            "OVERFIT SANITY FAILED: loss did not fall by at least 20%. "
            "Fix the training/data path before full retraining."
        )
    if decoded_non_blank == 0:
        raise RuntimeError(
            "OVERFIT SANITY FAILED: even the shortest real sample still decodes to no non-blank tokens. "
            "Do not start the full training run yet."
        )
    if decoded_non_blank > 0 and non_blank == 0:
        raise RuntimeError(
            "OVERFIT SANITY FAILED: decoder/blank statistics are inconsistent."
        )

    print("OVERFIT SANITY: PASS")


if __name__ == "__main__":
    main()
