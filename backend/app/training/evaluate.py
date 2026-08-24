"""Evaluate the trained base model against real ISLTranslate data using
Character Error Rate (CER) — the standard metric for character-level
transcription tasks, since SimpleCharTokenizer is char-level (not word-level).

CER = edit_distance(predicted, ground_truth) / len(ground_truth)
  0.0  = perfect match
  1.0  = as bad as predicting nothing (roughly — can exceed 1.0 for
         predictions much longer than the ground truth)

Uses the exact same decode_logits() as backend/app/services/inference_service.py
(imported directly, not reimplemented) — so the numbers this script reports
match what the live /translate endpoint would actually produce, not a
separately-maintained approximation of it.

Usage:
    PYTHONPATH=backend python -m app.training.evaluate \\
        --data-dir data/processed/isltranslate \\
        --weights backend/app/models/weights/base_model.pt \\
        --max-samples 200 \\
        --worst-n 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.models.base_model import load_frozen_base_model
from app.services.inference_service import decode_logits
from app.training.isltranslate import ISLTranslateKeypointDataset, SimpleCharTokenizer, _downsample_to_max_length


def levenshtein(a: str, b: str) -> int:
    """Standard edit distance (insertions + deletions + substitutions),
    pure Python — fine for the short strings involved here (sentence-level
    text, not documents)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1]


def cer(predicted: str, ground_truth: str) -> float:
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    return levenshtein(predicted.lower(), ground_truth.lower()) / len(ground_truth)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="e.g. data/processed/isltranslate")
    parser.add_argument("--weights", required=True, help="path to base_model.pt")
    parser.add_argument("--max-samples", type=int, default=None, help="cap evaluation to N samples (default: all)")
    parser.add_argument("--worst-n", type=int, default=10, help="how many worst predictions to print")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = Path(args.weights).with_suffix(".vocab.json")
    tokenizer = SimpleCharTokenizer.load(vocab_path) if vocab_path.exists() else SimpleCharTokenizer()

    dataset = ISLTranslateKeypointDataset(args.data_dir, tokenizer=tokenizer)
    n = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    print(f"Evaluating {n} / {len(dataset)} samples from {args.data_dir}")

    model = load_frozen_base_model(args.weights, vocab_size=tokenizer.vocab_size).to(device)

    results: list[dict] = []
    empty_predictions = 0

    with torch.no_grad():
        for i in range(n):
            item = dataset[i]
            pose, face = _downsample_to_max_length(item["pose"], item["face"], item["uid"])
            pose = pose.unsqueeze(0).to(device)
            face = face.unsqueeze(0).to(device)

            logits = model(pose, face)
            predicted_text, confidence = decode_logits(logits)
            ground_truth = item["text"]
            score = cer(predicted_text, ground_truth)

            if predicted_text == "(no sign detected)":
                empty_predictions += 1

            results.append({
                "uid": item["uid"],
                "ground_truth": ground_truth,
                "predicted": predicted_text,
                "confidence": confidence,
                "cer": score,
            })

            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{n}")

    cers = [r["cer"] for r in results]
    mean_cer = sum(cers) / len(cers)
    exact_matches = sum(1 for r in results if r["predicted"].lower() == r["ground_truth"].lower())

    print(f"\n{'=' * 60}")
    print(f"Samples evaluated:     {n}")
    print(f"Mean CER:              {mean_cer:.4f}  (0.0 = perfect, lower is better)")
    print(f"Median CER:            {sorted(cers)[len(cers) // 2]:.4f}")
    print(f"Exact matches:         {exact_matches} / {n} ({100 * exact_matches / n:.1f}%)")
    print(f"Empty predictions:     {empty_predictions} / {n} ({100 * empty_predictions / n:.1f}%) — '(no sign detected)'")
    print(f"{'=' * 60}")

    worst = sorted(results, key=lambda r: r["cer"], reverse=True)[: args.worst_n]
    print(f"\nWorst {len(worst)} predictions:")
    for r in worst:
        print(f"  uid={r['uid']}  cer={r['cer']:.3f}  conf={r['confidence']:.3f}")
        print(f"    ground truth: {r['ground_truth']!r}")
        print(f"    predicted:    {r['predicted']!r}")


if __name__ == "__main__":
    main()
