"""Evaluate the trained base model against real ISLTranslate data using
Character Error Rate (CER) — the standard metric for character-level
transcription tasks, since SimpleCharTokenizer is char-level (not word-level).

Uses the exact live decoder and the vocabulary saved beside the checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.models.base_model import load_frozen_base_model
from app.services import inference_service
from app.training.isltranslate import (
    ISLTranslateKeypointDataset,
    SimpleCharTokenizer,
    _downsample_to_max_length,
)


def levenshtein(a: str, b: str) -> int:
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
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]


def cer(predicted: str, ground_truth: str) -> float:
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    return levenshtein(predicted.lower(), ground_truth.lower()) / len(ground_truth)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--worst-n", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = Path(args.weights).with_suffix(".vocab.json")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    tokenizer = SimpleCharTokenizer.load(vocab_path)
    dataset = ISLTranslateKeypointDataset(args.data_dir, tokenizer=tokenizer)
    n = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    if n <= 0:
        raise ValueError("Evaluation dataset contains zero usable samples")

    print(f"Evaluating {n} / {len(dataset)} samples from {args.data_dir}")

    model = load_frozen_base_model(args.weights, vocab_size=tokenizer.vocab_size).to(device)
    loaded_vocab = inference_service._load_vocab(str(args.weights))
    if len(loaded_vocab) != model.output_head.out_features:
        raise ValueError(
            "Checkpoint/vocabulary mismatch: "
            f"checkpoint output_head={model.output_head.out_features}, "
            f"vocab_size={len(loaded_vocab)}"
        )
    inference_service._id_to_token = loaded_vocab

    results: list[dict] = []
    empty_predictions = 0

    with torch.no_grad():
        for i in range(n):
            item = dataset[i]
            pose, face = _downsample_to_max_length(item["pose"], item["face"], item["uid"])
            pose = pose.unsqueeze(0).to(device)
            face = face.unsqueeze(0).to(device)

            logits = model(pose, face)
            predicted_text, confidence = inference_service.decode_logits(logits)
            ground_truth = item["text"]
            is_empty_prediction = predicted_text == "(no sign detected)"
            # Score against the actual decoded text (empty string), not the
            # human-readable placeholder — scoring the literal 19-character
            # placeholder string against a short ground truth produces a
            # misleadingly huge CER (e.g. 8.5) instead of the correct 1.0
            # for "predicted nothing."
            score = cer("" if is_empty_prediction else predicted_text, ground_truth)

            if is_empty_prediction:
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

    print("\n" + "=" * 60)
    print(f"Samples evaluated:     {n}")
    print(f"Mean CER:              {mean_cer:.4f}  (0.0 = perfect, lower is better)")
    print(f"Median CER:            {sorted(cers)[len(cers) // 2]:.4f}")
    print(f"Exact matches:         {exact_matches} / {n} ({100 * exact_matches / n:.1f}%)")
    print(f"Empty predictions:     {empty_predictions} / {n} ({100 * empty_predictions / n:.1f}%)")
    print("=" * 60)

    worst = sorted(results, key=lambda r: r["cer"], reverse=True)[: args.worst_n]
    print(f"\nWorst {len(worst)} predictions:")
    for r in worst:
        print(f"  uid={r['uid']}  cer={r['cer']:.3f}  conf={r['confidence']:.3f}")
        print(f"    ground truth: {r['ground_truth']!r}")
        print(f"    predicted:    {r['predicted']!r}")


if __name__ == "__main__":
    main()
