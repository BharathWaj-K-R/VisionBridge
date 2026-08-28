"""Small real-data CTC optimization gate for the multimodal VisionBridge model."""
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
DEFAULT_MAX_SPACE_RATIO = 0.90
DEFAULT_MIN_MEANINGFUL_UNIQUE = 2
DEFAULT_MAX_CER = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-loss-reduction", type=float, default=0.50)
    parser.add_argument("--max-space-ratio", type=float, default=DEFAULT_MAX_SPACE_RATIO)
    parser.add_argument("--min-meaningful-unique", type=int, default=DEFAULT_MIN_MEANINGFUL_UNIQUE)
    parser.add_argument("--max-mean-cer", type=float, default=DEFAULT_MAX_CER)
    return parser.parse_args()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def cer(prediction: str, target: str) -> float:
    return levenshtein(prediction.lower(), target.lower()) / max(len(target), 1)


def greedy_ctc_decode(logits: torch.Tensor, tokenizer: SimpleCharTokenizer) -> tuple[str, float, float, int]:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape [1,T,V], got {tuple(logits.shape)}")
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = probs.max(dim=-1)
    frame_ids = top_ids[0].tolist()
    frame_probs = top_probs[0].tolist()

    collapsed: list[tuple[int, float]] = []
    previous: int | None = None
    for token_id, probability in zip(frame_ids, frame_probs):
        token_id = int(token_id)
        if token_id != previous:
            collapsed.append((token_id, float(probability)))
            previous = token_id

    decoded = [item for item in collapsed if item[0] != CTC_BLANK_ID]
    tokens = [tokenizer.id_to_token[token_id] for token_id, _ in decoded]
    text = "".join(tokens)
    confidence = sum(probability for _, probability in decoded) / len(decoded) if decoded else 0.0
    space_id = tokenizer.token_to_id[" "]
    space_ratio = sum(token_id == space_id for token_id in frame_ids) / max(len(frame_ids), 1)
    unique_meaningful = len({token for token in tokens if token.strip()})
    return text, confidence, space_ratio, unique_meaningful


def target_character_peak_probability(
    logits: torch.Tensor,
    target: str,
    tokenizer: SimpleCharTokenizer,
) -> float:
    """Report whether target characters receive probability mass anywhere in time.

    This is diagnostic only. It deliberately does not replace CTC decoding or
    acceptance criteria. A high peak with a blank greedy path points toward
    alignment/decoding behavior rather than an immediately obvious dead head.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape [1,T,V], got {tuple(logits.shape)}")
    probs = torch.softmax(logits, dim=-1)[0]
    target_ids = [tokenizer.token_to_id[ch] for ch in target]
    if not target_ids:
        return 0.0
    peaks = [float(probs[:, token_id].max().item()) for token_id in target_ids]
    return sum(peaks) / len(peaks)


def frame_argmax_ratios(logits: torch.Tensor, tokenizer: SimpleCharTokenizer) -> tuple[float, float]:
    probs = torch.softmax(logits, dim=-1)
    ids = probs.argmax(dim=-1)[0]
    blank_ratio = float((ids == CTC_BLANK_ID).float().mean().item())
    space_ratio = float((ids == tokenizer.token_to_id[" "]).float().mean().item())
    return blank_ratio, space_ratio


def semantic_gate_failures(
    target: str,
    prediction: str,
    space_ratio: float,
    unique_meaningful: int,
    *,
    max_space_ratio: float = DEFAULT_MAX_SPACE_RATIO,
    min_meaningful_unique: int = DEFAULT_MIN_MEANINGFUL_UNIQUE,
    max_cer: float = DEFAULT_MAX_CER,
) -> list[str]:
    failures: list[str] = []
    if not prediction.strip():
        failures.append("prediction is empty/whitespace-only")
    if space_ratio >= max_space_ratio:
        failures.append(f"space collapse {space_ratio:.3f} >= {max_space_ratio:.3f}")
    if unique_meaningful < min_meaningful_unique:
        failures.append(
            f"meaningful token diversity {unique_meaningful} < {min_meaningful_unique}"
        )
    sample_cer = cer(prediction, target)
    if sample_cer > max_cer:
        failures.append(f"CER {sample_cer:.3f} > {max_cer:.3f}")
    return failures


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = SimpleCharTokenizer()
    dataset = ISLTranslateKeypointDataset(args.data_dir, tokenizer=tokenizer)
    n = min(max(1, args.samples), len(dataset))

    # Prefer distinct short labels. Short labels are a cleaner optimization
    # probe and let us distinguish training-path failures from huge-target CTC difficulty.
    ordered = sorted(
        range(len(dataset)),
        key=lambda index: (len(dataset.examples[index].text), dataset.examples[index].uid),
    )
    chosen: list[int] = []
    seen_texts: set[str] = set()
    for index in ordered:
        text = dataset.examples[index].text
        if text not in seen_texts:
            chosen.append(index)
            seen_texts.add(text)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise RuntimeError("Insufficient distinct real labels for semantic overfit test")

    loader = DataLoader(
        Subset(dataset, chosen),
        batch_size=n,
        shuffle=False,
        collate_fn=collate_ctc_batch,
    )
    batch = next(iter(loader))

    pose = batch["pose"].to(device)
    face = batch["face"].to(device)
    left_hand = batch["left_hand"].to(device)
    right_hand = batch["right_hand"].to(device)
    labels = batch["labels"].to(device)
    input_lengths = batch["input_lengths"].to(device)
    label_lengths = batch["label_lengths"].to(device)

    model = VisionBridgeBaseModel(vocab_size=tokenizer.vocab_size, use_hands=True).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise RuntimeError("OVERFIT SANITY FAILED: scratch training model has zero trainable parameters.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    loss_fn = torch.nn.CTCLoss(blank=CTC_BLANK_ID, zero_infinity=True)

    def compute_loss() -> tuple[torch.Tensor, torch.Tensor]:
        logits = model(
            pose,
            face,
            left_hand,
            right_hand,
            input_lengths,
        )
        loss = loss_fn(
            torch.log_softmax(logits, dim=-1).transpose(0, 1),
            labels,
            input_lengths,
            label_lengths,
        )
        return logits, loss

    model.eval()
    with torch.inference_mode():
        initial_logits, initial_loss_tensor = compute_loss()
    initial_loss = float(initial_loss_tensor.item())

    print(f"Device: {device}")
    print(f"Sanity samples: {n}")
    print(f"Trainable parameters: {trainable:,}")
    print("Targets:")
    for i, text in enumerate(batch["text"]):
        print(f"  {i}: {text!r}")
    print("Initial frame argmax diagnostics:")
    for i, text in enumerate(batch["text"]):
        blank_ratio, space_ratio = frame_argmax_ratios(
            initial_logits[i : i + 1, : int(input_lengths[i].item())], tokenizer
        )
        peak = target_character_peak_probability(
            initial_logits[i : i + 1, : int(input_lengths[i].item())], text, tokenizer
        )
        print(
            f"  {i}: blank_ratio={blank_ratio:.3f} space_ratio={space_ratio:.3f} "
            f"target_char_peak_mean={peak:.3f}"
        )

    first_gradient_norms: dict[str, float] = {}
    first_parameter_delta = 0.0
    model.train()
    for step in range(1, args.steps + 1):
        _, loss = compute_loss()
        if not torch.isfinite(loss):
            raise RuntimeError(f"OVERFIT SANITY FAILED: non-finite loss at step {step}.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    first_gradient_norms[name] = float(parameter.grad.detach().norm().item())
            before = next(parameter for parameter in model.parameters() if parameter.requires_grad).detach().clone()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step == 1:
            after = next(parameter for parameter in model.parameters() if parameter.requires_grad).detach()
            first_parameter_delta = float((after - before).norm().item())
            if not first_gradient_norms:
                raise RuntimeError("OVERFIT SANITY FAILED: no trainable parameter received a gradient.")
        if step == 1 or step % max(1, args.steps // 10) == 0:
            print(f"step={step} loss={float(loss.item()):.4f}")

    if not first_gradient_norms or first_parameter_delta <= 0.0:
        raise RuntimeError("OVERFIT SANITY FAILED: optimizer did not update trainable parameters.")
    top_gradients = sorted(first_gradient_norms.items(), key=lambda item: item[1], reverse=True)[:5]
    print("First-step gradient diagnostics:")
    for name, norm in top_gradients:
        print(f"  {name}: grad_norm={norm:.6g}")
    print(f"First-step parameter delta norm: {first_parameter_delta:.6g}")

    model.eval()
    with torch.inference_mode():
        final_logits, final_loss_tensor = compute_loss()
    final_loss = float(final_loss_tensor.item())
    loss_reduction = 1.0 - (final_loss / initial_loss) if initial_loss else 0.0

    metrics: list[dict[str, object]] = []
    print("\nPredictions on overfit samples:")
    for i, target in enumerate(batch["text"]):
        frames = int(input_lengths[i].item())
        sample_logits = final_logits[i : i + 1, :frames]
        prediction, confidence, space_ratio, unique_meaningful = greedy_ctc_decode(
            sample_logits, tokenizer
        )
        blank_ratio, frame_space_ratio = frame_argmax_ratios(sample_logits, tokenizer)
        sample_cer = cer(prediction, target)
        target_peak = target_character_peak_probability(sample_logits, target, tokenizer)
        metrics.append(
            {
                "target": target,
                "prediction": prediction,
                "confidence": confidence,
                "blank_ratio": blank_ratio,
                "space_ratio": space_ratio,
                "frame_space_ratio": frame_space_ratio,
                "target_char_peak_mean": target_peak,
                "unique_meaningful": unique_meaningful,
                "cer": sample_cer,
            }
        )
        print(
            f"  {i}: truth={target!r} predicted={prediction!r} "
            f"confidence={confidence:.3f} cer={sample_cer:.3f} "
            f"blank_ratio={blank_ratio:.3f} space_ratio={space_ratio:.3f} "
            f"target_char_peak_mean={target_peak:.3f} "
            f"unique_meaningful={unique_meaningful}"
        )

    mean_cer = sum(float(m["cer"]) for m in metrics) / len(metrics)
    max_space_ratio = max(float(m["space_ratio"]) for m in metrics)
    min_unique_meaningful = min(int(m["unique_meaningful"]) for m in metrics)

    print("\n" + "=" * 64)
    print(f"Initial CTC loss:       {initial_loss:.4f}")
    print(f"Final CTC loss:         {final_loss:.4f}")
    print(f"Loss reduction:         {loss_reduction * 100:.1f}%")
    print(f"Mean CER:               {mean_cer:.4f}")
    print(f"Max space ratio:        {max_space_ratio:.4f}")
    print(f"Min meaningful tokens:  {min_unique_meaningful}")
    print(f"Trainable parameters:   {trainable:,}")
    print("=" * 64)

    failures: list[str] = []
    if loss_reduction < args.min_loss_reduction:
        failures.append(
            f"loss reduction {loss_reduction:.3f} < {args.min_loss_reduction:.3f}"
        )
    for metric in metrics:
        failures.extend(
            semantic_gate_failures(
                str(metric["target"]),
                str(metric["prediction"]),
                float(metric["space_ratio"]),
                int(metric["unique_meaningful"]),
                max_space_ratio=args.max_space_ratio,
                min_meaningful_unique=args.min_meaningful_unique,
                max_cer=args.max_mean_cer,
            )
        )

    if failures:
        raise RuntimeError(
            "OVERFIT SANITY FAILED: "
            + "; ".join(dict.fromkeys(failures))
            + ". Do not run full training or push a checkpoint."
        )

    print("OVERFIT SANITY: PASS")


if __name__ == "__main__":
    main()
