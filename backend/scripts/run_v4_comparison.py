"""Validation-only multi-seed comparison of V3 and V4 candidates.

The final test split is evaluated exactly once, only after a validation-selected
candidate and checkpoint have been frozen. This script writes local reports, not
dataset or checkpoint files to Git.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from app.models.letter_model import (
    EMBEDDING_DIM,
    INPUT_DIM,
    LETTER_LABELS,
    load_model_checkpoint,
)
from app.training.letter_base import load_split, train_model
from app.training.evaluate_letter_base import evaluate

SPLITS = ("train", "val", "test")
DEFAULT_SEEDS = (42, 123, 314)
DEFAULT_MARKDOWN = Path(__file__).resolve().parents[2] / "docs" / "experiments" / "v4_comparison.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_rows(root: Path, split: str) -> list[dict]:
    path = root / f"{split}_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {split} manifest: {path}; aligned V1/V4 split identities are required"
        )
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            status = row.get("landmark_status")
            if status is not None and status != "ok":
                continue
            label = row.get("label", row.get("letter"))
            if label not in LETTER_LABELS:
                raise ValueError(f"{path}:{line_number}: invalid A-Z label")
            if not row.get("source_path") or not row.get("image_sha256"):
                raise ValueError(f"{path}:{line_number}: source_path and image_sha256 are required")
            rows.append(row)
    return rows


def _identity(row: dict) -> tuple:
    return (
        row.get("sample_id"),
        row.get("source_path"),
        row.get("image_sha256"),
        row.get("label", row.get("letter")),
    )


def verify_aligned_splits(v1_dir: Path, v4_dir: Path) -> dict[str, dict[str, str | int]]:
    """Fail closed unless arrays have identical ordered sample identities."""
    evidence = {}
    for split in SPLITS:
        v1_manifest = v1_dir / f"{split}_manifest.jsonl"
        v4_manifest = v4_dir / f"{split}_manifest.jsonl"
        v1_rows = _manifest_rows(v1_dir, split)
        v4_rows = _manifest_rows(v4_dir, split)
        if [_identity(row) for row in v1_rows] != [_identity(row) for row in v4_rows]:
            raise ValueError(f"{split}: V1 and V4 manifest sample identities/order differ")

        v1_x, v1_y = load_split(v1_dir, split, input_dim=INPUT_DIM)
        v4_x, v4_y = load_split(v4_dir, split, input_dim=141)
        expected_labels = torch.tensor(
            [LETTER_LABELS.index(row.get("label", row.get("letter"))) for row in v1_rows],
            dtype=torch.long,
        )
        if len(v1_rows) != len(v1_x) or len(v4_rows) != len(v4_x):
            raise ValueError(f"{split}: manifest and feature-array sample counts differ")
        if not torch.equal(v1_y, v4_y) or not torch.equal(v1_y, expected_labels):
            raise ValueError(f"{split}: V1/V4 labels or row order differ")
        if v1_x.shape[1] != INPUT_DIM or v4_x.shape[1] != 141:
            raise ValueError(f"{split}: unexpected V1/V4 input dimensions")

        evidence[split] = {
            "samples": len(v1_rows),
            "v1_manifest_sha256": sha256_file(v1_manifest),
            "v4_manifest_sha256": sha256_file(v4_manifest),
        }
    return evidence


def measure_python_latency(checkpoint: Path, data_dir: Path, batch_size: int) -> dict:
    model = load_model_checkpoint(checkpoint)
    model.eval()
    x, _ = load_split(data_dir, "val", input_dim=model.input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = x[: min(len(x), batch_size)].to(device)
    if not len(batch):
        raise ValueError("Cannot benchmark an empty validation split")
    timings = []
    with torch.inference_mode():
        for _ in range(5):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(20):
            start = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    mean_batch_ms = float(np.mean(timings))
    return {
        "device": str(device),
        "batch_size": len(batch),
        "warmup_runs": 5,
        "measured_runs": 20,
        "mean_batch_ms": mean_batch_ms,
        "mean_sample_ms": mean_batch_ms / len(batch),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def _format_pm(result: dict, percent: bool = True) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{result['mean'] * scale:.3f} ± {result['std'] * scale:.3f}{suffix}"


def _markdown(report: dict) -> str:
    selected = report["selection"]["variant"]
    test = report["final_test"]
    lines = [
        "# V3 / V4 model and feature comparison",
        "",
        f"Run status: {report['status']}",
        f"Selection rule: {report['selection']['rule']}",
        f"Seeds: {', '.join(str(seed) for seed in report['protocol']['seeds'])}",
        f"Epoch budget: {report['protocol']['epochs']} per seed and candidate",
        f"Loss: {report['protocol']['loss']}",
        "",
        "V3 and V4 used identical ordered train/validation/test sample identities and labels. Candidate training read train and validation only. The test split was opened for accuracy exactly once, after selecting the candidate and freezing its checkpoint SHA-256.",
        "",
        "| Candidate | Features | Parameters | Validation overall (mean ± std) | Validation macro (mean ± std) | Weakest-class validation (mean ± std) | Python batch latency, ms (mean ± std) | Final test overall | Final test macro |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in report["candidates"].items():
        latency = row["python_latency_ms"]
        if variant == selected:
            test_overall = f"{test['overall_accuracy']:.6%}"
            test_macro = f"{test['macro_accuracy']:.6%}"
        else:
            test_overall = "Not evaluated"
            test_macro = "Not evaluated"
        lines.append(
            "| {name} | {features} | {parameters} | {overall} | {macro} | {weakest} | {latency} | {test_overall} | {test_macro} |".format(
                name=variant,
                features=row["feature_contract"],
                parameters=row["parameter_count"],
                overall=_format_pm(row["validation"]["overall_accuracy"]),
                macro=_format_pm(row["validation"]["macro_accuracy"]),
                weakest=_format_pm(row["validation"]["worst_class_accuracy"]),
                latency=_format_pm(latency, percent=False),
                test_overall=test_overall,
                test_macro=test_macro,
            )
        )
    lines.extend([
        "",
        "## Final candidate evidence",
        "",
        f"- Selected variant: {selected}",
        f"- Selected seed: {report['selection']['seed']}",
        f"- Checkpoint SHA-256: {test['checkpoint_sha256']}",
        f"- Test samples: {test['samples']}",
        f"- Test worst class: {test['worst_letter']} ({test['worst_letter_accuracy']:.6%})",
        f"- Device: {test['device']}",
        "- Browser latency: NOT MEASURED",
        "- Signer-independent evaluation: BLOCKED until verified signer IDs and a fresh consented held-out signer cohort exist.",
        "",
        "The complete per-seed validation evidence, manifest hashes, confusion matrix, and latency protocol are in the machine-readable report generated beside this document.",
        "",
    ])
    return "\n".join(lines)


def run_comparison(
    *,
    v1_dir: Path,
    v4_dir: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    markdown_output: Path,
) -> dict:
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("At least three distinct seeds are required")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    aligned = verify_aligned_splits(v1_dir, v4_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = (
        ("v3_mlp", "v3", None, v1_dir, "two-hand-wrist-scale-v1"),
        ("v4_features_old_mlp", "v4", "v4_features_old_mlp", v4_dir, "two-hand-geometry-v2"),
        ("v1_features_shared_encoder", "v4", "v1_features_shared_encoder", v1_dir, "two-hand-wrist-scale-v1"),
        ("full_v4", "v4", "full_v4", v4_dir, "two-hand-geometry-v2"),
    )
    per_run: dict[str, list[dict]] = {}
    for name, version, variant, data_dir, feature_contract in candidates:
        per_run[name] = []
        for seed in seeds:
            checkpoint = output_dir / "candidates" / f"{name}-seed-{seed}.pt"
            report = train_model(
                root=data_dir,
                output_path=checkpoint,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                target_class_accuracy=1.0,
                seed=seed,
                hidden_dim=128,
                embedding_dim=EMBEDDING_DIM,
                dropout=0.10,
                loss_name="cross_entropy",
                focal_gamma=1.5,
                landmark_jitter_std=0.0,
                rotation_degrees=0.0,
                focus_letters=(),
                focus_weight=1.0,
                validation_only=True,
                model_version=version,
                v4_variant=variant or "full_v4",
                fixed_epoch_budget=True,
            )
            latency = measure_python_latency(checkpoint, data_dir, batch_size)
            per_run[name].append({
                "seed": seed,
                "best_epoch": report["best_epoch"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "parameter_count": report["parameter_count"],
                "validation_accuracy": report["validation_accuracy"],
                "validation_macro_accuracy": report["validation_macro_accuracy"],
                "worst_validation_accuracy": report["worst_validation_accuracy"],
                "validation_per_letter": report["validation_per_letter"],
                "python_latency": latency,
            })
            print(
                f"candidate={name} seed={seed} "
                f"val_overall={report['validation_accuracy']:.6%} "
                f"val_macro={report['validation_macro_accuracy']:.6%} "
                f"val_worst={report['worst_validation_accuracy']:.6%} "
                f"parameters={report['parameter_count']} "
                f"python_batch_ms={latency['mean_batch_ms']:.6f}"
            )

    summaries = {}
    for name, version, variant, data_dir, feature_contract in candidates:
        runs = per_run[name]
        parameter_counts = {run["parameter_count"] for run in runs}
        if len(parameter_counts) != 1:
            raise RuntimeError(f"{name}: parameter count changed between seeds")
        validation = {
            "overall_accuracy": _mean_std([run["validation_accuracy"] for run in runs]),
            "macro_accuracy": _mean_std([run["validation_macro_accuracy"] for run in runs]),
            "worst_class_accuracy": _mean_std([run["worst_validation_accuracy"] for run in runs]),
        }
        latency = {
            "mean": float(np.mean([run["python_latency"]["mean_batch_ms"] for run in runs])),
            "std": float(np.std([run["python_latency"]["mean_batch_ms"] for run in runs], ddof=0)),
        }
        summaries[name] = {
            "feature_contract": feature_contract,
            "architecture_variant": variant or "v3_mlp",
            "parameter_count": parameter_counts.pop(),
            "validation": validation,
            "python_latency_ms": latency,
            "runs": runs,
        }

    # Rank variants using validation means only; test data is not loaded here.
    selected_name = max(
        summaries,
        key=lambda name: (
            summaries[name]["validation"]["macro_accuracy"]["mean"],
            summaries[name]["validation"]["worst_class_accuracy"]["mean"],
            summaries[name]["validation"]["overall_accuracy"]["mean"],
        ),
    )
    selected_row = max(
        per_run[selected_name],
        key=lambda run: (
            run["worst_validation_accuracy"],
            run["validation_macro_accuracy"],
            run["validation_accuracy"],
        ),
    )
    selected_checkpoint = Path(selected_row["checkpoint"])
    selected_data_dir = v1_dir if selected_name in {"v3_mlp", "v1_features_shared_encoder"} else v4_dir
    frozen_checkpoint = output_dir / "final_selected_candidate.pt"
    shutil.copy2(selected_checkpoint, frozen_checkpoint)
    frozen_hash = sha256_file(frozen_checkpoint)
    test_evaluation_id = uuid.uuid4().hex
    selection_frozen_at = datetime.now(timezone.utc).isoformat()
    selection_manifest = {
        "selected_variant": selected_name,
        "selected_seed": selected_row["seed"],
        "checkpoint_sha256": frozen_hash,
        "checkpoint_path": str(frozen_checkpoint),
        "selected_data_dir": str(selected_data_dir),
        "protocol_seeds": list(seeds),
        "epoch_budget": epochs,
        "loss": "cross_entropy",
        "validation_only_selection": True,
        "test_evaluation_id": test_evaluation_id,
        "frozen_at_utc": selection_frozen_at,
        "evaluator_identity": getpass.getuser(),
        "aligned_splits": aligned,
    }
    selection_path = output_dir / "selection_frozen.json"
    selection_path.write_text(json.dumps(selection_manifest, indent=2), encoding="utf-8")
    selection_hash = sha256_file(selection_path)
    test_started_at = datetime.now(timezone.utc).isoformat()

    # The final candidate has been frozen and hashed. Open the test split once.
    test_report = evaluate(
        checkpoint=frozen_checkpoint,
        data_dir=selected_data_dir,
        split="test",
        batch_size=batch_size,
    )
    test_finished_at = datetime.now(timezone.utc).isoformat()
    if test_report["checkpoint_sha256"] != frozen_hash:
        raise RuntimeError("Final test report checkpoint hash does not match the frozen candidate")
    test_report["evaluation_run"] = {
        "evaluation_id": test_evaluation_id,
        "invocation": (
            f"evaluate(checkpoint={str(frozen_checkpoint)!r}, "
            f"data_dir={str(selected_data_dir)!r}, split='test', batch_size={batch_size})"
        ),
        "checkpoint_sha256": frozen_hash,
        "started_at_utc": test_started_at,
        "finished_at_utc": test_finished_at,
        "evaluator_identity": getpass.getuser(),
        "selection_manifest_sha256": selection_hash,
        "test_accuracy_evaluation_calls": 1,
    }

    try:
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = "unavailable"

    report = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_identity": getpass.getuser(),
        "code_commit_sha": code_commit,
        "protocol": {
            "seeds": list(seeds),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss": "cross_entropy",
            "dropout": 0.10,
            "target_per_class_validation_accuracy": 1.0,
            "candidate_training_validation_only": True,
            "test_accuracy_evaluation_calls": 1,
            "browser_latency": "NOT MEASURED",
        },
        "aligned_splits": aligned,
        "selection_manifest": {
            "path": str(selection_path),
            "sha256": selection_hash,
            "frozen_at_utc": selection_frozen_at,
            "test_evaluation_id": test_evaluation_id,
        },
        "selection": {
            "variant": selected_name,
            "seed": selected_row["seed"],
            "rule": "Validation-only ranking by mean macro accuracy, then mean weakest-class accuracy, then mean overall accuracy; seed checkpoint selected by weakest-class, macro, then overall validation accuracy.",
            "checkpoint_sha256": frozen_hash,
        },
        "candidates": summaries,
        "final_test": test_report,
    }
    report_path = output_dir / "v4_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(f"selection={selected_name} seed={selected_row['seed']}")
    print(f"checkpoint_sha256={frozen_hash}")
    print(f"final_test_overall_accuracy={test_report['overall_accuracy']:.6%}")
    print(f"final_test_macro_accuracy={test_report['macro_accuracy']:.6%}")
    print("test_accuracy_evaluation_calls=1")
    print(f"json_report={report_path}")
    print(f"markdown_report={markdown_output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched multi-seed validation-only V3/V4 comparison.")
    parser.add_argument("--data-dir-v1", required=True, type=Path)
    parser.add_argument("--data-dir-v4", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    run_comparison(
        v1_dir=args.data_dir_v1,
        v4_dir=args.data_dir_v4,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        markdown_output=args.markdown_output,
    )


if __name__ == "__main__":
    main()
