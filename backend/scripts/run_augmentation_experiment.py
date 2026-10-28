"""Compare one raw-coordinate augmentation against the Phase 2 winner on validation only.

Invoke this script separately for each augmentation. It never loads a test split,
confusion matrix, or test score. The final test is reserved for the separate
one-shot finalization command after standalone experiments (and, if justified,
one combined run) have been decided from validation results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.services.letter_fewshot import normalize_hand_pair, normalize_hand_pair_v2
from app.training.evaluate_letter_base import evaluate
from app.training.landmark_augment import (
    AUGMENTATION_NAMES,
    apply_augmentation_with_fallback,
)
from app.training.letter_base import load_split, train_model
from app.models.letter_model import INPUT_DIM, LETTER_LABELS

SUPPORTED_EXPERIMENT_AUGMENTATIONS = (
    "gaussian_jitter",
    "in_plane_rotation",
    "depth_scale",
    "finger_joint_angles",
)
PHASE2_VARIANTS = (
    "v3_mlp",
    "v4_features_old_mlp",
    "v1_features_shared_encoder",
    "full_v4",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required split manifest is missing: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("landmark_status") not in (None, "ok"):
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


def verify_train_val_alignment(data_dir: Path, raw_dir: Path, input_dim: int) -> dict:
    """Validate only train and validation. This function deliberately never opens test files."""
    evidence = {}
    for split in ("train", "val"):
        data_manifest = data_dir / f"{split}_manifest.jsonl"
        raw_manifest = raw_dir / f"{split}_manifest.jsonl"
        data_rows = _manifest_rows(data_manifest)
        raw_rows = _manifest_rows(raw_manifest)
        if [_identity(row) for row in data_rows] != [_identity(row) for row in raw_rows]:
            raise ValueError(f"{split}: raw/source ordered sample identities differ")

        x, y = load_split(data_dir, split, input_dim=input_dim)
        raw_path = raw_dir / f"{split}_raw.npz"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw pre-normalization {split} data: {raw_path}")
        with np.load(raw_path, allow_pickle=False) as raw_data:
            raw_x = np.asarray(raw_data["x"], dtype=np.float32)
            raw_y = np.asarray(raw_data["y"], dtype=np.int64)
            left_present = np.asarray(raw_data["left_present"], dtype=np.bool_)
            right_present = np.asarray(raw_data["right_present"], dtype=np.bool_)

        expected_y = np.asarray(
            [LETTER_LABELS.index(row.get("label", row.get("letter"))) for row in raw_rows],
            dtype=np.int64,
        )
        if (
            raw_x.shape != (len(y), INPUT_DIM)
            or not np.isfinite(raw_x).all()
            or raw_y.shape != y.numpy().shape
            or not np.array_equal(raw_y, y.numpy())
            or not np.array_equal(raw_y, expected_y)
            or left_present.shape != raw_y.shape
            or right_present.shape != raw_y.shape
        ):
            raise ValueError(f"{split}: raw features, explicit presence flags, labels or row order are misaligned")
        if len(data_rows) != len(x) or len(raw_rows) != len(raw_x):
            raise ValueError(f"{split}: manifest and feature-array counts differ")

        evidence[split] = {
            "samples": len(y),
            "source_manifest_sha256": sha256_file(data_manifest),
            "raw_manifest_sha256": sha256_file(raw_manifest),
        }
    return evidence


def _load_phase2_selection(report_path: Path) -> tuple[str, str, str, str]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Phase 2 result JSON is required; no winner will be guessed: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError("Phase 2 comparison is not completed; augmentation experiments are blocked")
    selection = report.get("selection")
    candidates = report.get("candidates")
    if not isinstance(selection, dict) or not isinstance(candidates, dict):
        raise ValueError("Phase 2 report lacks a recorded validation-selected candidate")
    variant = selection.get("variant")
    if variant not in PHASE2_VARIANTS or variant not in candidates:
        raise ValueError("Phase 2 selected candidate is missing or unsupported")
    candidate = candidates[variant]
    feature_contract = candidate.get("feature_contract")
    expected_contract = (
        "two-hand-wrist-scale-v1"
        if variant in {"v3_mlp", "v1_features_shared_encoder"}
        else "two-hand-geometry-v2"
    )
    if feature_contract != expected_contract:
        raise ValueError("Phase 2 winner feature contract does not match its declared variant")
    model_version = "v3" if variant == "v3_mlp" else "v4"
    v4_variant = variant if model_version == "v4" else "full_v4"
    return variant, feature_contract, model_version, v4_variant


def _normalized_augmented_features(
    raw_x: np.ndarray,
    left_present: np.ndarray,
    right_present: np.ndarray,
    *,
    feature_contract: str,
) -> np.ndarray:
    rows = []
    for index, raw in enumerate(raw_x):
        if feature_contract == "two-hand-wrist-scale-v1":
            row = normalize_hand_pair(raw.tolist())
        elif feature_contract == "two-hand-geometry-v2":
            row = normalize_hand_pair_v2(
                raw.tolist(),
                left_present=bool(left_present[index]),
                right_present=bool(right_present[index]),
            )
        else:
            raise ValueError(f"Unsupported feature contract: {feature_contract}")
        rows.append(row)
    features = np.stack(rows).astype(np.float32, copy=False)
    expected_width = 126 if feature_contract == "two-hand-wrist-scale-v1" else 141
    if features.shape != (len(raw_x), expected_width) or not np.isfinite(features).all():
        raise ValueError("Augmented and normalized feature matrix violates the selected contract")
    return features


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _summary(runs: list[dict]) -> dict:
    return {
        "overall_accuracy": _mean_std([run["validation"]["overall_accuracy"] for run in runs]),
        "macro_accuracy": _mean_std([run["validation"]["macro_accuracy"] for run in runs]),
        "worst_class_accuracy": _mean_std([run["validation"]["worst_letter_accuracy"] for run in runs]),
        "mean_batch_latency_ms": _mean_std([run["validation"]["latency"]["mean_batch_ms"] for run in runs]),
        "mean_sample_latency_ms": _mean_std([run["validation"]["latency"]["mean_sample_ms"] for run in runs]),
    }


def _mean_per_letter_delta(baseline_runs: list[dict], candidate_runs: list[dict]) -> dict[str, dict[str, float]]:
    result = {}
    for letter in LETTER_LABELS:
        paired = [
            candidate["validation"]["per_letter_accuracy"][letter]
            - baseline["validation"]["per_letter_accuracy"][letter]
            for baseline, candidate in zip(baseline_runs, candidate_runs)
        ]
        result[letter] = _mean_std(paired)
    return result


def _w_confusion_delta(baseline_runs: list[dict], candidate_runs: list[dict]) -> dict[str, dict[str, float]]:
    labels = list(LETTER_LABELS)
    truth_w = labels.index("W")
    result = {}
    for predicted in ("M", "N", "O"):
        column = labels.index(predicted)
        paired = [
            candidate["validation"]["confusion_matrix"][truth_w][column]
            - baseline["validation"]["confusion_matrix"][truth_w][column]
            for baseline, candidate in zip(baseline_runs, candidate_runs)
        ]
        result[f"W_to_{predicted}"] = _mean_std([float(value) for value in paired])
    return result


def _create_work_data(root: Path, source_data_dir: Path, train_x: np.ndarray, train_y: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source_data_dir / "labels.json").read_text(encoding="utf-8"))
    (root / "labels.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(root / "train.npz", x=train_x.astype(np.float32, copy=False), y=train_y.astype(np.int64, copy=False))
    with np.load(source_data_dir / "val.npz", allow_pickle=False) as val_data:
        val_x = np.asarray(val_data["x"], dtype=np.float32)
        val_y = np.asarray(val_data["y"], dtype=np.int64)
    np.savez_compressed(root / "val.npz", x=val_x, y=val_y)


def _train_and_validate(
    *,
    role: str,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    source_data_dir: Path,
    output_dir: Path,
    model_version: str,
    v4_variant: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict:
    work_root = output_dir / "_work" / f"{role}-seed-{seed}"
    if work_root.exists():
        shutil.rmtree(work_root)
    _create_work_data(work_root, source_data_dir, train_x, train_y)
    checkpoint = output_dir / "checkpoints" / f"{role}-seed-{seed}.pt"
    training = train_model(
        root=work_root,
        output_path=checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        target_class_accuracy=1.0,
        seed=seed,
        hidden_dim=128,
        embedding_dim=64,
        dropout=0.10,
        loss_name="cross_entropy",
        focal_gamma=1.5,
        landmark_jitter_std=0.0,
        rotation_degrees=0.0,
        focus_letters=(),
        focus_weight=1.0,
        validation_only=True,
        model_version=model_version,
        v4_variant=v4_variant,
        fixed_epoch_budget=True,
    )
    validation = evaluate(
        checkpoint=checkpoint,
        data_dir=work_root,
        split="val",
        batch_size=batch_size,
    )
    return {
        "role": role,
        "seed": seed,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_best_epoch": training["best_epoch"],
        "training_epochs_completed": training["epochs_completed"],
        "parameter_count": training["parameter_count"],
        "validation": validation,
    }


def run_augmentation_experiment(
    *,
    phase2_report: Path,
    data_dir_v1: Path,
    data_dir_v4: Path,
    raw_dir: Path,
    output_dir: Path,
    augmentation: str,
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> dict:
    if augmentation not in SUPPORTED_EXPERIMENT_AUGMENTATIONS:
        raise ValueError("Exactly one supported augmentation must be named per run")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("At least three distinct seeds are required")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    winner, feature_contract, model_version, v4_variant = _load_phase2_selection(phase2_report)
    source_data_dir = data_dir_v1 if feature_contract == "two-hand-wrist-scale-v1" else data_dir_v4
    input_dim = 126 if feature_contract == "two-hand-wrist-scale-v1" else 141

    # This reads only train and val manifests and arrays. It never opens test files.
    split_evidence = verify_train_val_alignment(source_data_dir, raw_dir, input_dim)
    raw_meta = json.loads((raw_dir / "labels.json").read_text(encoding="utf-8"))
    if raw_meta.get("raw_landmark_contract") != "raw-mediapipe-xyz-before-normalization-v1":
        raise ValueError("Raw input lacks the required pre-normalization coordinate contract")

    train_x, train_y = load_split(source_data_dir, "train", input_dim=input_dim)
    with np.load(raw_dir / "train_raw.npz", allow_pickle=False) as raw_data:
        raw_train_x = np.asarray(raw_data["x"], dtype=np.float32)
        raw_train_y = np.asarray(raw_data["y"], dtype=np.int64)
        left_present = np.asarray(raw_data["left_present"], dtype=np.bool_)
        right_present = np.asarray(raw_data["right_present"], dtype=np.bool_)
    if not np.array_equal(raw_train_y, train_y.numpy()):
        raise ValueError("Raw train labels differ from model-space training labels")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_runs = []
    candidate_runs = []
    rejection_reports = []
    for seed in seeds:
        baseline_runs.append(_train_and_validate(
            role=f"{augmentation}-baseline",
            seed=seed,
            train_x=train_x.numpy(),
            train_y=train_y.numpy(),
            source_data_dir=source_data_dir,
            output_dir=output_dir,
            model_version=model_version,
            v4_variant=v4_variant,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ))

        augmented_raw, rejection_report = apply_augmentation_with_fallback(
            raw_train_x,
            augmentation=augmentation,
            seed=seed,
        )
        augmented_features = _normalized_augmented_features(
            augmented_raw,
            left_present,
            right_present,
            feature_contract=feature_contract,
        )
        if augmented_features.shape != train_x.shape or not np.isfinite(augmented_features).all():
            raise ValueError("Augmented feature matrix shape differs from baseline train data")
        rejection_reports.append(rejection_report)
        candidate_runs.append(_train_and_validate(
            role=f"{augmentation}-candidate",
            seed=seed,
            train_x=augmented_features,
            train_y=train_y.numpy(),
            source_data_dir=source_data_dir,
            output_dir=output_dir,
            model_version=model_version,
            v4_variant=v4_variant,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ))
        print(
            f"augmentation={augmentation} seed={seed} "
            f"baseline_val_macro={baseline_runs[-1]['validation']['macro_accuracy']:.6%} "
            f"candidate_val_macro={candidate_runs[-1]['validation']['macro_accuracy']:.6%} "
            f"rejection_rate={rejection_report['rejection_rate']:.6%}"
        )

    baseline_summary = _summary(baseline_runs)
    candidate_summary = _summary(candidate_runs)
    macro_gain = (
        candidate_summary["macro_accuracy"]["mean"]
        - baseline_summary["macro_accuracy"]["mean"]
    )
    required_gain = max(
        baseline_summary["macro_accuracy"]["std"],
        candidate_summary["macro_accuracy"]["std"],
    )
    keep_candidate = macro_gain > required_gain
    per_letter_delta = _mean_per_letter_delta(baseline_runs, candidate_runs)
    w_confusion_delta = _w_confusion_delta(baseline_runs, candidate_runs)
    rejection_summary = {
        "attempted": sum(item["attempted"] for item in rejection_reports),
        "accepted": sum(item["accepted"] for item in rejection_reports),
        "rejected": sum(item["rejected"] for item in rejection_reports),
        "rejection_rate": (
            sum(item["rejected"] for item in rejection_reports)
            / max(sum(item["attempted"] for item in rejection_reports), 1)
        ),
        "by_seed": rejection_reports,
    }

    report = {
        "status": "validation_completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "augmentation": augmentation,
        "phase2_winner": winner,
        "phase2_report_sha256": sha256_file(phase2_report),
        "feature_contract": feature_contract,
        "model_version": model_version,
        "v4_variant": v4_variant if model_version == "v4" else None,
        "seeds": list(seeds),
        "protocol": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss": "cross_entropy",
            "dropout": 0.10,
            "fixed_epoch_budget": True,
            "candidate_and_baseline_use_identical_seed_set": True,
            "train_sample_count_unchanged": True,
            "validation_unaugmented": True,
            "test_split_opened": False,
            "test_accuracy_evaluation_calls": 0,
            "selection_rule": (
                "Keep only if candidate validation macro mean minus baseline validation "
                "macro mean exceeds the larger of their sample standard deviations (ddof=1)."
            ),
        },
        "aligned_train_val_manifests": split_evidence,
        "baseline": {
            "summary": baseline_summary,
            "runs": baseline_runs,
        },
        "candidate": {
            "summary": candidate_summary,
            "runs": candidate_runs,
            "mean_macro_gain": macro_gain,
            "required_gain_threshold": required_gain,
            "keep_candidate": keep_candidate,
            "per_letter_accuracy_delta": per_letter_delta,
            "w_confusion_count_delta": w_confusion_delta,
            "rejection": rejection_summary,
        },
        "test_status": "NOT OPENED; reserved for one final evaluation after all validation decisions",
        "browser_latency": "NOT MEASURED",
    }
    report_path = output_dir / "augmentation_result.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"mean_macro_gain={macro_gain:.6%}")
    print(f"required_seed_variance_threshold={required_gain:.6%}")
    print(f"keep_candidate={str(keep_candidate).lower()}")
    print(f"rejection_rate={rejection_summary['rejection_rate']:.6%}")
    print(f"test_accuracy_evaluation_calls=0")
    print(f"result_json={report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one standalone raw-landmark augmentation against the Phase 2 winner.")
    parser.add_argument("--phase2-report", required=True, type=Path)
    parser.add_argument("--data-dir-v1", required=True, type=Path)
    parser.add_argument("--data-dir-v4", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--augmentation", required=True, choices=SUPPORTED_EXPERIMENT_AUGMENTATIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 314])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    run_augmentation_experiment(
        phase2_report=args.phase2_report,
        data_dir_v1=args.data_dir_v1,
        data_dir_v4=args.data_dir_v4,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        augmentation=args.augmentation,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
