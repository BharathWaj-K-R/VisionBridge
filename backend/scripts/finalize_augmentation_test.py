"""Freeze the validation-selected augmentation candidate, then evaluate test once.

This command refuses a second attempt in the same output directory. It is the
only Phase 3 script allowed to open the test split.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

from app.training.evaluate_letter_base import evaluate
from scripts.aggregate_augmentation_results import AUGMENTATIONS, load_standalone_reports


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_rank(run: dict) -> tuple[float, float, float, int]:
    validation = run["validation"]
    return (
        float(validation["worst_letter_accuracy"]),
        float(validation["macro_accuracy"]),
        float(validation["overall_accuracy"]),
        -int(run["seed"]),
    )


def _best_run(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("No completed validation runs are available for final candidate selection")
    return max(runs, key=_validation_rank)


def _validate_combined_report(report_path: Path, report: dict, standalone: dict, phase2_hash: str) -> None:
    kept = [name for name in AUGMENTATIONS if standalone[name]["candidate"]["keep_candidate"]]
    if len(kept) < 2:
        raise ValueError("Combined report is not allowed when fewer than two standalone augmentations were kept")
    if not report_path.is_file():
        raise FileNotFoundError(f"Combined validation report is missing: {report_path}")
    if report.get("status") != "validation_completed" or report.get("augmentation") != "combined":
        raise ValueError("Combined report is not a completed combined validation run")
    if report.get("phase2_report_sha256") != phase2_hash:
        raise ValueError("Combined report uses a different Phase 2 comparison")
    if report.get("phase2_winner") != next(iter(standalone.values()))["phase2_winner"]:
        raise ValueError("Combined report uses a different Phase 2 winner")
    if report.get("seeds") != next(iter(standalone.values()))["seeds"]:
        raise ValueError("Combined report uses a different seed set")
    if report.get("augmentation_components") != kept:
        raise ValueError("Combined report components do not exactly equal the standalone validation keep set")
    protocol = report.get("protocol", {})
    if protocol.get("test_split_opened") is not False or protocol.get("test_accuracy_evaluation_calls") != 0:
        raise ValueError("Combined report must be validation-only")


def choose_final_candidate(
    *,
    standalone_paths: list[Path],
    phase2_report_path: Path,
    combined_result_path: Path | None = None,
) -> dict:
    reports = load_standalone_reports(standalone_paths)
    if set(reports) != set(AUGMENTATIONS):
        raise ValueError("Final selection requires all four standalone augmentation reports")

    phase2_hash = sha256_file(phase2_report_path)
    phase2_report = json.loads(phase2_report_path.read_text(encoding="utf-8"))
    phase2_winner = phase2_report.get("selection", {}).get("variant")
    for report in reports.values():
        if report["phase2_report_sha256"] != phase2_hash or report["phase2_winner"] != phase2_winner:
            raise ValueError("Standalone augmentation reports do not match the supplied Phase 2 winner/report")

    kept = [name for name in AUGMENTATIONS if reports[name]["candidate"]["keep_candidate"]]
    combined = None
    selected_augmentation = "none"
    selected_report_path = standalone_paths[0]
    selected_runs: list[dict]
    selected_role = "baseline"

    if len(kept) == 0:
        first_report = reports[AUGMENTATIONS[0]]
        selected_runs = first_report["baseline"]["runs"]
        selected_augmentation = "none"
        selected_role = "baseline"
    elif len(kept) == 1:
        name = kept[0]
        selected_report_path = next(
            path for path in standalone_paths
            if json.loads(path.read_text(encoding="utf-8")).get("augmentation") == name
        )
        selected_runs = reports[name]["candidate"]["runs"]
        selected_augmentation = name
        selected_role = "candidate"
    else:
        if combined_result_path is None:
            raise ValueError("Two or more augmentations were kept; run the single gated combined validation experiment before finalization")
        combined = json.loads(combined_result_path.read_text(encoding="utf-8"))
        _validate_combined_report(combined_result_path, combined, reports, phase2_hash)
        if combined["candidate"]["keep_candidate"]:
            selected_runs = combined["candidate"]["runs"]
            selected_augmentation = "combined"
            selected_role = "candidate"
            selected_report_path = combined_result_path
        else:
            # The combined run failed the predeclared keep rule. Keep the best
            # individually validated candidate; never use test to choose it.
            best_name = max(
                kept,
                key=lambda name: (
                    reports[name]["candidate"]["summary"]["macro_accuracy"]["mean"],
                    reports[name]["candidate"]["summary"]["worst_class_accuracy"]["mean"],
                    reports[name]["candidate"]["summary"]["overall_accuracy"]["mean"],
                ),
            )
            selected_runs = reports[best_name]["candidate"]["runs"]
            selected_augmentation = best_name
            selected_role = "candidate"
            selected_report_path = next(
                path for path in standalone_paths
                if json.loads(path.read_text(encoding="utf-8")).get("augmentation") == best_name
            )

    chosen = _best_run(selected_runs)
    return {
        "phase2_winner": phase2_winner,
        "phase2_report_sha256": phase2_hash,
        "standalone_reports": {
            name: {
                "path": str(next(
                    path for path in standalone_paths
                    if json.loads(path.read_text(encoding="utf-8")).get("augmentation") == name
                )),
                "sha256": sha256_file(next(
                    path for path in standalone_paths
                    if json.loads(path.read_text(encoding="utf-8")).get("augmentation") == name
                )),
                "keep_candidate": reports[name]["candidate"]["keep_candidate"],
                "mean_macro_gain": reports[name]["candidate"]["mean_macro_gain"],
            }
            for name in AUGMENTATIONS
        },
        "kept_standalone_augmentations": kept,
        "combined_report": (
            {"path": str(combined_result_path), "sha256": sha256_file(combined_result_path),
             "keep_candidate": combined["candidate"]["keep_candidate"]}
            if combined is not None else None
        ),
        "selected_augmentation": selected_augmentation,
        "selected_role": selected_role,
        "selected_validation_report": str(selected_report_path),
        "selected_validation_report_sha256": sha256_file(selected_report_path),
        "selected_seed": int(chosen["seed"]),
        "checkpoint_path": str(Path(chosen["checkpoint_path"]).resolve()),
        "checkpoint_sha256": chosen["checkpoint_sha256"],
        "validation": {
            "worst_letter_accuracy": chosen["validation"]["worst_letter_accuracy"],
            "macro_accuracy": chosen["validation"]["macro_accuracy"],
            "overall_accuracy": chosen["validation"]["overall_accuracy"],
            "per_letter_accuracy": chosen["validation"]["per_letter_accuracy"],
            "confusion_matrix": chosen["validation"]["confusion_matrix"],
        },
        "selection_rule": "Validation-only: when choosing a run within a selected experiment, rank by weakest-class accuracy, then macro accuracy, then overall accuracy; use lower seed only as a tie-breaker.",
    }


def finalize_test_evaluation(
    *,
    phase2_report: Path,
    standalone_results: list[Path],
    output_dir: Path,
    data_dir_v1: Path,
    data_dir_v4: Path,
    batch_size: int,
    combined_result: Path | None = None,
) -> dict:
    output_dir = output_dir.resolve()
    selection_path = output_dir / "selection_frozen.json"
    attempt_path = output_dir / "test_evaluation_attempted.json"
    result_path = output_dir / "final_test_evaluation.json"
    if selection_path.exists() or attempt_path.exists() or result_path.exists():
        raise FileExistsError("Final test evaluation is already frozen/attempted in this output directory; refusing to re-open test")

    selection = choose_final_candidate(
        standalone_paths=standalone_results,
        phase2_report_path=phase2_report,
        combined_result_path=combined_result,
    )
    checkpoint = Path(selection["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
    actual_checkpoint_hash = sha256_file(checkpoint)
    if actual_checkpoint_hash != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint SHA-256 changed after validation selection")

    phase2_variant = selection["phase2_winner"]
    selected_feature_contract = (
        "two-hand-wrist-scale-v1"
        if phase2_variant in {"v3_mlp", "v1_features_shared_encoder"}
        else "two-hand-geometry-v2"
    )
    selected_data_dir = data_dir_v1 if selected_feature_contract == "two-hand-wrist-scale-v1" else data_dir_v4
    test_npz = selected_data_dir / "test.npz"
    if not test_npz.is_file():
        raise FileNotFoundError(f"Selected feature-contract test split is missing: {test_npz}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "status": "selection_frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_identity": getpass.getuser(),
        "phase2_winner": phase2_variant,
        "feature_contract": selected_feature_contract,
        "selected_augmentation": selection["selected_augmentation"],
        "selected_role": selection["selected_role"],
        "selected_seed": selection["selected_seed"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint_hash,
        "selection_evidence": selection,
        "test_evaluation_calls_allowed": 1,
        "test_was_read_before_freeze": False,
    }
    selection_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    selection_sha256 = sha256_file(selection_path)
    evaluation_id = uuid.uuid4().hex
    invocation = (
        f"evaluate(checkpoint={str(checkpoint)!r}, data_dir={str(selected_data_dir.resolve())!r}, "
        f"split='test', batch_size={batch_size})"
    )
    attempt = {
        "status": "attempt_started",
        "evaluation_id": evaluation_id,
        "attempt_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_identity": getpass.getuser(),
        "selection_manifest_path": str(selection_path),
        "selection_manifest_sha256": selection_sha256,
        "checkpoint_sha256": actual_checkpoint_hash,
        "invocation": invocation,
        "test_accuracy_evaluation_calls_allowed": 1,
    }
    # Persist the one-shot gate before the evaluator opens test; a failed attempt
    # is still an attempt and cannot silently be retried against the same test.
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf-8")
    report = evaluate(
        checkpoint=checkpoint,
        data_dir=selected_data_dir,
        split="test",
        batch_size=batch_size,
    )
    if report.get("checkpoint_sha256") != actual_checkpoint_hash:
        raise RuntimeError("Final test evaluation checkpoint SHA-256 does not match the frozen selection")
    report["evaluation_run"] = {
        **attempt,
        "status": "completed",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"selected_augmentation={selection['selected_augmentation']}")
    print(f"selected_seed={selection['selected_seed']}")
    print(f"checkpoint_sha256={actual_checkpoint_hash}")
    print(f"test_overall_accuracy={report['overall_accuracy']:.6%}")
    print(f"test_macro_accuracy={report['macro_accuracy']:.6%}")
    print("test_accuracy_evaluation_calls=1")
    print(f"test_report={result_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Perform the one final test evaluation after augmentation decisions are frozen.")
    parser.add_argument("--phase2-report", required=True, type=Path)
    parser.add_argument("--standalone-result", action="append", type=Path, required=True)
    parser.add_argument("--combined-result", type=Path, default=None)
    parser.add_argument("--data-dir-v1", required=True, type=Path)
    parser.add_argument("--data-dir-v4", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    finalize_test_evaluation(
        phase2_report=args.phase2_report,
        standalone_results=args.standalone_result,
        combined_result=args.combined_result,
        output_dir=args.output_dir,
        data_dir_v1=args.data_dir_v1,
        data_dir_v4=args.data_dir_v4,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
