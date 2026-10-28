"""Aggregate independent validation-only augmentation reports into the experiment table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AUGMENTATIONS = ("gaussian_jitter", "in_plane_rotation", "depth_scale", "finger_joint_angles")
DOC_DEFAULT = Path(__file__).resolve().parents[2] / "docs" / "experiments" / "augmentation_comparison.md"
START_MARKER = "<!-- AUTO_RESULTS_START -->"
END_MARKER = "<!-- AUTO_RESULTS_END -->"


def _pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def _pm(metric: dict) -> str:
    return f"{_pct(float(metric['mean']))} ± {_pct(float(metric['std']))}"


def _signed(value: float) -> str:
    return f"{value * 100:+.3f} pp"


def _signed_pm(metric: dict) -> str:
    return f"{_signed(float(metric['mean']))} ± {float(metric['std']) * 100:.3f} pp"


def load_standalone_reports(paths: list[Path]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    expected_phase2_hash = expected_winner = expected_seeds = expected_splits = None
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Augmentation result report not found: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        augmentation = report.get("augmentation")
        if report.get("status") != "validation_completed" or augmentation not in AUGMENTATIONS:
            raise ValueError(f"{path}: not a completed standalone augmentation validation report")
        protocol = report.get("protocol", {})
        if protocol.get("test_split_opened") is not False or protocol.get("test_accuracy_evaluation_calls") != 0:
            raise ValueError(f"{path}: report indicates test data may have been opened")
        if augmentation in reports:
            raise ValueError(f"Duplicate standalone report for augmentation {augmentation}")
        if report.get("candidate", {}).get("keep_candidate") not in (True, False):
            raise ValueError(f"{path}: missing validation-based keep decision")

        phase2_hash = report.get("phase2_report_sha256")
        winner = report.get("phase2_winner")
        seeds = report.get("seeds")
        splits = report.get("aligned_train_val_manifests")
        if not phase2_hash or not isinstance(winner, str) or not isinstance(seeds, list) or len(seeds) < 3:
            raise ValueError(f"{path}: missing baseline/protocol identity")
        if expected_phase2_hash is None:
            expected_phase2_hash, expected_winner, expected_seeds, expected_splits = phase2_hash, winner, seeds, splits
        elif (
            phase2_hash != expected_phase2_hash
            or winner != expected_winner
            or seeds != expected_seeds
            or splits != expected_splits
        ):
            raise ValueError("Standalone reports do not share the same Phase 2 winner, seeds and train/val manifests")
        reports[augmentation] = report
    return reports


def render_results_table(reports: dict[str, dict]) -> str:
    status = "ALL FOUR STANDALONE RUNS COMPLETED" if len(reports) == 4 else f"PARTIAL ({len(reports)}/4 standalone runs completed)"
    rows = [
        f"**Real-data status: {status}.**",
        "",
        "| Augmentation | Validation macro mean ± std | Macro delta vs baseline | Required seed-variance threshold | Keep? | Rejection rate | W→M count Δ | W→N count Δ | W→O count Δ | Python model latency |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for augmentation in AUGMENTATIONS:
        if augmentation not in reports:
            rows.append(f"| {augmentation} | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |")
            continue
        candidate = reports[augmentation]["candidate"]
        summary = candidate["summary"]
        rejection = candidate["rejection"]
        w = candidate["w_confusion_count_delta"]
        rows.append(
            "| {name} | {macro} | {delta} | {threshold} | {keep} | {reject} | {wm} | {wn} | {wo} | {latency} ms |".format(
                name=augmentation,
                macro=_pm(summary["macro_accuracy"]),
                delta=_signed(float(candidate["mean_macro_gain"])),
                threshold=_signed(float(candidate["required_gain_threshold"])),
                keep="KEEP" if candidate["keep_candidate"] else "DROP",
                reject=_pct(float(rejection["rejection_rate"])),
                wm=_signed_pm(w["W_to_M"]),
                wn=_signed_pm(w["W_to_N"]),
                wo=_signed_pm(w["W_to_O"]),
                latency=_pm(summary["mean_batch_latency_ms"]),
            )
        )
    rows += [
        "",
        "### Per-class validation accuracy change (candidate minus unaugmented baseline)",
        "",
        "| Letter | gaussian_jitter | in_plane_rotation | depth_scale | finger_joint_angles |",
        "|---|---|---|---|---|",
    ]
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        values = []
        for augmentation in AUGMENTATIONS:
            if augmentation not in reports:
                values.append("NOT RUN")
            else:
                values.append(_signed_pm(reports[augmentation]["candidate"]["per_letter_accuracy_delta"][letter]))
        rows.append("| " + letter + " | " + " | ".join(values) + " |")
    rows += [
        "",
        "### Evidence and decision boundary",
        "",
        "All standalone reports share this Phase 2 winner: {}.".format(next(iter(reports.values()))["phase2_winner"] if reports else "NOT SELECTED"),
        "Only validation metrics were used for these decisions. The test split was not opened by these runs. Keep decisions use the predeclared rule: macro gain must exceed the larger of baseline and candidate sample standard deviations across seeds.",
        "The historical test confusion matrix remains context only; W→M/N/O tuning must use validation confusion counts above.",
    ]
    return "\n".join(rows)


def update_markdown(path: Path, reports: dict[str, dict]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment document not found: {path}")
    text = path.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        raise ValueError("Experiment document is missing AUTO_RESULTS markers")
    start = text.index(START_MARKER) + len(START_MARKER)
    end = text.index(END_MARKER, start)
    text = text[:start] + "\n" + render_results_table(reports) + "\n" + text[end:]
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed standalone augmentation reports into the Markdown table.")
    parser.add_argument("--result", action="append", type=Path, default=[], help="Standalone augmentation_result.json; repeat for each augmentation.")
    parser.add_argument("--output", type=Path, default=DOC_DEFAULT)
    args = parser.parse_args()
    reports = load_standalone_reports(args.result)
    update_markdown(args.output, reports)
    print(f"reports_loaded={len(reports)}")
    print(f"standalone_augmentations_complete={len(reports) == len(AUGMENTATIONS)}")
    print(f"updated_markdown={args.output}")


if __name__ == "__main__":
    main()
