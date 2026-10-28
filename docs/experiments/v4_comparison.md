# V3 / V4 comparison

**Status: NOT RUN.** No aligned, prepared V1 126D and V4 141D train/validation/test datasets are available in this execution environment. No validation or final-test accuracy numbers are published here.

The V4 implementation is selectable, but no candidate has been promoted or wired into the browser default. The active RealSign source does not provide verified signer IDs; signer-independent evaluation remains blocked. Do not infer signer IDs or treat the current class-folder split as signer held out.

## Candidate table

The parameter counts below are structural counts from the model definitions, not trained performance measurements. V4 variant tests assert the V4 counts; the V3 count is calculated from the existing V3 architecture.

| Candidate | Feature contract | Parameters | Validation overall (mean ± std) | Validation macro (mean ± std) | Python latency | Final test |
|---|---|---:|---|---|---|---|
| v3_mlp | two-hand-wrist-scale-v1, 126D | 26,582 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| v4_features_old_mlp | two-hand-geometry-v2, 141D | 28,532 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| v1_features_shared_encoder | two-hand-wrist-scale-v1, 126D | 25,156 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| full_v4 | two-hand-geometry-v2, 141D | 25,156 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

## Reproduction protocol

1. Generate V4 features aligned to the existing V1 manifests without changing split assignments:

       PYTHONPATH=backend python backend/scripts/prepare_v4_feature_dataset.py --input-root /path/to/raw_dataset --data-dir-v1 /path/to/visionbridge_letter_data --output-dir /path/to/visionbridge_v4_letter_data

2. Run the matched validation-only experiment from the repository root:

       PYTHONPATH=backend python backend/scripts/run_v4_comparison.py --data-dir-v1 /path/to/visionbridge_letter_data --data-dir-v4 /path/to/visionbridge_v4_letter_data --output-dir /path/to/v4_experiment_run --seeds 42 123 314 --epochs 1000 --batch-size 128

3. The script verifies identical ordered sample identities and labels across all splits, trains every candidate and seed with the same cross-entropy setup and epoch budget, ranks variants using validation only, freezes the chosen checkpoint, then invokes the evaluator for one final test accuracy run. It writes the full per-seed JSON report beside the checkpoint and populates this Markdown table with actual results.

The current table intentionally contains no fabricated accuracy, latency, or standard-deviation values. Browser latency must remain NOT MEASURED unless separately measured on a named browser/device. A final aggregate test result does not establish signer-independent generalization.
