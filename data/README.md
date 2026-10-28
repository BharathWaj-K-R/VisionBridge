# VisionBridge data workspace

The active model recognizes isolated ISL fingerspelled letters A–Z from a normalized two-hand landmark vector. Training data and generated arrays are workspace artifacts, not files tracked in this repository.

## Reproducible source

Training uses the public [RealSign Indian Sign Language alphabet dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset). The training notebook pins source commit `17c51dcc158b7b6359b7c6667edc51c7f148f3f4` and downloads:

- Archive size: `656689688` bytes
- Archive SHA-256: `008cae248e346b8c31fbbea057fcc3f69c6909d29a88e6bb1fb0369f528de2b5`

The archive is downloaded directly from the pinned public source. It is not committed to this repository, and Git LFS is not required for this dataset. The notebook also verifies the pinned MediaPipe Hand Landmarker task asset before preparation.

## Preparation

The canonical preparation script is `backend/scripts/prepare_letter_dataset.py`. It extracts two-hand landmarks with MediaPipe Tasks Hand Landmarker 0.10.35 and applies the active `two-hand-wrist-scale-v1` 126D feature contract.

The generated workspace contains:

```text
visionbridge_letter_data/
├── train.npz
├── val.npz
├── test.npz
├── labels.json
├── train_manifest.jsonl
├── val_manifest.jsonl
├── test_manifest.jsonl
└── duplicate_report.json
```

The preparation pipeline:

1. Keeps the dataset's source test split out of training and validation.
2. Builds a stratified 80/20 train/validation split from the source training and validation pools.
3. Records source paths, class labels, source split, and exact image SHA-256 values in JSONL manifests.
4. Groups exact-hash duplicates so they stay together during the generated train/validation split; reports exact-hash overlap involving the source test split.
5. Records class counts and images skipped when a usable hand landmark set cannot be extracted.

Exact-hash checks and landmark-space near-duplicate checks address data leakage. Signer-independent evaluation still requires verified signer IDs; legacy RealSign source folders do not provide them.

## Manifest v2 and extraction failure reporting

The schema and validator are documented in MANIFEST_SCHEMA.md. A manifest contains real, consented pseudonymous signer IDs; the tooling never infers signer identity. Place manifest_v2.jsonl under the input root or pass --manifest-v2. Rows may use split = unassigned for deterministic signer-group allocation, or may provide preassigned train/val/test splits that already keep each signer in only one partition.

Run the validator with:

    PYTHONPATH=backend python backend/scripts/validate_manifest.py --manifest /path/to/manifest_v2.jsonl

For legacy inputs without manifest v2, the existing sample-level train/validation procedure remains active and labels.json explicitly records signer_split = unavailable. Preparation writes extraction_failure_report.json with per-letter, per-signer and per-capture-condition failure rates. duplicate_report.json includes landmark-space near-duplicate checks; a cross-split near-duplicate is a pipeline failure. The RMS distance threshold defaults to 0.01 and can be changed with --near-duplicate-threshold.

## Training and evaluation

- Colab training notebook: `notebooks/train_letter_base_colab.ipynb`
- Evaluation-only notebook: `notebooks/evaluate_letter_base_v3_colab.ipynb`
- Training implementation: `backend/app/training/letter_base.py`
- Evaluation implementation: `backend/app/training/evaluate_letter_base.py`

The current recorded evaluation is in `visionbridge_letter_evaluation.json`, associated with checkpoint SHA-256:

```text
f903872e96f269c5fbee6ea897f0027b91e162f2fc45b4c9a115e8ba8e38c917
```

Recorded prepared-test result: 4,633 samples, 98.273257% overall accuracy, 98.291294% macro class accuracy, and 78.770950% for W, the weakest class. These are stored dataset-evaluation results, not a live camera guarantee; source images skipped during landmark extraction are not represented in the prepared-test metric.

## Important evaluation limitation

The active RealSign folder metadata does not provide verified signer identifiers. Signer-independent evaluation is therefore **BLOCKED** and remains a required release gate. Do not label the current evaluation as signer-held-out or signer-independent.

Signer calibration happens at application runtime from user examples; it does not require a separate offline adapter-training job. Replacing the base checkpoint changes the embedding space and requires recalibration.
