# VisionBridge Base-Model Debug Log

## 2026-08-25 — Root cause found

### Symptom

The committed `base_model.pt` loaded successfully and produced finite logits, but both known/training and unseen real ISL videos decoded to CTC blank:

```text
PREDICTED: (no sign detected)
BLANK RATIO: 1.0000
NON-BLANK FRAMES: 0
LOGITS FINITE: True
```

The same behavior was observed on a known `fever (2).MP4` sample whose ground truth was `i am suffering from fever`.

### Root cause

The Colab training notebook used `Path(video).stem` as the UID for processed pose/face files. The Kaggle ISL-CSLTR dataset stores clips inside sentence-label directories and reuses filenames such as `fever (2).MP4` across different directories.

Therefore different physical videos could map to the same processed paths:

```text
pose/fever (2).npy
face/fever (2).npy
```

Later extractions overwrote earlier features, while the CSV could still contain multiple rows using the same UID but different text labels. This silently paired the wrong keypoints with the wrong translation targets and could produce a model that optimizes CTC loss without learning a valid sign-to-text mapping.

The bug was confirmed from the actual notebook code and the runtime dataset layout observed in Colab.

### Fix implemented

1. `notebooks/train_base_model_colab.ipynb` was rebuilt to derive a globally unique UID from sentence label, filename stem, and a SHA-1 hash of the video's relative path.
2. The notebook rebuilds a clean private runtime feature directory.
3. The notebook explicitly checks UID collisions before extraction.
4. `backend/app/training/isltranslate.py` rejects duplicate UIDs and unusable targets.
5. Regression coverage was added for UID integrity.
6. The training notebook added semantic acceptance before checkpoint publication.

### Status

**VERIFIED root cause:** dataset UID collision / feature overwrite in the previous Colab preparation path.

**VERIFIED code fix:** collision-safe UID generation + duplicate UID guard are present on `main`.

**NOT YET VERIFIED:** a production-quality trained checkpoint.

---

## 2026-08-28 — Hand-aware training-path audit

The hand-aware migration was inspected and concrete test/verification regressions were corrected. The old pose+face checkpoint remains intentionally rejected.

Static fixes included the hand-aware padding-mask test, semantic-gate threshold defaults, richer CTC diagnostics, and frontend CI typecheck/build gates.

The overfit gate now reports first-step gradients, parameter updates, blank/space ratios, and target-character probability peaks so a real GPU run can distinguish optimizer failure, feature/representation failure, and CTC decoding/alignment problems.

**RUNTIME STATUS:** real-data hand-aware extraction and semantic overfit were not available in this environment.

---

## 2026-09-06 — CTC model stabilization fix

### Problem

The previous hand-aware model used four independent Transformer stream encoders followed by gated fusion, temporal convolution, and a shared Transformer. That design was valid structurally but remained unproven against the observed CTC blank/space collapse.

### Fix

`backend/app/models/base_model.py` was migrated to `hand-aware-gru-ctc-v2`:

```text
pose / face / left hand / right hand
        |
per-stream LayerNorm + projection
        |
frame-to-frame motion projection
        |
learned gated multimodal fusion
        |
temporal depthwise + pointwise convolution
        |
packed bidirectional GRU
        |
character CTC head
```

The change deliberately reduces optimization complexity while preserving the four-stream hand-aware contract. It also adds per-frame input normalization, explicit motion features, packed sequence handling, and a negative initial CTC blank bias so the model is less likely to start in an all-blank regime.

The legacy checkpoint is still rejected because its state dictionary does not contain the hand-aware stream parameters and is incompatible with the new architecture.

### Adapter integration fix

`backend/app/models/bridge_adapter.py` previously assumed `base_model.shared_encoder.layers`, which is a Transformer API. The new base model uses a GRU, so that assumption would break adapter calibration/runtime even when the base model itself was correct.

The adapter now supports both forms:

```text
Transformer -> run encoder layers, then adapters
GRU         -> run recurrent encoder, then adapter stack
```

The adapter test was updated to determine the temporal layer count from `num_layers` when the encoder is recurrent.

### Training-gate fix

The first semantic gate is now a true single-sample capacity probe by default:

```text
samples = 1
learning rate = 1e-3
weight decay = 0
```

This isolates whether the new model can actually learn one real sample before multi-sample/generalization acceptance is attempted later in the pipeline.

The canonical Colab notebook was updated to use the same settings.

### Verification

GitHub Actions showed the frontend regression job passing on the earlier diagnostic commit. The backend job exposed the Transformer-specific adapter test regression, which was then fixed.

The latest backend workflow for the adapter fix was still running at the time of this diary update, so the following remain **NOT VERIFIED** until the job completes:

```text
backend pytest
frontend build on latest commit
real ISL extraction
single-sample hand-aware semantic overfit
full training
real-video validation
```

### Release rule

Do not publish or deploy a newly trained checkpoint until:

```text
single-sample semantic gate PASS
        -> full training
        -> train/held-out acceptance PASS
        -> multi-video real validation PASS
```
