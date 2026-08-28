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

1. `notebooks/train_base_model_colab.ipynb` was rebuilt to derive a globally unique UID from:
   - sentence label
   - filename stem
   - SHA-1 hash of the video's relative path
2. The notebook deletes the previous processed dataset before rebuilding it, preventing contaminated old stem-only `.npy` files from being reused.
3. The notebook explicitly checks for UID collisions before extraction.
4. `backend/app/training/isltranslate.py` now rejects duplicate UIDs in `ISLTranslate.csv` and rejects samples with no encodable target text.
5. `backend/tests/test_dataset_uid_integrity.py` adds regression coverage for duplicate UID rejection and empty-target handling.
6. The training notebook now has a post-training acceptance gate that checks both a training example and a held-out example for non-blank CTC output before an optional GitHub push.

### Relevant commits

- `6733550beac5fd81067e9804933af1144184cc60` — reject duplicate dataset UIDs
- `7b034bf75acddb2616ee82f2c26ad44954441a8e` — rebuild Colab training notebook with collision-safe UIDs
- `d54b198e7fb86a66fde0c58560731f294a3eda49` — duplicate-UID regression tests

### Status

**VERIFIED root cause:** dataset UID collision / feature overwrite in the previous Colab preparation path.

**VERIFIED code fix:** collision-safe UID generation + duplicate UID guard are present on `main`.

**NOT YET VERIFIED:** the newly trained checkpoint. A fresh training run must be executed from the rebuilt notebook before the new model can be called working.

### Next steps

1. Pull `main` in a fresh Colab runtime.
2. Run `notebooks/train_base_model_colab.ipynb` from the first cell.
3. Confirm the printed dataset integrity check passes with globally unique UIDs.
4. Confirm the 1-sample real-data CTC sanity gate passes.
5. Complete the full training run.
6. Require the post-training model acceptance gate to pass on both a training and held-out sample.
7. Push the checkpoint only after that gate passes.
8. Run `notebooks/validate_base_model_colab.ipynb` on multiple real videos.
9. Only then resume BridgeAdapter personalization work.

## 2026-08-28 — Hand-aware training-path audit

### Current repository state inspected

`main` is at `29a870afc088e005508174fd8367c102d4f57216`. The repository contains the hand-aware four-stream model, dataset/collator, training gate, React/Vite frontend, Render configuration, and the legacy `base_model.pt` artifact.

The legacy checkpoint remains intentionally incompatible with the new hand-aware loader and must not be promoted as a production model.

### Static findings

1. The production model constructor is trainable, while `load_frozen_base_model()` freezes only the inference instance as intended.
2. The training loop passes all four streams and explicit CTC input/target lengths.
3. The semantic gate already blocks empty, space-dominated, trivial, and high-CER outputs.
4. A regression test still used the pre-migration two-stream model signature. It has been aligned with the hand-aware four-stream contract.
5. The semantic gate test called `semantic_gate_failures()` without thresholds even though the function previously required them. Threshold defaults are now explicit and shared with the CLI defaults.
6. The overfit gate previously exposed loss and greedy output but did not reveal whether target characters were receiving probability mass or whether optimizer updates actually occurred. It now reports first-step gradient norms, first-step parameter delta, framewise blank/space argmax ratios, and target-character peak probability.
7. CI previously compiled/tests the backend but only syntax-checked JavaScript on the frontend. It now installs frontend dependencies and runs the TypeScript check plus Vite production build.

### Changes on audit branch

Branch: `agent/hand-aware-training-diagnostics`

PR: #3

No model checkpoint was generated, modified, or pushed.

### Verification status

**CODE FIXED:** hand-aware padding-mask regression test and semantic-gate threshold defaults.

**CODE FIXED:** diagnostic observability for gradient/update/CTC-collapse analysis.

**CODE FIXED:** CI now contains explicit frontend typecheck/build gates.

**STATIC VERIFIED:** current repository configuration and changed source files were inspected through GitHub.

**NOT VERIFIED:** backend pytest, frontend `npm run check`, frontend `npm run build`, and real-data hand-aware overfit could not be executed in this environment. The container cannot resolve `github.com`, and no real ISL processed dataset/GPU runtime is available here.

**NOT VERIFIED:** the hand-aware model is not production-ready and the legacy checkpoint remains rejected.

### Next required runtime evidence

Run the diagnostic overfit gate on a fresh GPU runtime. Interpret the new diagnostics before changing architecture:

- finite gradients + nonzero parameter delta + high target-character peaks + blank greedy path: investigate CTC decoding/alignment;
- finite gradients + nonzero parameter delta + low target-character peaks: investigate feature signal/model optimization;
- missing gradients or zero parameter delta: investigate graph/optimizer/trainability;
- invalid lengths or target alignment: stop at dataset/collation;
- only after these checks pass should full training be attempted.
