# VisionBridge Multi-Agent Engineering Contract

This file is the shared source of truth for ChatGPT, Claude, Codex, Gemini/Antigravity, and human developers working on VisionBridge.

## 0. State-update rule: MANDATORY

**Every meaningful step must update this file.** A meaningful step includes:
- a code/notebook change
- a test run
- a training run
- a validation run
- a new error/failure
- a root-cause discovery
- a workaround
- a decision to change or preserve architecture
- a blocked step
- a completed milestone

When an issue occurs, update `Current State`, `Problem Analysis`, `Active Fix`, and `Next Step` in the same logical change/commit whenever possible.

Never write only "fixed". Record evidence.

Every state entry must use one of:
- `VERIFIED` = directly demonstrated by code/test/run
- `NOT VERIFIED` = intended or suspected but not demonstrated
- `BLOCKED` = cannot proceed until another condition is satisfied
- `FAILED` = attempted and failed

Each issue entry must contain:
1. Symptom / exact error
2. Evidence
3. Root-cause analysis
4. Files affected or suspected
5. Fix applied or proposed
6. Verification result
7. Next action

## 1. Project Goal

VisionBridge is a continuous Indian Sign Language (ISL) -> English translation system.

```text
Real ISL video
  -> MediaPipe Holistic
  -> pose + face landmarks
  -> VisionBridgeBaseModel
  -> CTC decoding
  -> English text
  -> BridgeAdapter personalization
```

Immediate priority: make the frozen base model produce meaningful non-blank predictions on real ISL videos. BridgeAdapter work is blocked until the base model passes real-video quality checks.

## 2. Non-negotiable rules

1. Read this file before changing anything.
2. Sync and inspect current `main` before relying on chat history.
3. Verify proposed changes in code. Never assume they exist.
4. Modify only files required for the current task.
5. Do not rename notebooks.
6. Do not fabricate datasets, checkpoints, predictions, metrics, or test results.
7. Do not silently fall back to random weights.
8. Preserve other agents' verified fixes.
9. Before commit, inspect diff and run relevant tests.
10. Record every meaningful state transition in this file.
11. If another agent has changed the same area, inspect first and avoid overwriting it.
12. Expensive training is not allowed until the current acceptance gate passes.

## 3. Current repository state

Branch: `main`

Latest known model/debug work: dataset UID collision fix and training-notebook rebuild.

Latest verified historical backend test result: `30 passed, 0 failures`.

Current model checkpoint exists at:

```text
backend/app/models/weights/base_model.pt
backend/app/models/weights/base_model.vocab.json
```

The checkpoint previously tested is **NOT VERIFIED as usable**. It produced 100% blank CTC output on both an unseen real video and a known training video.

## 4. Verified model contract

```text
POSE_INPUT_DIM = 132     # 33 landmarks x 4
FACE_INPUT_DIM = 1404    # 468 landmarks x 3
MAX_SEQUENCE_LENGTH = 1024
CTC_BLANK_ID = 0
```

Architecture:

```text
Pose StreamEncoder
Face StreamEncoder
CrossModalFusion
Shared Transformer Encoder
Linear character head
CTC objective / greedy decode
```

The base model is frozen at inference. Personalization belongs in `BridgeAdapter`.

## 5. Verified padding-mask work

`backend/app/models/base_model.py` accepts `lengths` and constructs a padding mask. The mask is applied through pose self-attention, face self-attention, cross-modal attention, and shared Transformer attention.

`backend/app/training/train_base_model.py` passes `input_lengths` into the model and CTCLoss.

Status: **VERIFIED IN CODE, NOT SUFFICIENT TO PROVE MODEL QUALITY**.

Do not reimplement this unless inspection proves it was removed or broken.

## 6. Verified dataset/training code

`backend/app/training/isltranslate.py`:
- loads pose and face `.npy`
- uses `SimpleCharTokenizer`
- reserves token `0` for CTC blank
- downsamples clips over 1024 frames
- pads batches and records true `input_lengths`

`backend/app/training/train_base_model.py`:
- AdamW
- `CTCLoss(blank=0, zero_infinity=True)`
- gradient clipping
- validation split
- resumable checkpoints

`backend/app/training/overfit_sanity.py`:
- real-data single-sample extreme-overfit test
- checks loss reduction and non-blank decoding

## 7. Root cause discovered on 2026-08-25

### Problem

The ISL-CSLTR Kaggle dataset contains repeated filenames in different sentence folders, for example:

```text
.../i am suffering from fever/fever (2).MP4
.../<other label>/fever (2).MP4
```

The old training notebook used `Path(video).stem` as the UID. That is **not globally unique**.

### Consequence

Different videos could write to the same:

```text
pose/<uid>.npy
face/<uid>.npy
```

later extractions could overwrite earlier features, and the manifest could pair the wrong features with the wrong English label.

This is a valid data-corruption path that can explain CTC collapse and garbage learning.

### Evidence

The real runtime showed the notebook selecting `fever (2).MP4`, extracting real `[72,132]` pose and `[72,1404]` face, yet the committed checkpoint still returned 100% blank on that known training example.

The same checkpoint returned 100% blank on the automatically selected `good (3).MP4` validation video.

### Fix status

`notebooks/train_base_model_colab.ipynb` has been rebuilt to derive globally unique clip IDs from sentence label + filename + relative-path hash, rebuild processed data from scratch, detect UID collisions, and validate dataset integrity.

Status: **IMPLEMENTED, NOT YET VERIFIED BY A CLEAN FULL TRAINING RUN**.

A focused dataset-integrity regression test was also added during this work.

## 8. Validation notebook state

Primary validation notebook:

```text
notebooks/validate_base_model_colab.ipynb
```

It automatically:
- downloads ISL-CSLTR via Kaggle
- selects one real sentence-level video
- derives ground truth from the sentence folder
- extracts real MediaPipe 132/1404 features
- loads the committed checkpoint
- prints prediction, confidence, blank ratio, non-blank frames, unique tokens, CER

Important environment behavior:
- Colab runtime may be CPU even when a GPU was expected
- MediaPipe is isolated in Python 3.12 with `mediapipe==0.10.21`
- Kaggle filenames use `.MP4`; the repository extractor historically expected `.mp4`, so notebook/runtime code may need a case-normalized copy. This is a notebook/data-ingestion issue, not a model conclusion.

## 9. Failed model evidence

### Unseen real video

```text
GROUND TRUTH:     you are good
PREDICTED:        (no sign detected)
BLANK RATIO:      1.0000
NON-BLANK FRAMES: 0
CONFIDENCE:       0.836
LOGITS FINITE:    True
```

### Known training video

```text
GROUND TRUTH:     i am suffering from fever
PREDICTED:        (no sign detected)
BLANK RATIO:      1.0000
NON-BLANK FRAMES: 0
CONFIDENCE:       0.836
LOGITS FINITE:    True
```

Interpretation: this is **not merely a generalization failure**. The current checkpoint cannot demonstrate useful decoding on a known real training example.

## 10. Current problem analysis

### Primary hypothesis: supervised-data corruption
Status: `SUPPORTED / FIXED IN NOTEBOOK, NOT YET VERIFIED`

Duplicate UID handling was a genuine defect in dataset preparation and has now been addressed in the new training notebook.

### Secondary hypotheses to verify after a clean rebuild

1. Blank-index consistency across tokenizer, target encoding, CTCLoss, and decoder.
2. Target labels are non-empty and correctly tokenized.
3. Training and inference preprocessing are identical.
4. The new globally unique manifest maps each UID to the correct video and text.
5. The model can escape the blank solution on one clean real example.
6. The checkpoint vocabulary and output-head dimension match.
7. Only after the above pass should architecture changes be considered.

### Architecture decision

**KEEP the current Transformer for now.** No evidence currently justifies replacing it. Fix and verify data/CTC contracts first.

## 11. Current acceptance gates

### Gate A — dataset integrity

Must show:

```text
unique UIDs
non-empty labels
pose [T,132]
face [T,1404]
matching frame counts
```

Status: `NOT VERIFIED ON CLEAN REBUILD`

### Gate B — one-sample CTC sanity

Must show:
- finite final loss
- substantial loss reduction
- at least one decoded non-blank token
- consistent blank statistics

Status: `NOT VERIFIED ON CLEAN REBUILD`

### Gate C — full training

Only after Gate B passes.

Monitor:
- train loss
- validation loss
- blank ratio
- non-blank ratio
- decoded validation samples
- CER/WER

Status: `BLOCKED`

### Gate D — real-video evaluation

Must test multiple real clips and report:
- ground truth
- prediction
- CER
- blank ratio
- confidence
- empty-prediction rate

Status: `BLOCKED UNTIL NEW CHECKPOINT`

### Gate E — BridgeAdapter

Status: `BLOCKED UNTIL BASE MODEL PASSES REAL-VIDEO QUALITY GATE`.

## 12. Mandatory issue-tracking template

Every new issue added to this file must use:

```text
### YYYY-MM-DD — <short issue name>
Status: FAILED | BLOCKED | FIXED-NOT-VERIFIED | VERIFIED

Symptom:
<exact error/output>

Evidence:
<commands, logs, file paths, test results>

Root-cause analysis:
<what the evidence supports; distinguish hypothesis from fact>

Files involved:
- ...

Fix:
<what was changed or why no code was changed>

Verification:
<exact test/run and result>

Next step:
<one concrete action>
```

## 13. Mandatory step-state template

At each workflow step, append a short entry:

```text
### STEP STATE — YYYY-MM-DD HH:MM
Step:
Status:
What changed:
What was verified:
What failed / remains unknown:
Next step:
```

Do this even when the step succeeds. Success without state recording is how multi-agent teams rediscover the same bug.

## 14. Multi-agent workflow

### Before work

```bash
git checkout main
git pull --ff-only
git log --oneline -10
```

Read `AGENTS.md`.

### Before edits

Identify:
- exact files to modify
- why each is required
- protected files
- expected verification

### After edits

Run focused tests, then:

```bash
python -m pytest backend/tests
```

For model work, run the smallest meaningful real-data experiment.

Then inspect:

```bash
git status
git diff --stat
git diff
```

Commit narrowly.

Immediately update `AGENTS.md` with the new verified state and next action.

## 15. Agent handoff format

```text
TASK:
<one specific task>

CURRENT VERIFIED STATE:
<facts only>

LATEST ISSUE:
<exact symptom if any>

ROOT-CAUSE ANALYSIS:
<verified facts + hypotheses clearly separated>

CHANGES ALREADY IMPLEMENTED:
<files + behavior>

DO NOT TOUCH:
<protected files/areas>

EXPECTED RESULT:
<observable behavior>

COMMANDS:
<exact commands>

HANDOFF:
<what the next agent must run/report>
```

Use `VERIFIED`, `NOT VERIFIED`, `FAILED`, and `BLOCKED` explicitly.

## 16. Agent roles

### ChatGPT
Architecture, root-cause reasoning, experiment design, cross-agent coordination, deciding the next smallest test.

### Claude
Independent code review, alternative diagnosis, test design, edge-case review.

### Codex
Focused repository edits, tests, commits. Must respect scope.

### Gemini/Antigravity
Notebook/Colab execution, dataset acquisition, runtime debugging, external research.

## 17. Roadmap

### Phase 0 — Repository synchronization
Status: COMPLETE

### Phase 1 — Clean dataset rebuild and sanity
Status: ACTIVE

Rebuild processed data with globally unique UIDs and prove the single-sample CTC gate can escape blank collapse.

### Phase 2 — Full base-model training
Status: BLOCKED BY PHASE 1

### Phase 3 — Multi-video real inference
Status: BLOCKED BY PHASE 2

### Phase 4 — Base-model quality gate
Status: BLOCKED

### Phase 5 — BridgeAdapter personalization
Status: BLOCKED UNTIL BASE MODEL PASSES

### Phase 6 — Backend integration
Status: PARTIALLY IMPLEMENTED

### Phase 7 — Render deployment
Status: PARTIALLY IMPLEMENTED

### Phase 8 — Demo
Status: FUTURE

## 18. Current next steps

1. Start a **fresh Colab runtime**.
2. Run `notebooks/train_base_model_colab.ipynb` from the first cell.
3. Confirm the rebuilt manifest has globally unique UIDs and no collisions.
4. Confirm `DATASET INTEGRITY: PASS`.
5. Run the one-sample CTC gate.
6. If Gate B fails, stop. Do not full-train. Record the exact output here and inspect tokenizer/targets/blank-index/preprocessing contracts.
7. If Gate B passes, run controlled full training.
8. Validate the new checkpoint on one known training clip and multiple held-out real clips.
9. Only after real-video quality is demonstrated, resume BridgeAdapter work.

## 19. Change log

### 2026-08-25 — Duplicate-UID root cause discovered

- Real Kaggle dataset contains duplicate filename stems across sentence folders.
- Old training notebook used filename stem as UID.
- This could overwrite pose/face `.npy` files and pair wrong features with text.
- Rebuilt training notebook to generate globally unique UIDs from label + stem + relative-path hash.
- Added dataset integrity protection.
- Added `MODEL_DEBUG_LOG.md` with the diagnosis.
- Existing checkpoint remains **NOT VERIFIED**.

### 2026-08-25 — Real-video validation failure confirmed

- Extracted `you are good` to pose `(69,132)` and face `(69,1404)`.
- Model produced 100% blank output.
- Extracted known `i am suffering from fever` training clip to pose `(72,132)` and face `(72,1404)`.
- Same checkpoint again produced 100% blank output.
- This ruled out simple unseen-video generalization as the only explanation.

### 2026-08-25 — Multi-agent state protocol strengthened

- Every meaningful step must update this file.
- Every issue must record symptom, evidence, root cause, fix, verification, and next step.
- Every successful step must also record what is now VERIFIED and what remains unknown.
- Full training remains blocked until the clean-data single-sample CTC gate passes.

## Golden rule

**A change is not VERIFIED because the code looks correct. It is VERIFIED only when the relevant test or real-data experiment passes. Every step and every issue must leave a written state in this file.**
