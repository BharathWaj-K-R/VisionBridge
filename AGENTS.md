# VisionBridge Multi-Agent Engineering Contract

This file is the shared handoff/state document for every AI agent working on VisionBridge.

Agents currently used by the team may include:
- ChatGPT
- Claude
- Codex
- Gemini/Antigravity when available
- Human developers

The purpose of this file is to prevent agents from repeating work, overwriting fixes, or assuming that a proposed change has already been implemented.

---

## 1. Project Goal

VisionBridge is a continuous Indian Sign Language (ISL) -> English translation system.

Current intended pipeline:

```text
Real ISL video
    -> MediaPipe Holistic
    -> pose + face landmarks
    -> VisionBridgeBaseModel
    -> CTC decoding
    -> English text
    -> BridgeAdapter personalization
```

The immediate priority is to make the frozen base model produce meaningful non-blank predictions on real ISL videos.

Personalization with `BridgeAdapter` comes only after the base model is demonstrably functional.

---

## 2. Non-Negotiable Engineering Rules

1. Read this file before changing anything.
2. Inspect the current `main` branch before relying on previous conversation context.
3. Never assume a proposed change is already implemented. Verify it in code.
4. Change only files required for the current task.
5. Do not rename notebooks.
6. Do not modify training notebooks while debugging the inference/check notebook unless the notebook itself is proven to be the cause.
7. Do not fabricate checkpoints, predictions, datasets, or test results.
8. Do not silently fall back to random model weights.
9. Preserve working changes made by other agents.
10. Before committing, run tests relevant to the changed code and inspect `git diff`.
11. Every agent must record what it changed and what remains unresolved in this file or in the commit message.
12. If two agents are working in parallel, do not overwrite another agent's changes. Rebase/merge or stop and report the conflict.

---

## 3. Current Repository State

Current branch: `main`

Latest known commit:

`f653c7266054b8ac3eff65699c38c94fd6770ef8`

Latest verified test result:

`30 passed, 0 failures`

The latest test fix corrected FastAPI lifespan initialization in the live-pipeline test. `Base.metadata.create_all()` now runs in the application lifespan, so the TestClient must enter the lifespan before tests that write database rows. The commit reports the full suite passing after this fix.

---

## 4. Verified Model Contract

The current base model is a pose + face fusion Transformer.

Input dimensions:

```text
POSE_INPUT_DIM = 132    # 33 landmarks x 4
FACE_INPUT_DIM = 1404   # 468 landmarks x 3
MAX_SEQUENCE_LENGTH = 1024
```

The model contains:

```text
Pose StreamEncoder
Face StreamEncoder
CrossModalFusion
Shared Transformer Encoder
Linear output head
CTC decoding
```

The model uses character-level tokenization in the current training pipeline.

CTC blank ID:

```text
0
```

The base model is intended to be frozen after pretraining. Signer-specific adaptation belongs in `BridgeAdapter`.

---

## 5. Verified Padding-Mask Fix

A previous failure showed the trained model producing:

```text
GROUND TRUTH: He is going into the room
PREDICTED:    (no sign detected)
CONFIDENCE:   0.0
BLANK RATIO:  1.0
NON-BLANK:    0
UNIQUE TOKENS: 0
LOGITS FINITE: True
```

This was classified as CTC blank collapse.

The training batches are padded to the longest sequence in each batch. The model was therefore changed to accept `input_lengths` and construct a padding mask.

The padding mask is now passed through:

- pose self-attention
- face self-attention
- pose-to-face cross-modal attention
- shared Transformer attention

`train_base_model.py` passes `input_lengths` into the model.

A regression test verifies that changes to padded frames cannot affect valid predictions.

Important: this fix is implemented in the current repository. Do not re-implement it unless inspection proves it was removed or broken.

---

## 6. Verified Training Pipeline

`backend/app/training/train_base_model.py` currently:

- loads `ISLTranslateKeypointDataset`
- uses `SimpleCharTokenizer`
- batches pose/face sequences with `collate_ctc_batch`
- passes true `input_lengths` to the model
- uses `torch.nn.CTCLoss(blank=0, zero_infinity=True)`
- uses AdamW
- clips gradients
- evaluates a validation split each epoch
- saves the best validation-loss model
- can optionally save resumable checkpoints via `--checkpoint-dir`
- supports `--resume`

Do not launch an expensive full training run until the tiny overfit sanity test passes.

---

## 7. Tiny Overfit Sanity Test

The repository contains:

`backend/app/training/overfit_sanity.py`

Purpose:

> Prove that the current model + dataset + tokenizer + CTC pipeline can learn a handful of real examples before full training is attempted.

Use real data. Do not use synthetic landmark arrays to claim model success.

Expected command:

```bash
PYTHONPATH=backend python -m app.training.overfit_sanity \
  --data-dir data/processed/isltranslate \
  --samples 4 \
  --steps 150
```

The important outputs are:

```text
Initial CTC loss
Final CTC loss
Loss reduction
Final blank ratio
Decoded predictions
OVERFIT SANITY: PASS/FAIL
```

If this test fails, diagnose the training/data/CTC pipeline before full training.

---

## 8. Current Checkpoint Status

The current checkpoint path is:

`backend/app/models/weights/base_model.pt`

Vocabulary:

`backend/app/models/weights/base_model.vocab.json`

The previously tested checkpoint produced 100% CTC blank output on a real video. Do not assume that checkpoint is usable merely because it loads successfully.

The current loader correctly treats a missing checkpoint as a deployment/configuration error rather than silently creating random weights.

A new checkpoint must be validated on real video before being called working.

---

## 9. Real-Video Check Notebook

Primary inference/check notebook:

`notebooks/train_base_model_colab_fixed.ipynb`

Despite its historical filename, this is the model-check/inference notebook for the current workflow.

It must remain an inference/check notebook.

It must not be turned into a training notebook.

The Colab workflow was hardened around MediaPipe compatibility and isolated Python 3.12 execution.

The real-video check extracts actual MediaPipe pose/face landmarks and runs the actual checkpoint.

---

## 10. MediaPipe / Colab History

Colab's Python/MediaPipe compatibility caused several failures during development.

The check notebook was changed to isolate MediaPipe in a Python 3.12 environment and avoid Matplotlib backend conflicts.

Do not undo this isolation without first proving that the current Colab runtime supports the required MediaPipe API.

The real extraction pipeline has already been demonstrated to run and produce real pose/face arrays.

---

## 11. Model Readiness and Backend Hardening

Recent Codex work added:

- missing-checkpoint protection
- model availability/readiness reporting
- stricter pose/face validation
- finite-value validation
- maximum inference frame validation
- user/adapter ownership validation
- calibration input validation
- improved `/health` information
- improved `/translate` error handling
- improved `/calibration` error handling
- absolute/configured backend paths
- SQLite foreign-key enforcement
- FastAPI lifespan-based DB initialization
- frontend API endpoint configuration
- Render configuration updates
- model availability tests

These are production/application concerns and should not be casually reverted while debugging the base model.

---

## 12. BridgeAdapter Design

`backend/app/models/bridge_adapter.py` contains the intended personalization mechanism.

Current design:

```text
Frozen VisionBridgeBaseModel
        |
        +-- small Houlsby-style bottleneck adapters
        |
        +-- signer-specific calibration
```

Target design characteristics:

- base model frozen
- adapter only is trained during personalization
- bottleneck dimension currently 16
- one adapter per shared Transformer layer
- CTC is used for sentence-level calibration
- target is <2% of base-model parameter count
- personalization should require only a small calibration set

Do not optimize or redesign BridgeAdapter until the base model produces meaningful predictions.

---

# 13. Multi-Agent Workflow

Every agent must follow this sequence:

### STEP A — Synchronize

```bash
git checkout main
git pull --ff-only
```

Then inspect `AGENTS.md` and `git log --oneline -10`.

### STEP B — Inspect

Before changing code:

- inspect the relevant files
- inspect the latest commits touching those files
- understand the current behavior
- identify the exact root cause or missing requirement

### STEP C — Declare scope

Before editing, state internally or in the task response:

```text
Files I intend to modify:
- ...

Why each file is necessary:
- ...
```

Do not modify unrelated files.

### STEP D — Implement the smallest correct change

Prefer a minimal fix over architecture rewrites.

### STEP E — Verify

Run focused tests first, then the full suite when practical:

```bash
python -m pytest backend/tests
```

For model work, also run the smallest meaningful real-data or overfit check.

### STEP F — Review diff

```bash
git status
git diff --stat
git diff
```

Confirm unrelated files are untouched.

### STEP G — Commit

Use a precise commit message such as:

```text
fix: correct CTC training padding mask
```

or:

```text
test: validate real-model CTC decoding
```

### STEP H — Update this handoff

If the work changes the project's verified state, update the appropriate section of this file in a separate commit or the same logically scoped commit.

Record:

- what was changed
- why
- tests run
- result
- remaining issue

---

# 14. Agent Responsibilities

## ChatGPT

Primary role:

- architecture/reasoning
- debugging strategy
- research verification
- cross-agent coordination
- deciding the next smallest experiment
- reviewing changes before expensive training

ChatGPT should not assume code was changed merely because a previous agent proposed it.

## Claude

Primary role:

- code review
- alternative implementation analysis
- test design
- finding edge cases
- reviewing Codex changes for regressions

Claude should preserve existing contracts unless a bug is proven.

## Codex

Primary role:

- direct repository implementation
- focused refactors/fixes
- tests
- commits

Codex must respect the file-scope rule and must not make broad unrelated cleanup changes during model debugging.

## Gemini / Antigravity

Primary role when used:

- Colab workflow
- dataset acquisition
- notebook execution/debugging
- external research

Any proposed repository change must still follow this file's change-control rules.

---

# 15. Communication Protocol Between Agents

When handing work to another agent, use this structure:

```text
TASK:
<one specific task>

CURRENT VERIFIED STATE:
<facts only>

CHANGES ALREADY IMPLEMENTED:
<files + behavior>

DO NOT TOUCH:
<protected files/areas>

EXPECTED RESULT:
<test or observable behavior>

COMMANDS TO RUN:
<exact commands>

HANDOFF:
<what the next agent should report>
```

Never hand off with vague statements such as "the model should be fixed now."

Use verified language:

- VERIFIED
- NOT VERIFIED
- PROPOSED
- BLOCKED

---

# 16. Current Roadmap

## Phase 0 — Repository synchronization

Status: COMPLETE

Current GitHub state has been inspected and the multi-agent contract is being established.

## Phase 1 — Base-model training sanity

Status: NEXT

Run the tiny real-data overfit test.

```text
4 real samples -> short training -> loss reduction -> non-blank predictions
```

Do not start full training until this passes.

## Phase 2 — Full base-model training

Status: WAITING FOR PHASE 1

Train on the selected ISLTranslate/iSign-derived processed keypoint dataset.

Use checkpoint/resume support for long Colab runs.

Monitor:

- train loss
- validation loss
- blank ratio
- non-blank token ratio
- decoded samples

## Phase 3 — Real-video evaluation

Status: WAITING FOR PHASE 2

Run the real-video check notebook.

Report:

- ground truth
- prediction
- CER
- edit distance
- blank ratio
- confidence

The model is not considered working merely because inference completes.

## Phase 4 — Base model quality gate

Status: WAITING

Require evidence that the model produces meaningful non-blank predictions on multiple real videos.

A single lucky prediction is not sufficient.

## Phase 5 — BridgeAdapter personalization

Status: BLOCKED UNTIL BASE MODEL PASSES

Validate:

- adapter parameter count
- <2% budget
- frozen base weights
- calibration loss reduction
- before/after CER
- inference latency

## Phase 6 — Backend integration

Status: PARTIALLY IMPLEMENTED

Connect the working base model and adapter through the existing APIs.

## Phase 7 — Render deployment

Status: PARTIALLY IMPLEMENTED

Validate:

- model file availability
- startup
- health endpoint
- CPU inference
- memory usage
- request validation

## Phase 8 — Demo

Status: FUTURE

Final demo flow:

```text
Signer video
    -> keypoint extraction
    -> base translation
    -> optional signer calibration
    -> personalized translation
    -> English output
```

---

# 17. Immediate Next Steps

1. Pull current `main`.
2. Confirm the padding-mask implementation is still present.
3. Run the tiny overfit sanity test on real processed data.
4. If PASS: start a controlled full training run.
5. If FAIL: diagnose the exact CTC/data/model failure before retraining.
6. Validate the new checkpoint on multiple real videos.
7. Only after the base model passes, continue BridgeAdapter personalization.
8. Run the complete backend test suite after model/API changes.
9. Keep all changes narrowly scoped and documented.

---

# 18. Current Decision Gate

**DO NOT FULL-RETRAIN YET unless the tiny overfit sanity test passes.**

Reason: the previous real checkpoint produced 100% CTC blank predictions. The padding-mask correction is now implemented, but its ability to restore learning has not yet been proven by the required tiny overfit experiment.

The next meaningful evidence is the output of:

```bash
PYTHONPATH=backend python -m app.training.overfit_sanity \
  --data-dir data/processed/isltranslate \
  --samples 4 \
  --steps 150
```

---

# 19. Change Log

### 2026-08-25 — Multi-agent handoff initialized

- Inspected current `main` branch.
- Confirmed latest test-fix commit `f653c726...` reports 30 passing tests.
- Confirmed padding-mask fix exists in the current base model/training pipeline.
- Confirmed tiny CTC overfit sanity tooling exists.
- Confirmed model readiness and stricter API validation changes are present.
- Established this `AGENTS.md` as the shared coordination contract.
- Next gate: tiny real-data overfit test before full retraining.

### Previous verified work

- Real-video Colab inference/check workflow established.
- MediaPipe Colab compatibility isolated to Python 3.12.
- Real pose/face extraction verified.
- Multi-sample CTC blank diagnostic added.
- CTC blank collapse observed on the previous checkpoint.
- Padding masks added to Transformer attention.
- Sequence lengths passed from training batches into the model.
- Regression test added for padded-frame invariance.
- Tiny CTC overfit sanity test added.
- Missing checkpoint no longer falls back to random model weights.
- Backend model readiness and request validation hardened.
- FastAPI DB initialization moved to lifespan.
- Test suite corrected for lifespan behavior; latest reported result: 30 passed.

---

## Golden Rule

**No agent gets to say "fixed" because the code looks reasonable. A fix is only VERIFIED after the relevant test or real-data experiment passes.**
