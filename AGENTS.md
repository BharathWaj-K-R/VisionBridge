# VisionBridge Multi-Agent Engineering Contract

This file is the shared source of truth for ChatGPT, Claude, Codex, Gemini/Antigravity, and human developers working on VisionBridge.

## 0. State-update rule: MANDATORY

Every meaningful step must update this file, including code/notebook changes, tests, training, validation, errors, root-cause discoveries, workarounds, decisions, blocked steps, and completed milestones.

Every issue entry must contain symptom, evidence, root-cause analysis, files, fix, verification, and next action. Use `VERIFIED`, `NOT VERIFIED`, `FAILED`, and `BLOCKED` explicitly.

## 1. Project Goal

VisionBridge is a continuous Indian Sign Language (ISL) -> English translation system.

```text
Real ISL video -> MediaPipe Holistic -> pose + face landmarks -> VisionBridgeBaseModel -> CTC decoding -> English text -> BridgeAdapter personalization
```

Immediate priority: make the frozen base model produce meaningful non-blank predictions on real ISL videos. BridgeAdapter work is blocked until the base model passes real-video quality checks.

## 2. Non-negotiable rules

1. Read this file before changing anything.
2. Sync and inspect current `main` before relying on chat history.
3. Verify proposed changes in code; never assume they exist.
4. Modify only files required for the current task.
5. Do not rename notebooks.
6. Do not fabricate datasets, checkpoints, predictions, metrics, or test results.
7. Do not silently fall back to random weights.
8. Preserve other agents' verified fixes.
9. Before commit, inspect diff and run relevant tests.
10. Record every meaningful state transition here.
11. Expensive training is blocked until the current acceptance gate passes.

## 3. Current repository state

Branch: `main`.

Historical verified backend result: `30 passed, 0 failures`. This is historical, not a verification of the latest changes because this environment cannot execute the repository test suite.

Current checkpoint files exist in the repository, but the tested checkpoint is **NOT VERIFIED as usable**.

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

## 5. Padding-mask status

`backend/app/models/base_model.py` accepts `lengths` and applies padding masks through pose self-attention, face self-attention, cross-modal attention, and shared Transformer attention.

`backend/app/training/train_base_model.py` passes `input_lengths` into the model and CTCLoss.

Status: **VERIFIED IN CODE; MODEL QUALITY NOT VERIFIED.**

## 6. Dataset/training status

`backend/app/training/isltranslate.py` loads pose/face arrays, uses `SimpleCharTokenizer`, reserves token 0 for CTC blank, downsamples over-length clips, pads batches, and records true input lengths. It now rejects duplicate UIDs and empty encodable targets.

`backend/app/training/train_base_model.py` uses AdamW, CTCLoss(blank=0, zero_infinity=True), gradient clipping, validation, checkpoint/resume, and now deterministic train/validation splitting with `--seed` defaulting to 42.

`backend/app/training/overfit_sanity.py` runs a real-data one-sample CTC sanity test and rejects all-blank decoding.

## 7. Root cause discovered: duplicate dataset UIDs

### Symptom

Real ISL-CSLTR contains repeated filename stems in different sentence folders, such as `fever (2).MP4`.

### Root cause

The old training notebook used `Path(video).stem` as the UID. That is not globally unique.

### Consequence

Different clips could overwrite the same pose/face `.npy` files and create wrong feature/label pairings, contaminating supervision and plausibly producing CTC collapse.

### Fix

`notebooks/train_base_model_colab.ipynb` now rebuilds the processed dataset from scratch and derives a UID from sentence label + filename stem + relative-path hash. It explicitly checks for collisions.

Status: **IMPLEMENTED, NOT VERIFIED BY A CLEAN TRAINING RUN.**

## 8. Real-video failure evidence

Previously validated on real MediaPipe outputs:

```text
you are good -> pose (69,132), face (69,1404) -> prediction empty -> blank ratio 1.0

i am suffering from fever -> pose (72,132), face (72,1404) -> prediction empty -> blank ratio 1.0
```

Both used the same prior checkpoint. Therefore the failure was not only an unseen-video generalization issue.

## 9. Deep static audit findings: additional loose ends

### Issue A — evaluator decoder vocabulary was not initialized

**Status: FIXED, NOT VERIFIED.**

`backend/app/training/evaluate.py` loaded the checkpoint and called the shared `decode_logits()` directly, but that decoder depends on module-level `_id_to_token`, which is initialized by `get_base_model()`. The evaluator bypassed `get_base_model()`, so decoded output could become raw `<id>` placeholders instead of characters.

Fix: evaluator now loads the checkpoint-adjacent vocabulary, verifies vocabulary size equals the model output head, assigns the decoder vocabulary, and refuses an empty evaluation dataset.

### Issue B — training notebook acceptance split did not necessarily match training split

**Status: FIXED, NOT VERIFIED.**

The training script previously used an unseeded `random_split()`. The notebook later reconstructed a seeded split, so the sample labeled `TRAIN` in the acceptance gate was not guaranteed to have been used for training.

Fix: `train_base_model.py` now accepts `--seed`, seeds Python/Torch, and uses a seeded `random_split`. The training notebook passes `--seed 42` and reconstructs the same split for acceptance.

### Issue C — notebook acceptance decoder bypassed vocabulary initialization

**Status: FIXED, NOT VERIFIED.**

The notebook directly loaded the model and then called `decode_logits()` without initializing the decoder vocabulary.

Fix: notebook acceptance now initializes `inference_service._id_to_token` from the saved vocabulary and checks output-head/vocabulary size compatibility before decoding.

## 10. Secondary hypotheses still to verify after a clean rebuild

1. Blank-index consistency across tokenizer, target encoding, CTCLoss, and decoder.
2. Correct target text/token encoding for every sample.
3. Identical preprocessing during training and inference.
4. Clean globally unique manifest with correct video/text pairing.
5. One-sample CTC test escapes the blank solution.
6. Checkpoint vocabulary and output-head dimension match.
7. Architecture changes are considered only after these checks pass.

## 11. Acceptance gates

### Gate A — dataset integrity

Must show unique UIDs, non-empty labels, pose `[T,132]`, face `[T,1404]`, matching frame counts, and no collisions.

Status: **NOT VERIFIED ON CLEAN REBUILD.**

### Gate B — one-sample CTC sanity

Must show finite loss, substantial loss reduction, at least one decoded non-blank token, and consistent blank statistics.

Status: **NOT VERIFIED ON CLEAN REBUILD.**

### Gate C — full training

Blocked until Gate B passes. Monitor train/validation loss, blank ratio, non-blank ratio, decoded predictions, and CER/WER.

### Gate D — real-video evaluation

Blocked until a new checkpoint passes Gate C and must cover multiple real clips.

### Gate E — BridgeAdapter

Blocked until the base model passes the real-video quality gate.

## 12. Mandatory issue template

```text
### YYYY-MM-DD — <short issue name>
Status: FAILED | BLOCKED | FIXED-NOT-VERIFIED | VERIFIED

Symptom:
...

Evidence:
...

Root-cause analysis:
...

Files involved:
- ...

Fix:
...

Verification:
...

Next step:
...
```

## 13. Mandatory step-state template

```text
### STEP STATE — YYYY-MM-DD HH:MM
Step:
Status:
What changed:
What was verified:
What failed / remains unknown:
Next step:
```

## 14. Multi-agent workflow

Before work:

```bash
git checkout main
git pull --ff-only
git log --oneline -10
```

After edits, run focused tests, then the full suite when execution is available:

```bash
python -m pytest backend/tests
```

For model work, run the smallest meaningful real-data experiment. Inspect `git status`, `git diff --stat`, and `git diff` before committing. Update this file immediately after a meaningful state transition.

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

## 16. Agent roles

ChatGPT: architecture, root-cause analysis, experiment design, coordination.

Claude: independent code review, alternative diagnosis, test/edge-case review.

Codex: focused repository implementation, tests, commits.

Gemini/Antigravity: notebook/Colab execution, dataset acquisition, runtime debugging, external research.

## 17. Roadmap

Phase 0 — Repository synchronization: COMPLETE.

Phase 1 — Clean dataset rebuild and CTC sanity: ACTIVE.

Phase 2 — Full base-model training: BLOCKED BY PHASE 1.

Phase 3 — Multi-video real inference: BLOCKED.

Phase 4 — Base-model quality gate: BLOCKED.

Phase 5 — BridgeAdapter personalization: BLOCKED UNTIL BASE MODEL PASSES.

Phase 6 — Backend integration: PARTIALLY IMPLEMENTED.

Phase 7 — Render deployment: PARTIALLY IMPLEMENTED.

Phase 8 — Demo: FUTURE.

## 18. Current next steps

1. Start a fresh Colab runtime.
2. Run `notebooks/train_base_model_colab.ipynb` from the first cell.
3. Confirm unique UIDs and dataset integrity.
4. Run the one-sample CTC gate.
5. If Gate B fails, stop and record the exact result here.
6. If Gate B passes, run controlled full training.
7. Validate the new checkpoint with `backend/app/training/evaluate.py` and the real-video notebook.
8. Only after the base model passes multiple real clips, resume BridgeAdapter work.

## 19. Change log

### 2026-08-25 — Duplicate-UID root cause discovered

- Repeated filename stems across sentence folders were confirmed.
- Old filename-stem UID logic could overwrite features and mispair labels.
- Collision-safe training notebook implemented.

### 2026-08-25 — Real-video model failure confirmed

- `you are good`: real pose `(69,132)`, face `(69,1404)`, 100% blank output.
- `i am suffering from fever`: real pose `(72,132)`, face `(72,1404)`, 100% blank output.
- Current checkpoint remains unverified.

### 2026-08-25 — Deep audit: evaluator and split-consistency defects found

- Evaluator could call `decode_logits()` before loading the saved vocabulary, producing raw token IDs.
- Training notebook's post-training `TRAIN`/`VAL` split was not guaranteed to match the training split because the training script used unseeded `random_split()`.
- Both were fixed: evaluator initializes and validates the vocabulary; training is deterministic via `--seed 42`, and the notebook uses the same split seed.
- No repository test suite has been executed in this environment after these changes. These fixes are therefore **NOT VERIFIED** by runtime tests yet.

## Golden rule

A change is not VERIFIED because the code looks reasonable. It is VERIFIED only when the relevant test or real-data experiment passes. Every step and every issue must leave a written state here.
