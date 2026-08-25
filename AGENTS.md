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

Historical verified backend result: `30 passed, 0 failures`. This predates the latest autonomous audit changes and is **not** current verification.

Current checkpoint files exist in the repository, but the tested checkpoint is **NOT VERIFIED as usable**.

GitHub connector has read/write access. Direct shell clone was attempted and failed because this execution environment cannot resolve `github.com`; therefore local pytest/build/runtime execution is **BLOCKED** in this agent environment.

## 4. Verified model contract

```text
POSE_INPUT_DIM = 132
FACE_INPUT_DIM = 1404
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

Base model is frozen at inference. Personalization belongs in `BridgeAdapter`.

## 5. Padding-mask status

`backend/app/models/base_model.py` accepts `lengths` and applies padding masks through pose self-attention, face self-attention, cross-modal attention, and shared Transformer attention.

`backend/app/training/train_base_model.py` passes `input_lengths` into the model and CTCLoss.

Status: **VERIFIED IN CODE; MODEL QUALITY NOT VERIFIED.**

## 6. Dataset and CTC contract status

`backend/app/training/isltranslate.py` enforces:
- unique manifest UIDs
- pose shape `(frames,132)`
- face shape `(frames,1404)`
- pose/face frame alignment
- non-empty/encodable targets
- finite feature values
- CTC target length not greater than post-downsampled input frames
- batch feature dimensions matching the model contract

`backend/app/training/train_base_model.py` uses:
- AdamW
- `CTCLoss(blank=0, zero_infinity=True)`
- gradient clipping
- deterministic train/validation split with `--seed` (default 42)
- resumable checkpoints

Status: **IMPLEMENTED, NOT RUNTIME-VERIFIED AFTER LATEST CHANGES.**

## 7. Root cause discovered: duplicate dataset UIDs

### Symptom

Real ISL-CSLTR contains repeated filename stems in different sentence folders, such as `fever (2).MP4`.

### Root cause

Old training notebook used `Path(video).stem` as UID. That was not globally unique.

### Consequence

Different clips could overwrite the same pose/face `.npy` files and pair wrong features with labels.

### Fix

`notebooks/train_base_model_colab.ipynb` now rebuilds processed data from scratch and derives a UID from sentence label + filename stem + relative-path hash.

Status: **IMPLEMENTED, NOT VERIFIED BY A CLEAN TRAINING RUN.**

## 8. Deep-audit issue: evaluator/notebook decoder state

### Symptom

`decode_logits()` depends on module-level `_id_to_token`. Direct callers that load a model themselves can bypass `get_base_model()`, leaving the decoder vocabulary empty and producing `<id>` placeholders.

### Fix

`backend/app/training/evaluate.py` loads the checkpoint-adjacent vocabulary, validates output-head/vocabulary size equality, and initializes the shared decoder map.

The training notebook acceptance cell does the same before decoding its train/validation acceptance examples.

Status: **FIXED, NOT RUNTIME-VERIFIED.**

## 9. Deep-audit issue: training/acceptance split mismatch

### Symptom

The training script previously used unseeded `random_split()`, while the notebook acceptance gate reconstructed a seeded split. A sample labeled `TRAIN` was not guaranteed to have been seen during training.

### Fix

`train_base_model.py` seeds Python/Torch and uses a seeded `random_split` with `--seed 42` by default. The training notebook passes `--seed 42` and reconstructs the exact same split for acceptance.

Status: **FIXED, NOT RUNTIME-VERIFIED.**

## 10. Real-video failure evidence

Previous checkpoint results:

```text
you are good -> pose (69,132), face (69,1404) -> empty prediction -> blank ratio 1.0

i am suffering from fever -> pose (72,132), face (72,1404) -> empty prediction -> blank ratio 1.0
```

The old checkpoint is not accepted as a working model.

## 11. Newly discovered issue: standalone raw-video extension handling

### Status
FIXED-NOT-VERIFIED

### Symptom

Real Kaggle files use uppercase extensions such as `good (3).MP4` and `fever (2).MP4`, while the standalone extractor constructed only `<uid>.mp4` paths. This produced the previously observed false `missing video` errors.

### Evidence

Historical runtime output reported missing `.mp4` while the actual dataset contained `.MP4` files.

### Root-cause analysis

Filename extension comparison was case-sensitive and only considered the literal `.mp4` path.

### Files involved

- `backend/scripts/extract_keypoints.py`
- `backend/tests/test_feature_contract_scripts.py`

### Fix

Added `resolve_video_path()` with case-insensitive extension matching for `.mp4`, `.avi`, `.mov`, and `.mkv`. The extractor now resolves by filename stem instead of assuming lowercase `.mp4`.

Added regression tests for uppercase `.MP4` and unsupported extensions.

### Verification

Code inspection PASS. Runtime test execution BLOCKED because direct GitHub clone failed in this agent environment due DNS/network resolution failure.

### Next action

Run `python -m pytest backend/tests` in Colab/local CI and require PASS.

## 12. Newly discovered issue: stale 1434-face converter contract

### Status
FIXED-NOT-VERIFIED

### Symptom

`backend/scripts/convert_isign_pose.py` documented and emitted `1434` face features while the VisionBridge model contract is `1404`.

### Root-cause analysis

The converter had retained a 478-landmark face assumption while the deployed/training model uses legacy MediaPipe Holistic's 468 face landmarks.

### Impact

A user could run the converter successfully, generate `.npy` files, and only discover the dimensional mismatch later inside dataset/model loading. This violates the fail-fast data contract.

### Files involved

- `backend/scripts/convert_isign_pose.py`
- `backend/tests/test_feature_contract_scripts.py`

### Fix

Changed the converter contract to `EXPECTED_FACE_DIM = 1404`, made missing components fail immediately, made incompatible pose/face shapes raise instead of warn, added frame/finite-value checks, and prevented a zero-success conversion from being reported as success.

### Verification

Static inspection PASS. Runtime test execution BLOCKED in this agent environment.

### Next action

Run the new script tests plus existing backend tests in Colab/CI.

## 13. Security/configuration audit status

### Observed

`backend/app/core/config.py` correctly rejects the development `SECRET_KEY` when `ENV=production`.

`backend/app/main.py` invokes `settings.validate_for_runtime()` in application lifespan before creating tables.

### Remaining warning

CORS and database settings are environment-driven, but production deployment correctness depends on actual Render environment values, mounted storage, and runtime behavior. Those external settings were not directly verified here.

Status: **WARNING / NOT TESTED**.

## 14. Production API boundary

The `/translate` API currently accepts pre-extracted pose/face keypoints and validates the 132/1404 contract. It does **not** decode raw video itself.

Therefore:

```text
Colab raw-video validation -> supported
Production raw-video HTTP upload -> NOT IMPLEMENTED IN THIS API
```

This is an architectural boundary, not a model-training bug. A complete raw-video production demo still requires a video -> MediaPipe extraction layer before `/translate` or a dedicated endpoint/service.

Status: **PARTIAL / BLOCKED FOR FULL RAW-VIDEO PRODUCTION FLOW.**

## 15. Remaining hypotheses after clean-data rebuild

1. Verify blank-index consistency end-to-end.
2. Verify targets are semantically correct and non-empty.
3. Verify train/inference preprocessing equivalence.
4. Verify no remaining data/label corruption.
5. Prove the model can escape the blank solution on one clean sample.
6. Prove checkpoint/vocabulary compatibility.
7. Only then consider architecture changes.

## 16. Acceptance gates

### Gate A — dataset integrity

Unique UIDs, non-empty targets, `(T,132)`, `(T,1404)`, matching frames, finite values, no target longer than input.

Status: **NOT VERIFIED ON CLEAN REBUILD.**

### Gate B — one-sample CTC sanity

Finite loss, meaningful loss reduction, non-blank decoded token(s), consistent blank statistics.

Status: **NOT VERIFIED ON CLEAN REBUILD.**

### Gate C — full training

Blocked until Gate B passes. Monitor train/validation loss, blank ratio, non-blank ratio, decoded predictions, CER/WER.

### Gate D — real-video evaluation

Blocked until a new checkpoint is trained and must cover multiple clips.

### Gate E — BridgeAdapter

Blocked until the base model passes the real-video quality gate.

## 17. Mandatory issue template

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

## 18. Mandatory step-state template

```text
### STEP STATE — YYYY-MM-DD HH:MM
Step:
Status:
What changed:
What was verified:
What failed / remains unknown:
Next step:
```

## 19. Multi-agent workflow

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

Inspect `git status`, `git diff --stat`, and `git diff` before commit. Update this file after every meaningful state transition.

## 20. Agent roles

ChatGPT: architecture, root-cause analysis, experiment design, coordination.

Claude: independent code review, alternative diagnosis, test/edge-case review.

Codex: focused repository implementation, tests, commits.

Gemini/Antigravity: notebook/Colab execution, dataset acquisition, runtime debugging, external research.

## 21. Roadmap

Phase 0 — Repository synchronization: COMPLETE.

Phase 1 — Clean dataset rebuild and CTC sanity: ACTIVE.

Phase 2 — Full base-model training: BLOCKED BY PHASE 1.

Phase 3 — Multi-video real inference: BLOCKED.

Phase 4 — Base-model quality gate: BLOCKED.

Phase 5 — BridgeAdapter personalization: BLOCKED UNTIL BASE MODEL PASSES.

Phase 6 — Backend integration: PARTIALLY IMPLEMENTED.

Phase 7 — Render deployment: PARTIALLY IMPLEMENTED.

Phase 8 — Demo: FUTURE.

## 22. Current next steps

1. Start a fresh Colab runtime.
2. Run `notebooks/train_base_model_colab.ipynb` from cell 1.
3. Confirm unique UIDs and `DATASET INTEGRITY: PASS`.
4. Run the one-sample CTC sanity gate.
5. If Gate B fails, stop and record the exact output here.
6. If Gate B passes, run controlled full training using `--seed 42`.
7. Run `python -m app.training.evaluate` against the new checkpoint and confirm decoded text uses the real vocabulary.
8. Validate the new checkpoint on multiple real clips.
9. Only after the base model passes the quality gate, resume BridgeAdapter work.
10. Run `python -m pytest backend/tests` including `test_feature_contract_scripts.py`.
11. Verify actual production environment variables, database persistence, and frontend -> API -> MediaPipe integration separately before claiming raw-video production readiness.

## 23. Change log

### 2026-08-25 — Duplicate-UID root cause discovered

- Repeated filename stems across sentence folders were confirmed.
- Old filename-stem UID logic could overwrite features and mispair labels.
- Collision-safe training notebook implemented.

### 2026-08-25 — Real-video validation failure confirmed

- `you are good`: real pose `(69,132)`, face `(69,1404)`, 100% blank output.
- `i am suffering from fever`: real pose `(72,132)`, face `(72,1404)`, 100% blank output.
- Current checkpoint remains unverified.

### 2026-08-25 — Deep static audit fixes

- Made training split deterministic with `--seed 42` and aligned notebook acceptance with the exact split.
- Fixed evaluator vocabulary initialization and added checkpoint/vocabulary dimension validation.
- Added runtime feature-dimension, frame-alignment, finite-value, non-empty-target, and CTC target-length guards to the dataset/collator.
- Runtime verification was not available from the agent environment.

### 2026-08-25 — Autonomous audit pass: extension + converter contract

- Audited repository root, backend structure, ML training path, notebooks, model weights presence, configuration, API boundary, and recent Git history.
- Direct shell `git clone` was attempted and failed with DNS resolution for `github.com`; repository inspection therefore used authenticated GitHub repository access instead of pretending local execution worked.
- Found standalone extractor case-sensitive video-extension lookup. Fixed with `resolve_video_path()` and regression tests.
- Found stale iSign converter `1434` face-dimension contract. Fixed to strict `1404` and fail-fast validation.
- Added `backend/tests/test_feature_contract_scripts.py` covering uppercase video extensions and the current face contract.
- Updated this ledger with evidence, status, blockers, and next actions.
- No runtime tests were executed after these latest changes because local clone/runtime access is blocked in this agent environment. **NOT VERIFIED**.

## 24. Final current-state verdict

**NOT READY**.

Reason:
- the old checkpoint is not trusted;
- the clean-data retrain has not been runtime-verified;
- the one-sample CTC gate has not been re-passed after the latest fixes;
- backend raw-video extraction is not implemented at the production API boundary;
- repository runtime tests could not be executed from this agent environment.

The repository is in a stronger state than before, but the acceptance evidence required for a working product still has to be produced by the Colab/CI runtime.

## Golden rule

A change is not VERIFIED because the code looks correct. It is VERIFIED only when the relevant test or real-data experiment passes. Every step and every issue must leave a written state here.