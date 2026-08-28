# VisionBridge Multi-Agent Engineering Contract + Project Diary

## Operating rules

This is the persistent hand-off record for ChatGPT/Codex, Claude, and future agents. Read it before editing. Every meaningful change, failure, test result, design decision, runtime blocker, and deployment finding must be recorded here. Distinguish `CODE FIXED`, `STATIC VERIFIED`, `RUNTIME VERIFIED`, `CLAUDE-REPORTED`, and `NOT VERIFIED`.

Change only files required by the current task. Never use destructive Git cleanup in Colab. Never push a model because loss decreased, logits are finite, or a trivial token is emitted. A model checkpoint is acceptable only after real-data semantic validation.

Engineering loop:
```text
DISCOVER -> UNDERSTAND -> TRACE -> REPRODUCE -> ROOT CAUSE
-> DESIGN -> IMPLEMENT -> INTEGRATE -> TEST -> REGRESSION
-> REVIEW -> IMPROVE -> RE-AUDIT -> ZERO-LOOSE-ENDS
```

The project follows the uploaded autonomous engineering protocol. The protocol requires whole-repository reconnaissance, end-to-end feature completion, creative improvement, security/performance/UX reviews, regression testing, and a final evidence-based release verdict.

---

# Mission

Continuous Indian Sign Language -> English translation with few-shot signer adaptation.

Current intended pipeline:
```text
Camera / real video
 -> MediaPipe Holistic
 -> pose + face + left-hand + right-hand skeletons
 -> hand-aware temporal PyTorch model
 -> character CTC decoder
 -> English text
 -> optional BridgeAdapter signer personalization
```

Frontend is standardized on React + Vite + TypeScript. Backend remains FastAPI + SQLAlchemy. Training remains PyTorch with Colab and persistent Lightning Studio orchestration. Visual language: monochrome, neat, stylish, minimalistic.

---

# Current multimodal contract

| Stream | Features/frame |
|---|---:|
| Pose | 132 = 33 * (x,y,z,visibility) |
| Face | 1404 = 468 * (x,y,z) |
| Left hand | 63 = 21 * (x,y,z) |
| Right hand | 63 = 21 * (x,y,z) |
| CTC blank | 0 |
| Vocabulary | 49 |
| Maximum sequence | 1024 frames |

Every trained hand-aware sample must carry four synchronized streams.

---

# Chronological problem diary

## A. Dataset/path failures

### Missing processed dataset
Observed:
```text
/content/VisionBridge/data/processed MISSING
/content/VisionBridge/data/processed/isltranslate MISSING
```
The notebooks were originally assuming processed training data already existed.

Fix: canonical Colab notebook downloads/reconstructs the real ISL-CSLTR data and creates a private runtime processed directory instead of assuming local processed files.

### `.MP4` vs `.mp4`
Linux extension matching skipped files such as `fever (2).MP4` while expecting `fever (2).mp4`.

Fix: extraction resolves supported video extensions case-insensitively.

### Feature output missing after extraction failure
Validation cells tried `np.load()` even when extraction had skipped the video.

Fix: extraction result is validated before feature loading; failure is surfaced with a useful report.

### Duplicate filename / UID collision
Sentence folders contain repeated filenames. Using only `path.stem` risks feature/label collisions.

Fix: UID uses normalized label + source stem + relative-path hash.

---

## B. MediaPipe/Colab failures

Observed:
```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
ModuleNotFoundError: No module named 'mediapipe.python'
ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value
```

Fixes:
- dedicated Python 3.12 environment for MediaPipe extraction;
- `mediapipe==0.10.21`;
- `numpy==1.26.4`;
- isolated `MPLBACKEND=Agg` and `MPLCONFIGDIR`;
- temporary extraction helpers live outside the repo.

---

## C. Notebook portability failures

Observed:
```text
NotebookMissingRequiredFieldsError: nbformat_minor
```

Fix: canonical notebooks rebuilt as valid nbformat 4 notebooks with `nbformat_minor` present.

The intended Colab UX is now: fresh GPU runtime -> run cell 1 -> run to the final cell without manual data preparation.

---

## D. Feature dimensions

Original contract was verified as:
```text
pose = (T,132)
face = (T,1404)
```

Real examples previously produced:
```text
you are good -> (69,132), (69,1404)
i am suffering from fever -> (72,132), (72,1404)
```

New hand-aware contract is:
```text
pose       (T,132)
face       (T,1404)
left_hand  (T,63)
right_hand (T,63)
```

All streams must have the same frame count and finite values.

---

## E. CTC failures

### Blank collapse
Earlier checkpoints produced:
```text
PREDICTED: (no sign detected)
BLANK RATIO: 1.0
NON-BLANK: 0
```

The checkpoint was rejected.

### False acceptance gate
An earlier gate only checked that at least one frame was non-blank. Therefore this passed:
```text
TRAIN -> ' '
VAL   -> ' '
MODEL ACCEPTANCE: PASS
```

Fix: semantic gate rejects empty/whitespace-only, space-dominated, trivial-token, and high-CER outputs.

### Space collapse forensic result
The bad checkpoint produced:
```text
TRAIN prediction: ' '
VAL prediction:   ' '
Blank ratio: 0.0000
Space ratio: 1.0000
Meaningful tokens: 0
Mean P(space) ~= 0.8521
```
Manual greedy decoding and repository decoding agreed, so it was not a decoder mismatch. Input features were finite and non-identical.

### Latest old-architecture overfit attempt
Fresh Colab run on one real target:
```text
Target: 'help me'
Trainable parameters: 7,512,369
Loss: 38.2470 -> 1.99 (94.7% reduction)
Prediction: ''
CER: 1.0000
Meaningful tokens: 0
```
The strengthened semantic gate correctly blocked full training. Lesson: loss reduction is not semantic learning.

---

# F. Inference freezing clarification

Forensic inspection through `load_frozen_base_model()` showed zero trainable parameters. That is expected for inference because the loader intentionally freezes the production backbone.

Actual training constructs `VisionBridgeBaseModel` directly and must contain trainable parameters.

Rule:
```text
0 trainable parameters in inference loader = expected
0 trainable parameters in training model = bug
```

Do not remove inference-time freezing to make diagnostics look healthy.

---

# G. Training-path hardening

`train_base_model.py` was hardened to:
- verify trainable parameter count;
- restrict optimizer to trainable parameters;
- verify gradients on the first training batch;
- reject missing/non-finite gradients;
- avoid stale checkpoint reuse on fresh runs;
- support pinned memory and workers;
- support resumable checkpoints;
- verify final checkpoint artifact.

`overfit_sanity.py` was hardened to:
- train on real samples;
- measure loss reduction;
- reject blank collapse;
- reject space collapse;
- reject low meaningful-token diversity;
- calculate CER;
- block full training when semantic learning fails.

---

# H. Hand-aware multimodal redesign

## Requirement
Track hand positions/signs and represent them as skeleton frames, then make that information available to the actual recognition model rather than keeping it as decorative UI only.

## Decision
The old pose+face checkpoint is insufficient for a hand-aware model. The project therefore performs a deliberate model-contract migration.

MediaPipe Holistic hand landmarks:
```text
left hand  = 21 landmarks * 3 = 63 values/frame
right hand = 21 landmarks * 3 = 63 values/frame
```

New architecture:
```text
Pose encoder
Face encoder
Left-hand encoder
Right-hand encoder
        |
        v
Learned gated multimodal fusion
        |
        v
Temporal convolution
        |
        v
Shared Transformer
        |
        v
49-class character CTC head
```

The temporal convolution is intended to capture local signing motion before global Transformer attention.

## Implementation
`backend/app/models/base_model.py`
- added `HAND_INPUT_DIM=63`;
- added separate left/right hand encoders;
- added four-stream gated fusion;
- added temporal convolution;
- accepts synchronized length masks;
- still freezes only in the inference loader.

`backend/scripts/extract_keypoints.py`
- extracts left/right 21-point hand skeletons;
- writes `left_hand/<uid>.npy` and `right_hand/<uid>.npy`;
- validates dimensions, frame alignment, and finite values.

`backend/app/training/isltranslate.py`
- loads all four streams;
- validates all four streams;
- keeps frame alignment while downsampling;
- pads all four streams in CTC collation.

`backend/app/training/train_base_model.py`
- trains the hand-aware model with four modalities;
- preserves gradient/trainability/checkpoint protections.

`backend/app/training/overfit_sanity.py`
- tests the hand-aware model on real examples;
- rejects trivial outputs.

`backend/app/services/inference_service.py`
- accepts four modalities;
- rejects the legacy pose+face checkpoint;
- reports legacy checkpoint readiness as unavailable until retraining.

`backend/app/api/translate.py` and `backend/app/api/calibration.py`
- validate and pass both hand streams through the API.

`backend/app/models/bridge_adapter.py` and `backend/app/services/calibration_service.py`
- adapt/calibrate using the hand-aware multimodal path;
- retain support logic for legacy pose+face base modules inside the adapter implementation where applicable.

## Checkpoint rule
The previous `backend/app/models/weights/base_model.pt` is a legacy pose+face artifact. Do not use it with the new production path.

`load_frozen_base_model()` now raises a clear retraining error for a legacy checkpoint instead of attempting an unsafe partial load.

Status:
```text
HAND-AWARE CODE: CODE FIXED
LEGACY CHECKPOINT: INVALID FOR NEW MODEL
NEW CHECKPOINT: NOT YET TRAINED
MODEL QUALITY: NOT VERIFIED
```

---

# I. Frontend framework migration

## Framework
The frontend is standardized on:
```text
React + Vite + TypeScript
```

Reason: the application has multiple connected authenticated workflows and real-time camera state. React provides a single state/rendering model while Vite gives a straightforward production build. The backend remains FastAPI rather than forcing an unrelated server framework onto the ML/API layer.

## Design language
User preference:
```text
monochrome
neat
stylish
minimalistic
```

The new UI uses:
- restrained black/white/gray palette;
- high whitespace;
- subtle borders;
- compact status indicators;
- minimal decorative UI;
- real application state rather than fake metrics.

## New frontend structure
```text
frontend/
  index.html
  package.json
  vite.config.ts
  tsconfig.json
  src/
    main.tsx
    App.tsx
    api.ts
    landmarks.ts
    useLandmarkSession.ts
    styles.css
    vite-env.d.ts
```

`landmarks.ts` centralizes MediaPipe loading, hand skeleton topology, and synchronized frame extraction.

`useLandmarkSession.ts` owns camera, MediaPipe lifecycle, sampling, canvas overlay, and synchronized landmark capture.

`App.tsx` contains the connected application routes:
```text
/dashboard
/translate
/calibration
/history
/evaluation
/settings
```

The live translation screen:
- uses the real camera;
- tracks pose/face/left/right hands;
- draws both 21-point hand skeletons;
- sends four-stream windows to the API;
- displays real prediction/confidence/latency/errors.

Calibration captures synchronized multimodal frames rather than merely running a timer.

## Legacy frontend removal
The previous static page system duplicated the application and created a second UI source of truth. The obsolete `frontend/pages/*` pages and legacy `frontend/assets/*` CSS/JS files were removed so the Vite React application is the canonical deployed frontend.

Status:
```text
REACT/VITE STRUCTURE: CODE FIXED
LEGACY DUPLICATE FRONTEND: REMOVED
BROWSER RUNTIME: NOT VERIFIED
```

---

# J. Notebook redesign

Canonical notebooks:
```text
notebooks/train_base_model_colab.ipynb
notebooks/train_base_model_lightning.ipynb
notebooks/validate_base_model_colab.ipynb
```

The Colab notebook now follows:
```text
fresh runtime
 -> safe git sync
 -> GPU check
 -> isolated MediaPipe
 -> automatic ISL-CSLTR download
 -> collision-safe manifest
 -> four-stream extraction
 -> data contract validation
 -> semantic overfit gate
 -> full hand-aware training
 -> multi-sample semantic acceptance
 -> optional safe model push
```

The validation notebook:
```text
automatic dataset download
 -> several real sentence videos
 -> four-stream extraction
 -> hand-aware checkpoint check
 -> inference
 -> ground truth/prediction/confidence/CER/blank/space/shape report
```

The Lightning notebook is a persistent-workspace orchestrator around the same repository training code so Colab and Lightning do not silently implement different models.

---

# K. Git incidents / model push history

Earlier model push:
```text
241d8fcb664ac81612f3aa6e28360dd7c8dc9e5b
```
Only the model checkpoint was staged; `data/model_check/` remained untracked.

Git identity failure:
```text
Author identity unknown
```
Fix: explicit user.name/user.email in Colab before local commit.

Whitelist failure:
```text
Unexpected file staged/changed: data/model_check/
```
Fix: only explicitly approved model/vocabulary paths are staged.

Current hand-aware migration intentionally invalidates the old checkpoint for readiness. A new checkpoint must not be pushed until semantic gates pass.

---

# L. General backend/security hardening

Implemented or previously hardened:
- JWT authentication;
- server-side ownership checks;
- request dimension/finiteness validation;
- readiness vs liveness endpoints;
- calibration duration and frame caps;
- adapter budget enforcement;
- safe adapter deletion;
- real dashboard/history/evaluation data flows;
- authenticated CSV export;
- CORS/security headers;
- timeout-aware API client;
- removal of fake production translation/benchmark behavior.

Known production gaps:
```text
SQLite on Render -> ephemeral
localStorage bearer token -> still present
rate limiting -> not implemented
raw video server inference -> not implemented
```

---

# M. Claude polish history

Claude's earlier polish pass reported:
1. test self-reference false failure fixed;
2. pandas import moved to lazy import;
3. dead auth ternary simplified;
4. CSV export fixed to authenticated fetch/Blob;
5. Render cwd-fragility override removed;
6. CER placeholder evaluation bug fixed.

Claude reported 52/52 tests and compile/YAML checks. This remains `CLAUDE-REPORTED` until independently rerun.

---

# N. Required acceptance gates

## Gate A — Dataset
```text
metadata
pose
face
left_hand
right_hand
unique UID
aligned frame counts
finite features
valid targets
valid CTC alignment
```

## Gate B — Semantic overfit
Must demonstrate:
```text
finite loss
meaningful loss reduction
not blank-only
not space-only
multiple meaningful tokens
acceptable CER
```

## Gate C — Full training
Only after Gate B.

## Gate D — Multi-sample train/held-out acceptance
Use several train and validation examples and reject trivial output.

## Gate E — Real-video validation
Use multiple real ISL videos and record:
```text
ground truth
prediction
confidence
CER
blank ratio
space ratio
frame count
pose/face/hand shapes
```

## Gate F — Application E2E
Verify:
```text
auth
-> dashboard
-> live translation
-> hand skeleton overlay
-> calibration
-> adapter
-> history
-> evaluation
-> settings
```

---

# O. Verification status

Static GitHub inspection confirms the new hand-aware files and React frontend structure are present on `main`.

A local container clone attempt was blocked by the environment's inability to resolve `github.com`, so local compile/build execution was NOT VERIFIED here.

GitHub Actions for the latest hand-aware redesign commit had not produced a visible workflow run through the available connector at the time of this entry. Therefore CI is NOT VERIFIED.

The new Colab hand-aware training path has not yet passed a fresh runtime semantic overfit gate. The old checkpoint is explicitly rejected by the new loader. Therefore model quality remains NOT VERIFIED.

---

# P. Current blocker board

```text
A  fresh hand-aware Colab extraction             NOT VERIFIED
B  hand-aware semantic overfit                   NOT VERIFIED
C  full hand-aware training                      BLOCKED until B
D  new hand-aware checkpoint                     BLOCKED until C
E  multi-video real validation                   BLOCKED until D
F  adapter calibration on new checkpoint         BLOCKED until E
G  backend CI after migration                    NOT VERIFIED
H  frontend TypeScript/Vite build                NOT VERIFIED
I  browser camera + hand overlay                 NOT VERIFIED
J  Render E2E after Vite migration               NOT VERIFIED
K  durable production DB                          NOT IMPLEMENTED
L  HttpOnly production auth                       NOT IMPLEMENTED
M  production rate limiting                       NOT IMPLEMENTED
N  40 GB scale                                   BLOCKED until correctness
```

---

# Q. Required next execution

1. Start a fresh Colab GPU runtime.
2. Pull latest `main`.
3. Run `notebooks/train_base_model_colab.ipynb` from cell 1.
4. Confirm four feature directories exist and every sample is synchronized.
5. Confirm `DATASET INTEGRITY: PASS`.
6. Run the hand-aware semantic overfit gate.
7. If it fails, stop and diagnose the first failure. Do not full-train.
8. If it passes, run full hand-aware training.
9. Run multi-sample train/held-out semantic acceptance.
10. Push only after that acceptance passes.
11. Run `validate_base_model_colab.ipynb` against several real videos.
12. Run backend tests and `npm run check` / `npm run build` for the React frontend.
13. Run browser camera/E2E checks with actual hand skeleton rendering.
14. Verify Render build/readiness/auth/CORS.
15. Only then consider large-scale 8.5 GB -> 40 GB training.

---

# R. Scaling 8.5 GB -> 40 GB

More data does not repair a broken objective. Once correctness is established, the larger dataset should use:
```text
incremental extraction
 -> sharded persistent features
 -> bounded DataLoader memory
 -> pinned memory/workers
 -> deterministic manifests
 -> resumable checkpoints
 -> periodic semantic validation
```

Do not load the whole 40 GB source dataset into RAM.

---

# S. Final release verdict

## NOT READY

Reason:
- the previous checkpoint failed CTC semantics;
- the strengthened gate correctly blocked the latest old-architecture attempt;
- the hand-aware architecture and integrations are now implemented, but have not yet passed a fresh real-data semantic runtime gate;
- the old checkpoint is deliberately rejected by the new loader;
- browser/runtime/CI deployment verification remains open.

The product should only move to `READY WITH MINOR ISSUES` after the hand-aware model and application gates pass.

---

# T. LATEST DIARY ENTRY — FULL PROTOCOL REDESIGN + HAND-AWARE MIGRATION

## 2026-08-27

User requested full-scale redesign using the uploaded autonomous engineering protocol and a suitable project-wide framework, plus a preference for a monochrome/minimal visual system.

Actions performed:
```text
1. Reconstructed the ML model as a hand-aware four-stream temporal architecture.
2. Extended extraction to left/right MediaPipe hand skeletons.
3. Extended dataset loading/collation to four synchronized streams.
4. Extended training and semantic overfit gate to four streams.
5. Extended translation/calibration APIs and services.
6. Changed model readiness so the old legacy checkpoint is unavailable until retraining.
7. Added React/Vite/TypeScript application structure.
8. Added shared landmark/session modules and visible 21-point hand skeletons.
9. Replaced timer-only calibration with real capture/submission flow.
10. Migrated the Render frontend to Vite build/dist + SPA routing.
11. Removed duplicate legacy static pages and JS/CSS assets so React is the single frontend source.
12. Updated the canonical Colab and validation workflows for hand-aware data.
13. Rebuilt Lightning Studio notebook around the canonical repository trainer.
14. Added hand model/collation regression coverage.
15. Updated this diary with the full migration and current blocker state.
```

Tests/verification:
```text
GitHub source inspection: PASS
GitHub branch updates: PASS
Local compile/build: NOT VERIFIED (container network could not resolve github.com)
GitHub Actions latest run: NOT VERIFIED (no run surfaced)
Fresh Colab hand-aware semantic overfit: NOT VERIFIED
Browser E2E: NOT VERIFIED
Model quality: NOT VERIFIED
```

Important conclusion:
```text
The project is now architecturally migrated to a hand-aware multimodal stack,
but the new model must be trained and semantically validated before the product
can be called functional. Do not push the legacy checkpoint as a hand-aware model.
```

---

# U. LATEST DIARY ENTRY — 2026-08-28 HAND-AWARE TRAINING VERIFICATION AUDIT

## Repository state inspected

`main` was inspected at `29a870afc088e005508174fd8367c102d4f57216`. The hand-aware model, four-stream dataset/collator, inference readiness checks, React/Vite frontend, Render configuration, and legacy checkpoint are present.

The old `base_model.pt` is still a legacy pose+face artifact and remains intentionally invalid for the hand-aware production loader.

## Static findings

1. `VisionBridgeBaseModel` is trainable when constructed directly; `load_frozen_base_model()` freezes only the inference instance.
2. `train_base_model.py` passes pose, face, left hand, right hand, input lengths, and target lengths into CTC correctly at the API level.
3. `overfit_sanity.py` already blocks empty, whitespace-only, space-collapse, low-diversity, and high-CER outputs.
4. `backend/tests/test_model_padding_mask.py` still used the pre-migration two-stream call signature. This was a real regression in the test suite and has been corrected to use all four streams.
5. `semantic_gate_failures()` required three threshold keyword arguments even though the test suite called it without them. Explicit defaults were added, matching the CLI defaults.
6. The overfit gate previously reported loss and greedy output but could not distinguish optimizer failure, probability collapse, and decoder behavior. It now reports first-step gradient norms, first-step parameter delta, framewise blank/space argmax ratios, and target-character peak probability.
7. The existing GitHub Actions workflow only ran backend tests/compile and frontend JavaScript syntax checks. It now also runs `npm run check` and `npm run build` for the React/Vite frontend.

## Changes

Branch: `agent/hand-aware-training-diagnostics`

PR: #3

No model checkpoint was generated, modified, or pushed.

## Verification levels

```text
CODE FIXED:
- hand-aware padding-mask regression test
- semantic gate threshold defaults
- overfit diagnostic observability
- frontend CI typecheck/build commands

STATIC VERIFIED:
- current main repository structure
- changed source files and workflow configuration

RUNTIME VERIFIED:
- none in this environment for the changed repository

NOT VERIFIED:
- backend pytest
- frontend npm check/build
- real-data hand-aware extraction
- real-data semantic overfit
- full model training
- real-video inference
- browser E2E
- Render E2E
```

## Runtime diagnosis required next

Run the updated overfit gate on a fresh GPU/real ISL dataset before changing architecture. Interpret the evidence rather than tuning blindly:

```text
gradients missing or parameter delta == 0
    -> graph/trainability/optimizer failure

gradients valid + parameter updates valid + target-character peaks low
    -> feature signal or optimization/model representation problem

target-character peaks substantial + greedy path remains blank/space dominated
    -> CTC alignment/decoding behavior requires focused investigation
invalid frame/target lengths
    -> dataset/collation contract failure
```

Full training remains blocked until the semantic overfit gate passes. No checkpoint may be pushed merely because loss decreases.

## Current audit conclusion

The repository has concrete test/verification regressions that have now been corrected on the audit branch, but the original hand-aware semantic failure is **not yet root-caused** from the available environment because the required real dataset and GPU runtime are unavailable. The safest state is therefore to improve observability, keep the legacy checkpoint rejected, and stop before full training.
