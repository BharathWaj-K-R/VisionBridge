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

The project follows the uploaded autonomous engineering protocol. The protocol requires whole-repository reconnaissance, end-to-end feature completion, creative improvement, security/performance/UX reviews, regression testing, and a final evidence-based release verdict. fileciteturn771file0L94-L127 fileciteturn771file0L133-L175

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

Frontend is being standardized on React + Vite + TypeScript. Backend remains FastAPI + SQLAlchemy. Training remains PyTorch with Colab and persistent Lightning Studio orchestration. Visual language: monochrome, neat, stylish, minimalistic.

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

# H. Hand skeleton + model migration

## User requirement
Track hand positions/signs and show hands as skeleton frames.

## Design decision
This became a deliberate model-contract migration instead of a cosmetic overlay only.

MediaPipe Holistic provides 21 landmarks per hand. The project now treats the hands as first-class temporal modalities:
```text
left hand  = 63 values/frame
right hand = 63 values/frame
```

The new base model is:
```text
Pose encoder
Face encoder
Left-hand encoder
Right-hand encoder
        ↓
Learned gated multimodal fusion
        ↓
Temporal convolution block
        ↓
Shared Transformer
        ↓
49-class CTC head
```

The hand-aware model uses local temporal convolution before global attention because sign motion contains short-range movement patterns as well as longer temporal dependencies.

## Important checkpoint consequence
The previous `base_model.pt` is a legacy pose+face checkpoint. It is NOT silently loaded into the new architecture.

`load_frozen_base_model()` now rejects a legacy checkpoint with an explicit retraining message. `model_status()` reports:
```text
legacy_checkpoint_requires_retraining
```
for that artifact.

The project must train a new hand-aware checkpoint before translation readiness can become `ready`.

## Extraction
`extract_keypoints.py` now writes:
```text
pose/<uid>.npy
face/<uid>.npy
left_hand/<uid>.npy
right_hand/<uid>.npy
```
and retains a backward-compatible pose+face wrapper for old extraction callers.

## Dataset/collation
`isltranslate.py` now loads and validates all four streams and pads/downsamples them using one shared frame index sequence.

## API
Translation and calibration contracts now carry left/right hand arrays. The API validates frame synchronization, dimensions, finiteness, and maximum sequence length before tensor conversion.

## Adapter
BridgeAdapter calibration now consumes all four streams while keeping the trained base frozen.

## Regression test
`backend/tests/test_hand_contract.py` covers hand-aware output shape and required-hand validation.
The long-clip collation regression now includes both hand streams and verifies a finite CTC loss through the hand-aware model.

---

# I. Frontend framework redesign

The frontend is being standardized around:
```text
React + Vite + TypeScript
```
with a small reusable API client and a reusable MediaPipe landmark-session hook.

The new UI direction is:
```text
monochrome
minimal
high whitespace
subtle borders
restrained typography
no decorative fake metrics
```

Live translation now:
- opens the real camera;
- loads MediaPipe Holistic;
- tracks pose/face/hands;
- displays left/right 21-point skeleton overlays;
- buffers synchronized frames;
- sends all four streams to the API;
- displays real prediction/confidence/latency/error states.

Calibration now captures real synchronized multimodal frames for adapter fitting rather than pretending that a timer is calibration.

Dashboard/history/evaluation/settings consume backend data instead of fake static values.

Render frontend build now targets Vite `dist` with an SPA rewrite.

Runtime browser verification is still `NOT VERIFIED` until a real browser session confirms camera permissions, MediaPipe loading, overlay drawing, and API translation.

---

# J. Git incidents and protections

Observed:
```text
Author identity unknown
```
Fix: configure Git identity in Colab before local commits.

Observed:
```text
Unexpected file staged/changed: data/model_check/
```
The push whitelist correctly refused unrelated diagnostic artifacts.

Model-push rule:
Only explicitly approved checkpoint/vocabulary files may be staged by the training notebook. Temporary validation directories stay outside source history.

---

# K. Application hardening history

Previously completed general hardening includes:
- JWT registration/login;
- server-side ownership authorization;
- translation/calibration payload validation;
- readiness endpoint separate from liveness;
- safe adapter deletion staging;
- persistent dashboard/history/evaluation data flows;
- authenticated CSV export;
- CORS/security headers;
- timeout-aware frontend API client;
- removal of fake translation and fabricated benchmark claims.

Known production constraints:
- SQLite on Render is ephemeral;
- browser bearer token is still stored in localStorage;
- production rate limiting is not implemented;
- raw-video server-side inference is not implemented.

---

# L. Claude polish history

Claude's prior polish pass reported six fixes:
1. test self-reference false failure;
2. lazy pandas import for clean CI;
3. dead auth ternary removal;
4. authenticated CSV export via fetch/Blob;
5. removal of redundant Render cwd override;
6. CER evaluator correction.

Claude reported 52/52 passing plus compile/YAML checks. Treat this as `CLAUDE-REPORTED` until independently rerun.

---

# M. Canonical notebooks

Keep these three project notebooks:
```text
notebooks/train_base_model_colab.ipynb
notebooks/train_base_model_lightning.ipynb
notebooks/validate_base_model_colab.ipynb
```

Colab notebook responsibilities:
1. safe repo sync;
2. GPU check;
3. isolated MediaPipe setup;
4. automatic real dataset download;
5. collision-safe data rebuild;
6. four-stream extraction;
7. dataset contract validation;
8. semantic overfit gate;
9. hand-aware full training;
10. multi-sample semantic acceptance;
11. optional safe model push.

Lightning notebook responsibilities:
- persistent-workspace orchestration;
- same repository model/trainer rather than a divergent second implementation;
- four-stream validation;
- semantic gate;
- resumable long training.

Validation notebook responsibilities:
- automatic real-video selection;
- four-stream extraction;
- reject legacy checkpoint;
- report prediction/confidence/CER/blank/space ratios and shapes for several real videos.

---

# N. Current blocker board

```text
A  Hand-aware extraction on clean Colab            NOT VERIFIED
B  Hand-aware semantic overfit                      NOT VERIFIED
C  Hand-aware full training                         BLOCKED until B
D  New hand-aware checkpoint                        NOT AVAILABLE until C
E  Multi-video real validation                      BLOCKED until D
F  BridgeAdapter real calibration validation        BLOCKED until E
G  Current CI after redesign                        NOT VERIFIED
H  Browser runtime hand overlay                    NOT VERIFIED
I  Render end-to-end after React migration          NOT VERIFIED
J  Durable production DB                            NOT IMPLEMENTED
K  HttpOnly auth strategy                           NOT IMPLEMENTED
L  Production rate limiting                         NOT IMPLEMENTED
M  40 GB training scale                             BLOCKED until correctness is proven
```

---

# O. Required next execution

1. Start a fresh Colab GPU runtime.
2. Pull latest `main`.
3. Run `train_base_model_colab.ipynb` from cell 1.
4. Confirm extraction creates all four arrays with synchronized frame counts.
5. Confirm `DATASET INTEGRITY: PASS`.
6. Run the semantic overfit gate and stop immediately if it fails.
7. Do not use the old pose+face checkpoint.
8. Train the new hand-aware checkpoint only after Gate B passes.
9. Run multi-sample train/held-out acceptance.
10. Push checkpoint + vocabulary only after semantic acceptance.
11. Run `validate_base_model_colab.ipynb` on several automatically selected real videos.
12. Run backend tests and frontend TypeScript/build checks.
13. Test the React browser flow with a real camera and verify both hand skeletons visibly track.
14. Test auth -> dashboard -> translate -> history -> calibration -> adapter -> evaluation.
15. Only after correctness is established, scale the data from ~8.5 GB toward 40 GB using incremental extraction, sharding, bounded memory, workers, and resumable checkpoints.

---

# P. Design priorities for future agents

1. Correctness before scale.
2. Real data before benchmark claims.
3. Semantic evidence before checkpoint push.
4. Hands are first-class model inputs in the new architecture.
5. Reuse the canonical repository trainer instead of duplicating training logic in notebooks.
6. Keep temporary data outside source history.
7. Do not claim runtime verification without runtime evidence.
8. Preserve the monochrome/minimal product language.
9. Fix root causes, not symptoms.
10. Update this file whenever a gate passes/fails or a new blocker appears.

---

# Q. Release verdict

## NOT READY

The software architecture is substantially more coherent and the hand-aware redesign is implemented in code, but the central product promise is still blocked by model runtime verification.

The next valid milestone is:
```text
fresh real data
 -> four-stream extraction
 -> semantic overfit PASS
 -> hand-aware training
 -> multi-video real validation
 -> application E2E verification
```

Only after those gates pass should the project be treated as a functioning ISL translation product or scaled to 40 GB training.
