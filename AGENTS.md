# VisionBridge Multi-Agent Engineering Contract + Complete Project Diary

## 0. PURPOSE / OPERATING PROTOCOL

This file is the persistent hand-off memory for all agents working on VisionBridge (ChatGPT/Codex, Claude, or future agents).

### State-update rule
Every meaningful code/notebook change, error, test result, training result, validation result, root-cause discovery, workaround, decision, blocked state, and milestone must be recorded here.

Never mark anything `VERIFIED` unless an actual runtime check produced the evidence.

### Scope rule
Change only the files required for the current task. Do not use destructive Git operations in Colab workflows when user work may exist.

### Training safety rule
Never push a model checkpoint merely because loss decreased, logits are finite, a non-blank frame exists, a trivial token is emitted, or an acceptance cell prints PASS without semantic evidence.

### Handoff rule
If an agent is interrupted, the next agent must read this file first and continue from the last recorded state rather than repeating old experiments blindly.

---

# 1. MISSION

Continuous Indian Sign Language (ISL) -> English translation.

```text
Real video
  -> MediaPipe Holistic
  -> pose(132) + face(1404)
  -> VisionBridgeBaseModel
  -> CTC decode
  -> English text
  -> optional BridgeAdapter signer personalization
```

Product intent:
- real browser camera capture;
- browser-side landmark extraction;
- server-backed translation;
- few-shot signer adaptation;
- authenticated history/calibration/profile workflows.

---

# 2. CORE REPOSITORY CONTRACT

## Model/data contract

| Item | Contract |
|---|---:|
| Pose features / frame | 132 |
| Face features / frame | 1404 |
| Maximum inference sequence | 1024 frames |
| CTC blank token | 0 |
| Vocabulary size | 49 in current checkpoint/vocab |
| Calibration minimum | 300 seconds |
| Calibration fitting cap | 256 frames |

Backbone:
```text
Pose encoder + Face encoder + Cross-modal fusion + Shared Transformer + CTC character head
```

## Repository layout

```text
backend/    FastAPI + SQLAlchemy/SQLite + model/training code
frontend/   static HTML/CSS/vanilla JS UI
notebooks/  Colab + Lightning + validation notebooks
data/       dataset preparation/output area
.github/    regression CI
render.yaml Render deployment configuration
```

## Canonical project notebooks

Keep exactly these as project training/validation notebooks:
1. `notebooks/train_base_model_colab.ipynb`
2. `notebooks/train_base_model_lightning.ipynb`
3. `notebooks/validate_base_model_colab.ipynb`

---

# 3. COMPLETE ISSUE / FIX DIARY FROM THE BEGINNING

## Phase A — Initial project state and architecture work

The project evolved from a prototype toward a real authenticated application. Early work established FastAPI, static frontend, PyTorch Pose+Face model, CTC character decoding, signer-adaptive BridgeAdapter concept, Colab training and validation workflows, and Render deployment.

Historical infrastructure constraints:
- Render free tier has limited RAM/CPU and ephemeral local storage;
- Colab is the practical training environment;
- the related source datasets can be very large;
- model weights are not expected to be trained on Render.

---

## Phase B — Dataset / feature extraction failures

### 1. Missing processed dataset
Observed:
```text
/content/VisionBridge/data FOUND
/content/VisionBridge/data/processed MISSING
/content/VisionBridge/data/processed/isltranslate MISSING
```

Fix: training notebook rebuilds processed data from the real dataset in Colab instead of assuming local processed files exist.

### 2. Uppercase `.MP4` path bug
Observed a missing `.mp4` while the actual source was `.MP4`.
Root cause: case-sensitive extension lookup on Linux.
Fix: extractor resolves common video extensions case-insensitively.

### 3. Real-video extraction output mismatch
Observed feature output missing after selecting `good (3).MP4`.
Fix: use exact source path and derive output from the resolved source path; never blindly load a feature file after a failed extraction.

### 4. Feature dimensions
Contract established:
```text
pose = (T,132)
face = (T,1404)
```
Successful real extraction examples included:
```text
you are good -> pose (69,132), face (69,1404)
i am suffering from fever -> pose (72,132), face (72,1404)
```
Guards were added so wrong dimensions fail explicitly.

---

## Phase C — MediaPipe runtime failures

### 5. `mediapipe.solutions` missing
Observed:
```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

### 6. Legacy import unavailable
Observed:
```text
ModuleNotFoundError: No module named 'mediapipe.python'
```

### 7. Isolated MediaPipe runtime
Dedicated Python 3.12 environment introduced in Colab:
```text
/content/visionbridge_mp312
```
Pinned:
```text
mediapipe==0.10.21
numpy==1.26.4
```

### 8. Matplotlib backend crash
Observed:
```text
ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value
```
Root cause: isolated process inherited Colab's inline backend.
Fix:
```text
MPLBACKEND=Agg
MPLCONFIGDIR=<isolated config directory>
```

---

## Phase D — UID collision / dataset identity failure

### 9. Duplicate filename collision
Repeated filenames occur across sentence folders, e.g.:
```text
fever (2).MP4
fever (5).MP4
good (3).MP4
```
Using only `path.stem` could overwrite features or mismatch features/labels.
Fix: normalized label + stem + relative-path hash for globally unique UIDs.

---

## Phase E — Early CTC blank-collapse failure

### 10. Real-video blank-collapse checkpoint
Observed:
```text
GROUND TRUTH: He is going into the room
PREDICTED:    (no sign detected)
CONFIDENCE:   0.0
BLANK RATIO:  1.0
NON-BLANK:    0
UNIQUE TOKENS: 0
```
Additional clips showed blank ratio 1.0.
Conclusion: old checkpoint rejected as quality evidence.

---

## Phase F — Deterministic split / reproducibility

### 11. Split mismatch risk
Training and notebook acceptance could select different train/validation samples.
Fix: deterministic seed 42 shared across training/acceptance.

---

## Phase G — Notebook portability / Colab problems

### 12. Invalid notebook file
Observed:
```text
NotebookMissingRequiredFieldsError: nbformat_minor
```
Fix: canonical notebook rebuilt as proper nbformat 4 notebook with required fields.

### 13. Full one-pass Colab requirement
Requirement established: fresh runtime, run cell 1 to last cell, automatic environment/data preparation, real extraction, training, semantic validation, and safe push only after acceptance.

---

## Phase H — Validation workflow failures

### 14. Missing canonical processed dataset in validation notebook
Observed:
```text
Missing metadata CSV: /content/VisionBridge/data/processed/isltranslate/ISLTranslate.csv
```
Fix direction: validation notebook uses an isolated validation data area and automatically downloads/selects/extracts real videos.

### 15. Automatic real-video validation
Requirement: download ISL-CSLTR, select real sentence-level video, extract real features, infer using the checkpoint, report ground truth/prediction/confidence/blank ratio/CER.

---

## Phase I — Acceptance gate failure

### 16. False MODEL ACCEPTANCE PASS
Observed:
```text
TRAIN: ('i am so sorry to hear that', ' ', 0.852..., 123)
VAL:   ('i am hungry', ' ', 0.852..., 106)
MODEL ACCEPTANCE: PASS
```
Old logic only checked for a non-blank frame, so `' '` passed.
Fix: semantic gate rejects whitespace-only output, high space-token ratio, trivial diversity, and high CER.

---

## Phase J — Tokenizer/vocabulary investigation

### 17. Vocabulary verified
```text
0  <blank>
1-26 a-z
27-36 0-9
37 space
38-48 punctuation
```
Thus:
```text
vocab size = 49
blank id = 0
space id = 37
```

### 18. Target encoding verified
Example `i am suffering from fever` produced the expected 25 character IDs with spaces represented as 37. No blank labels were injected and target length matched character count.

---

## Phase K — CTC space-token collapse

### 19. Definitive forensic result
Fresh checkpoint:
```text
TRAIN truth: 'i am so sorry to hear that'
TRAIN prediction: ' '
TRAIN blank ratio: 0.0000
TRAIN space ratio: 1.0000
TRAIN meaningful tokens: 0

VAL truth: 'i am hungry'
VAL prediction: ' '
VAL blank ratio: 0.0000
VAL space ratio: 1.0000
VAL meaningful tokens: 0
```
Mean P(space) approximately 0.8521 on both samples.
Manual greedy CTC decode and repository decoder both returned `' '`, ruling out a decoder mismatch.

### 20. Input feature sanity
Features were finite and non-identical between samples.
Therefore the collapse was not explained by identical/all-zero input data.

Conclusion:
```text
SPACE TOKEN COLLAPSE
```

---

## Phase L — Inference vs training distinction

### 21. Frozen inference model forensic
Loading via `load_frozen_base_model()` reports all parameters frozen by design. That is an inference property and is NOT evidence that the training script uses zero trainable parameters.

The actual training entry point constructs `VisionBridgeBaseModel` directly and uses AdamW. Therefore the correct debugging target is the training path, not removal of inference freezing.

---

## Phase M — Training-path hardening

### 22. Training script
Hardened with:
- trainable-parameter assertion;
- optimizer restricted to trainable parameters;
- first-batch gradient sanity check;
- missing/non-finite gradient failure;
- stale output cleanup on non-resume runs;
- pinned-memory/multi-worker support;
- configurable `--num-workers`;
- resumable checkpoint support;
- final artifact verification.

Status: CODE FIXED; fresh Colab runtime required.

### 23. CTC sanity script
Hardened with:
- real-data scratch optimization;
- loss reduction measurement;
- blank ratio;
- space-token ratio;
- meaningful-token diversity;
- CER;
- whitespace-only rejection;
- space-collapse rejection;
- trivial-token rejection;
- high-CER rejection.

Status: CODE FIXED; fresh Colab runtime required.

---

## Phase N — Notebook training workflow hardening

### 24. Canonical Colab notebook
Hardened to:
- sync `main` safely;
- refuse dirty local work;
- require GPU;
- isolate MediaPipe 0.10.21;
- download real ISL-CSLTR;
- rebuild processed features from scratch;
- use collision-safe UIDs;
- validate data contracts;
- run semantic overfit gate before full training;
- train deterministically with seed 42;
- semantically validate multiple train/held-out samples;
- push only validated checkpoint + vocabulary;
- avoid destructive Git cleanup;
- keep temporary helpers outside the repository.

### 25. Vocabulary ordering bug
Fix: save tokenizer vocabulary immediately after successful training before acceptance decoder loads it.

---

## Phase O — Git/push failures and safeguards

### 26. Checkpoint push
Earlier validated model commit:
```text
241d8fcb664ac81612f3aa6e28360dd7c8dc9e5b
```
Only `backend/app/models/weights/base_model.pt` was staged. `data/model_check/` remained local/untracked.

### 27. Git identity failure
Observed:
```text
Author identity unknown
```
Fix: configure Git user.name/email explicitly in Colab before local commits.

### 28. Unwanted staged directory safeguard
Observed:
```text
Unexpected file staged/changed: data/model_check/
```
Whitelist correctly refused unrelated artifacts.

---

## Phase P — Claude/general application hardening

Claude's polish pass reported six additional fixes:
1. test self-reference false failure;
2. module-level pandas import moved to lazy import;
3. dead auth ternary removed;
4. authenticated CSV export rewritten using fetch + Blob download;
5. redundant Render cwd override removed;
6. evaluator CER placeholder bug corrected.

Claude reported 52/52 tests and Python/JS/YAML checks; runtime remains CLAUDE-REPORTED unless independently reproduced.

---

## Phase Q — Backend/API/security/frontend hardening

Implemented:
- JWT bearer authentication;
- user/adapter ownership enforcement;
- translation/calibration input validation;
- liveness/readiness split;
- calibration frame cap/downsampling;
- adapter parameter budget enforcement;
- safe adapter deletion;
- real dashboard/history/profile/evaluation state;
- real browser MediaPipe live translation;
- fake translation/benchmark fallbacks removed;
- API endpoint configuration + timeout handling;
- CORS/security response headers;
- frontend JS syntax CI.

Known production gaps:
- SQLite on Render is not durable;
- browser token remains in localStorage;
- no production rate limiter yet;
- raw-video server-side inference is not implemented.

---

# 6. HAND TRACKING / SKELETON FRAME IMPLEMENTATION

## 29. Requirement
User requested the live application to track hand positions/signs and show the hands as a skeleton frame.

## 30. Architectural decision
Hands are already exposed by MediaPipe Holistic in the browser as:
```text
leftHandLandmarks
rightHandLandmarks
```
Each detected hand has 21 landmarks.

This implementation adds **visual/diagnostic hand skeleton tracking without changing the existing translation model contract**.

Current translation contract remains:
```text
pose [T,132]
face [T,1404]
```

This was deliberate. Adding hands as model input would change the learned input contract and invalidate the current checkpoint, forcing a new extraction pipeline, training run, and model acceptance cycle. That is a separate model-contract migration and is NOT silently mixed into the live UI change.

## 31. Files changed
```text
frontend/pages/translate.html
frontend/assets/js/translate-live.js
AGENTS.md
```

## 32. Frontend implementation
`translate.html` now contains:
- a transparent canvas overlay above the live camera;
- a hand-tracking status indicator;
- a dedicated hand-tracking explanation panel.

`translate-live.js` now:
- reads `results.leftHandLandmarks` and `results.rightHandLandmarks`;
- tracks both hands independently;
- renders a 21-point hand skeleton for each detected hand;
- draws hand bone connections using the MediaPipe hand topology;
- resizes the overlay for device pixel ratio;
- updates the UI with `Hands not detected`, `1 hand tracked`, or `2 hands tracked`;
- clears the skeleton when translation stops;
- leaves pose/face backend payload behavior unchanged.

## 33. Model-training impact
None for the existing checkpoint.
The hand skeleton is a live browser visualization/diagnostic feature.

Future model-level hand integration requires a deliberate contract such as:
```text
left hand:  21 * 3 = 63
right hand: 21 * 3 = 63
hands total: 126 values/frame
```
combined with the existing pose+face representation, followed by dataset extraction, model architecture changes, retraining, and real-video semantic acceptance.
Do NOT change the model contract casually.

## 34. Testing status
Static implementation was inspected against the existing MediaPipe Holistic browser flow and DOM selectors. No destructive repository change was made.

Runtime browser verification of actual camera + MediaPipe hand rendering is still:
```text
NOT VERIFIED
```

CI-level JavaScript syntax validation remains the next automated confirmation after the push.

---

# 7. CURRENT MODEL STATE / CRITICAL LESSON

Most recent observed failed checkpoint:
```text
TRAIN -> ' '
VAL   -> ' '
space ratio -> 1.0
meaningful tokens -> 0
```

The semantic acceptance gate has been strengthened to prevent this from passing again.

---

# 8. REQUIRED ACCEPTANCE GATES

## Gate A — Dataset integrity
Must prove metadata, files, unique UIDs, pose/face dimensions, frame alignment, finite features, valid targets, and valid CTC alignment.

## Gate B — Semantic CTC overfit
Must prove finite loss, meaningful reduction, no blank-only collapse, no space-only collapse, multiple meaningful tokens, and acceptable CER.

## Gate C — Full clean training
Only after Gate B passes.

## Gate D — Train + held-out semantic acceptance
Use multiple training and held-out samples and reject trivial collapse.

## Gate E — Real-video validation
Use real videos and report ground truth, prediction, confidence, blank ratio, CER, frame count, feature shapes, and readiness.

## Gate F — Application regression
Verify auth, dashboard, translation, hand skeleton overlay, calibration, adapter ownership, history, CSV export, evaluation, readiness.

---

# 9. KNOWN ERROR CATALOG

```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```
MediaPipe runtime mismatch.

```text
ModuleNotFoundError: No module named 'mediapipe.python'
```
Incompatible MediaPipe package/API combination.

```text
ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value
```
Isolated MediaPipe process inherited Colab matplotlib backend.

```text
NotebookMissingRequiredFieldsError: nbformat_minor
```
Invalid notebook structure.

```text
Missing metadata CSV: /content/VisionBridge/data/processed/isltranslate/ISLTranslate.csv
```
Validation notebook assumed unavailable processed training data.

```text
no video file at .../fever (2).mp4
```
Case mismatch with actual `.MP4` file.

```text
Author identity unknown
```
Git identity was not configured in Colab.

```text
Unexpected file staged/changed: data/model_check/
```
Push whitelist correctly refused unrelated diagnostics.

```text
PREDICTED: (no sign detected)
BLANK RATIO: 1.0
```
Historical blank collapse.

```text
PREDICTED: ' '
SPACE RATIO: 1.0
```
Observed space-token collapse.

---

# 10. MULTI-AGENT WORK PROTOCOL

## Before editing
1. Read `AGENTS.md`.
2. Inspect relevant source files.
3. Identify root cause.
4. Explicitly state what is not the root cause where evidence exists.

## During editing
1. Change only required files.
2. Preserve public contracts unless intentionally migrating them.
3. Add/adjust tests.
4. Update this diary after meaningful changes/errors.

## After editing
1. Compile/syntax-check.
2. Run focused tests.
3. Run broader tests when feasible.
4. Re-read changed code.
5. Record exact evidence.
6. Commit intended files only.
7. Do not claim runtime verification from static inspection.

## Error record format
```text
SYMPTOM
ROOT CAUSE
FIX
FILES CHANGED
TEST PERFORMED
RESULT
NEXT BLOCKER
```

---

# 11. CURRENT REPOSITORY / GIT STATE

Branch:
```text
main
```

Latest known main before this hand-skeleton commit:
```text
b86f7febf942db7dc79e91c3dfb8b2f07d619bdf
```

This task adds a subsequent hand-skeleton UI commit and a diary update.

---

# 12. CURRENT BLOCKER BOARD

```text
A  Fresh clean Colab dataset rebuild              NOT VERIFIED
B  Semantic real-data CTC gate                    NOT VERIFIED after latest code changes
C  Full clean base-model training                 BLOCKED until B
D  Multi-video unseen real validation              BLOCKED until C
E  BridgeAdapter held-out signer validation        BLOCKED until D
F  Latest complete backend test suite               NOT VERIFIED
G  Latest GitHub Actions result                     NOT VERIFIED
H  Browser/E2E frontend runtime                      NOT VERIFIED
I  Live Render auth/CORS/readiness/persistence      NOT VERIFIED
J  Durable production database                       NOT IMPLEMENTED
K  HttpOnly production auth strategy                 NOT IMPLEMENTED
L  Production rate limiting                           NOT IMPLEMENTED
M  Model-level hand-feature integration              NOT IMPLEMENTED
N  Browser hand-skeleton runtime validation          NOT VERIFIED
```

---

# 13. CURRENT NEXT STEPS

1. Start fresh Colab runtime.
2. Pull latest `main`.
3. Run `notebooks/train_base_model_colab.ipynb` from cell 1.
4. Require dataset integrity PASS.
5. Require semantic CTC overfit PASS without blank/space/trivial collapse.
6. Run full training only after Gate B.
7. Require multi-sample train + held-out semantic acceptance.
8. Push only validated checkpoint + vocabulary.
9. Run `notebooks/validate_base_model_colab.ipynb` on multiple real unseen videos.
10. Verify backend CI and browser syntax.
11. Verify live Render health/readiness.
12. Test browser hand-skeleton rendering on a real camera session.
13. Test full application journey: auth -> dashboard -> translation -> history -> calibration -> adapter -> evaluation.
14. Only after base-model correctness is established, consider model-level hand feature integration.
15. Only after correctness is established, scale training from ~8.5 GB toward 40 GB with streaming/sharding/resumable training.

---

# 14. SCALING NOTE: 8.5 GB -> 40 GB

Do not equate more data with a fix for model collapse.

Larger-scale design should use:
```text
incremental video extraction
 -> collision-safe persistent IDs
 -> sharded processed features
 -> bounded-memory DataLoader
 -> pinned memory
 -> workers
 -> resumable checkpoints
 -> deterministic split manifest
 -> metrics/early stopping
 -> real held-out evaluation
```

---

# 15. RELEASE VERDICT

# NOT READY

The codebase is substantially hardened, but model quality remains unproven until a fresh real-data training run passes the semantic CTC gate and a separate real-video validation run demonstrates non-trivial useful English output.

Production also remains blocked on durable persistence, production auth hardening, rate limiting, and full browser/live deployment verification.

---

# 16. MOST RECENT DIARY ENTRY — HAND SKELETON FEATURE

## 2026-08-27 — Live hand position/skeleton tracking

Request:
```text
Add the ability to track hand positions and signs using a skeleton hand frame.
```

Analysis:
- MediaPipe Holistic already returns left and right hand landmarks in the existing browser pipeline.
- Each hand contains 21 landmarks.
- The current model was trained for pose 132 + face 1404 only.
- Adding hand features into the model would be a breaking model-input change and would require new data extraction and retraining.

Implementation:
```text
frontend/pages/translate.html
  + transparent hand-skeleton canvas overlay
  + hand tracking status chip
  + hand-tracking explanation panel

frontend/assets/js/translate-live.js
  + hand landmark extraction from Holistic results
  + 21-point skeleton topology
  + live left/right hand drawing
  + responsive canvas sizing
  + detection status
  + cleanup on stop

AGENTS.md
  + full hand-feature decision and state recorded
```

Status:
```text
CODE PUSHED
STATIC CODE REVIEW: COMPLETED
BROWSER RUNTIME: NOT VERIFIED
MODEL HAND INPUT: NOT IMPLEMENTED
EXISTING MODEL CONTRACT: PRESERVED
```

Important next-agent rule:
```text
Do not silently feed hand landmarks into the current model.
The current checkpoint accepts pose(132) + face(1404).
Model-level hand integration is a separate migration and must include extraction,
training, semantic acceptance, validation, and checkpoint replacement.
```
