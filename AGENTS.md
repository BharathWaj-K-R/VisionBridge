# VisionBridge Multi-Agent Engineering Contract + Complete Project Diary

## 0. PURPOSE / OPERATING PROTOCOL

This file is the persistent hand-off memory for all agents working on VisionBridge (ChatGPT/Codex, Claude, or future agents).

### State-update rule
Every meaningful code/notebook change, error, test result, training result, validation result, root-cause discovery, workaround, decision, blocked state, and milestone must be recorded here.

Never mark anything `VERIFIED` unless an actual runtime check produced the evidence.

### Scope rule
Change only the files required for the current task. Unrelated files must remain untouched. Do not use destructive Git operations (`git reset --hard`, `git clean`, blanket checkout/reset) in Colab workflows when user work may exist.

### Training safety rule
Never push a model checkpoint merely because:
- loss decreased;
- logits are finite;
- at least one non-blank frame exists;
- one trivial token is emitted;
- an acceptance cell prints `PASS` without semantic evidence.

A checkpoint is acceptable only after real-data semantic validation.

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

Temporary diagnostic/support notebooks should not become permanent repository clutter.

---

# 3. COMPLETE ISSUE / FIX DIARY FROM THE BEGINNING

## Phase A — Initial project state and architecture work

The project evolved from a prototype toward a real authenticated application. Early work established:
- FastAPI backend;
- static frontend;
- PyTorch Pose+Face model;
- CTC character decoding;
- signer-adaptive BridgeAdapter concept;
- Colab training and validation workflows;
- Render deployment.

Historical infrastructure constraints:
- Render free tier has limited RAM/CPU and ephemeral local storage;
- Colab is the practical training environment;
- the complete ISLTranslate/related source data can be very large;
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

The notebook expected:
```text
data/processed/isltranslate/ISLTranslate.csv
data/processed/isltranslate/pose/
data/processed/isltranslate/face/
```

Fix direction:
- rebuild processed data from the real dataset in Colab;
- make the training notebook download/reconstruct data instead of assuming local processed files exist.

Status: fixed in training notebook workflow; fresh end-to-end runtime still required for final verification.

### 2. Uppercase `.MP4` path bug
Observed:
```text
Skipping fever (2): no video file at .../fever (2).mp4
```
while actual file was:
```text
fever (2).MP4
```

Root cause: case-sensitive extension lookup on Linux.

Fix:
- extraction now resolves common video extensions case-insensitively:
  `.mp4/.MP4/.avi/.AVI/.mov/.MOV/.mkv/.MKV`.

Status: code fixed.

### 3. Real-video extraction output mismatch
Observed:
```text
VIDEO_ROOT contents:
/content/VisionBridge/data/model_check/videos/good (3).MP4
```
while an intermediate step expected the lower-case filename.

Fix:
- use the exact selected source path when copying/extracting;
- resolve paths independently of extension case;
- ensure generated UID and output names derive from the actual source path.

Status: code/workflow hardened.

### 4. Feature dimensions
The expected runtime contract became:
```text
pose = (T,132)
face = (T,1404)
```

Observed successful real extraction examples:
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

Root cause: Colab Python / MediaPipe API compatibility drift.

Attempted legacy import:
```python
from mediapipe.python.solutions import holistic
```
which then failed under an incompatible installation:
```text
ModuleNotFoundError: No module named 'mediapipe.python'
```

### 6. MediaPipe isolated runtime
A dedicated Python 3.12 environment was introduced inside Colab:
```text
/content/visionbridge_mp312
```
with pinned:
```text
mediapipe==0.10.21
numpy==1.26.4
```

### 7. Matplotlib backend crash
Observed:
```text
ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value
```

Root cause: isolated MediaPipe process inherited the notebook inline backend.

Fix:
```text
MPLBACKEND=Agg
MPLCONFIGDIR=<isolated config directory>
```

Status: code/notebook hardened.

---

## Phase D — UID collision / dataset identity failure

### 8. Duplicate filename collision
The ISL-CSLTR dataset contains repeated filenames across sentence folders, for example:
```text
fever (2).MP4
fever (5).MP4
good (3).MP4
```

Using only:
```text
path.stem
```
as UID could overwrite feature files or associate features with the wrong text.

Fix:
- generated IDs now include normalized label + filename stem + hash derived from the relative source path;
- training notebook rebuilds data using globally collision-safe IDs.

Status: FIXED IN CODE.

---

## Phase E — Early CTC blank-collapse failure

### 9. Real-video blank-collapse checkpoint
Observed before hardening:
```text
GROUND TRUTH: He is going into the room
PREDICTED:    (no sign detected)
CONFIDENCE:   0.0
BLANK RATIO:  1.0
NON-BLANK:    0
UNIQUE TOKENS: 0
```

Additional real clip evidence:
```text
you are good
-> no sign detected
-> blank ratio 1.0

i am suffering from fever
-> no sign detected
-> blank ratio 1.0
```

Conclusion:
- the old checkpoint was unusable;
- full training was blocked until the CTC optimization path could be proven on real data.

---

## Phase F — Deterministic split / reproducibility

### 10. Split mismatch risk
Training and notebook acceptance could reconstruct different train/validation selections.

Fix:
```text
seed = 42
```
shared for dataset split and relevant training/acceptance logic.

Status: CODE FIXED; fresh clean runtime re-verification required.

---

## Phase G — Notebook portability / Colab problems

### 11. Invalid notebook file
Observed:
```text
NotebookMissingRequiredFieldsError: nbformat_minor
```

This meant the generated/modified notebook was not valid for Colab.

Fix:
- regenerate proper nbformat 4 notebook structure including required metadata such as `nbformat_minor`.

Status: fixed; current canonical notebook was reconstructed as a valid Colab notebook.

### 12. Notebook should be fully runnable from first cell
Requirement established:
- fresh Colab runtime;
- run training notebook from cell 1 to last cell;
- dataset downloaded automatically;
- environment prepared automatically;
- real features extracted automatically;
- model trained;
- checkpoint validated;
- push only after semantic acceptance.

Status: canonical training notebook rebuilt around that workflow.

---

## Phase H — Temporary model validation workflow

### 13. Wrong assumption about preprocessed data in validation notebook
Observed:
```text
Missing metadata CSV: /content/VisionBridge/data/processed/isltranslate/ISLTranslate.csv
```

Then user established:
```text
/content/VisionBridge/data/model_check/processed
```
contained:
```text
pose/
face/
ISLTranslate.csv
```

Fix direction:
- validation notebook should construct its own isolated validation data directory;
- do not assume training data already exists under the canonical training path;
- automatically download/select real video and extract keypoints for validation.

Status: validation workflow hardened; multi-video runtime verification remains open.

### 14. Validation notebook should select a real video automatically
Requirement:
- download ISL-CSLTR automatically;
- identify a real sentence-level video;
- copy/extract it;
- use ground-truth sentence from its folder name;
- run real model inference;
- report prediction/confidence/blank ratio/CER.

---

## Phase I — Model acceptance gate was too weak

### 15. False `MODEL ACCEPTANCE: PASS`
Observed:
```text
TRAIN: ('i am so sorry to hear that', ' ', 0.852..., 123)
VAL:   ('i am hungry', ' ', 0.852..., 106)
MODEL ACCEPTANCE: PASS
```

The original acceptance logic effectively did:
```python
if non_blank_frames == 0: fail
```

Therefore:
```text
' '
```
passed because it is technically non-blank.

This was a serious acceptance-gate bug.

Fix:
- added semantic CTC gate;
- reject whitespace-only prediction;
- reject high space-token frame ratio;
- require meaningful-token diversity;
- calculate CER;
- reject high-CER predictions;
- retain loss-reduction requirement.

Status: CODE FIXED.

---

## Phase J — Tokenizer/vocabulary investigation

### 16. Vocabulary confirmed correct
Current vocabulary:
```text
0  <blank>
1  a
...
26 z
27 0
...
36 9
37 ' '
38 .
39 ,
40 ?
41 !
42 '
43 "
44 -
45 :
46 ;
47 (
48 )
```

Thus:
```text
vocab size = 49
blank id = 0
space id = 37
```

### 17. Target encoding confirmed sane
Example:
```text
'i am suffering from fever'
```
encoded as:
```text
[9, 37, 1, 13, 37, 19, 21, 6, 6, 5, 18, 9, 14, 7,
 37, 6, 18, 15, 13, 37, 6, 5, 22, 5, 18]
```

Target length matched character count.
No blank labels were injected.

Example:
```text
i am (age)
```
correctly included token IDs for `(` and `)`.

Conclusion:
- tokenizer target encoding was not the primary problem observed here.

---

## Phase K — CTC space-token collapse

### 18. Definitive forensic result
Fresh checkpoint inference on two real samples:

```text
TRAIN:
Truth: 'i am so sorry to hear that'
Prediction: ' '
Blank ratio: 0.0000
Space ratio: 1.0000
Unique meaningful tokens: 0
Confidence: ~0.852

VALIDATION:
Truth: 'i am hungry'
Prediction: ' '
Blank ratio: 0.0000
Space ratio: 1.0000
Unique meaningful tokens: 0
Confidence: ~0.852
```

Raw model distributions:
```text
sample 0: space on 123/123 frames
sample 1: space on 106/106 frames
```

Mean probability of space:
```text
~0.8521
```

Manual greedy CTC decoder and repository decoder agreed:
```text
' '
```

Therefore this was NOT a decoder mismatch.

### 19. Input feature sanity
Real feature data was finite and non-identical:
```text
Pose finite: True
Face finite: True
Pose mean abs difference between samples: ~0.1226
Face mean abs difference between samples: ~0.0512
```

Conclusion:
- input arrays were not simply all-zero or identical.

### 20. Root cause classification
```text
SPACE TOKEN COLLAPSE
```

Important nuance:
- this is the observed checkpoint behavior;
- the underlying optimization/data cause required a training-path investigation;
- do not assume the inference freeze is the cause.

---

## Phase L — Inference vs training distinction

### 21. Zero trainable parameters forensic result
A forensic diagnostic loaded the model using:
```python
load_frozen_base_model(...)
```

It reported:
```text
Total parameters: 7,512,369
Trainable parameters: 0
Frozen parameters: 7,512,369
```

Output head:
```text
Linear(..., out_features=49)
Output head trainable: 0
```

This initially looked like a training failure.

Repository inspection established the actual architecture boundary:
- `load_frozen_base_model()` intentionally freezes the model for inference;
- `train_base_model.py` constructs `VisionBridgeBaseModel` directly;
- training uses AdamW over the training model parameters.

Conclusion:
```text
0 trainable in inference loader = EXPECTED
0 trainable in training model = BUG
```

Do NOT remove inference-time freezing merely to make the forensic number non-zero.

---

## Phase M — Training path hardening

### 22. Training script changes
`backend/app/training/train_base_model.py` was hardened to:
- assert the scratch training model has trainable parameters;
- pass only trainable parameters to AdamW;
- verify gradients on the first training batch;
- fail on missing or non-finite gradients;
- clean stale output/vocab for non-resume runs so an old checkpoint cannot masquerade as a new run;
- support pinned memory / GPU-friendly loading;
- support configurable `--num-workers` (default 2 in notebook workflow);
- support resumable checkpoint state when requested;
- refuse to declare successful training without the expected final artifact.

Status: CODE FIXED; fresh Colab runtime required.

### 23. CTC sanity script changes
`backend/app/training/overfit_sanity.py` now:
- uses real samples;
- performs an actual scratch training loop;
- measures loss reduction;
- measures blank ratio;
- measures space-token ratio;
- measures meaningful-token diversity;
- calculates CER;
- rejects whitespace-only predictions;
- rejects high space collapse;
- rejects trivial token diversity;
- rejects high CER.

It exposes semantic-gate helpers for regression tests.

Status: CODE FIXED; runtime verification pending.

### 24. One important implementation note
The current sanity implementation intentionally uses a tiny, real-data subset and a real trainable `VisionBridgeBaseModel` from scratch. It is an optimization sanity gate, not a model-quality benchmark.

---

## Phase N — Notebook training workflow hardening

### 25. Canonical Colab notebook
`notebooks/train_base_model_colab.ipynb` was rebuilt to:
- clone/pull `main` safely;
- refuse to overwrite dirty repository work;
- require GPU;
- establish isolated MediaPipe 0.10.21 environment;
- download the real ISL-CSLTR dataset;
- select a controlled subset for practical Colab experimentation;
- rebuild processed features from scratch;
- use collision-safe UIDs;
- validate pose/face dimensions;
- validate dataset integrity before training;
- run semantic CTC overfit gate before full training;
- run full training using deterministic seed 42;
- validate train and held-out samples after training;
- push only checkpoint + vocabulary after acceptance;
- avoid destructive `git reset` / `git clean` behavior;
- keep temporary helper artifacts outside the repository.

Important: the notebook's source of truth is the repository, and runtime proof must still be obtained in Colab.

### 26. Training notebook vocabulary ordering
A bug was found where the acceptance decoder could request the vocabulary before it had been saved.

Fix:
```text
training succeeds
 -> tokenizer vocabulary is saved immediately
 -> acceptance decoder loads it
 -> only then is model acceptance performed
```

---

# 4. GIT / COMMIT / PUSH HISTORY

## 27. Model checkpoint push
A validated checkpoint was committed as:
```text
241d8fcb664ac81612f3aa6e28360dd7c8dc9e5b
```
with message:
```text
train: update validated VisionBridge base model
```

Only:
```text
backend/app/models/weights/base_model.pt
```
was staged/committed.

A local untracked directory remained:
```text
data/model_check/
```
which was intentionally not pushed.

### 28. Git identity failure
A Colab push attempt initially failed with:
```text
Author identity unknown
Please tell me who you are.
```

Lesson:
- Colab Git configuration must explicitly set user.name and user.email before local commits if committing from that runtime.

### 29. Unwanted staged directory safeguard
A later commit cell correctly refused:
```text
data/model_check/
```
when it was not an allowed artifact.

Conclusion:
- model push must explicitly whitelist expected paths;
- diagnostic/output directories must not accidentally enter source history.

---

# 5. CLAUDE POLISH / GENERAL APPLICATION HARDENING

Claude's polish pass reported six additional issues and a larger regression pass.

### 30. Test self-reference false failure
A test's own explanatory content contained a substring that a naive assertion banned.
Fix: reword test comments.

### 31. Pandas module-level import
A top-level pandas import could break clean CI.
Fix: lazy import.

### 32. Dead ternary in frontend auth code
Simplified `X ? Y : Y` to the actual expression.

### 33. CSV export auth bug
Plain anchor navigation could not send bearer authentication and would 401.
Fix:
- use authenticated fetch helper;
- create Blob download on success.

### 34. Render cwd-fragility override
A redundant `render.yaml` environment override reintroduced cwd sensitivity.
Fix: removed redundant override.

### 35. Evaluation CER bug
A local evaluator used a placeholder string instead of empty text, producing nonsense numbers for total misses.
Fix: score against actual empty prediction.

Claude reported:
```text
52/52 passing
Python compile clean
JavaScript syntax clean
YAML valid
```

Independent verification established that the claimed Claude commit exists and its diff matches the six fixes, but Claude's runtime numbers remain `CLAUDE-REPORTED` unless independently rerun.

---

# 6. BACKEND/API HARDENING DIARY

### Authentication
Implemented:
- registration;
- login;
- JWT bearer authentication;
- protected application routes.

Anonymous base-model translation remains allowed by design when a compatible checkpoint is installed.

### Authorization
Implemented:
- user-scoped access checks;
- adapter ownership checks;
- calibration authentication/ownership.

### Translation validation
Implemented checks for:
- frame count;
- pose 132;
- face 1404;
- finite values;
- maximum sequence length.

### Calibration validation
Implemented:
- target text/label contract;
- CTC minimum-length validation;
- minimum calibration duration;
- finite feature validation;
- 256-frame fitting cap;
- synchronized downsampling.

### Adapter lifecycle
Implemented:
- parameter budget enforcement;
- owner-scoped deletion;
- reversible file/DB deletion flow.

### Health/readiness
Changed from unsafe single health semantics to:
```text
GET /api/v1/health = liveness
GET /api/v1/ready  = model/vocabulary readiness
```
`/ready` returns HTTP 503 when model readiness is unavailable.
Render health check targets `/api/v1/ready`.
Readiness metadata is cached by model/vocab signature.

---

# 7. FRONTEND HARDENING DIARY

### Demo/static behavior removed
Removed:
- fake translation ticker;
- fake production translation fallback;
- hardcoded benchmark claims;
- static dashboard activity;
- static history cards;
- mock calibration behavior.

### Dashboard
Now uses authenticated backend data:
- model readiness;
- translation counts;
- average confidence;
- average latency;
- recent translation activity;
- latest adapter.

### History
Now uses persistent API data with:
- search;
- date ranges;
- sorting;
- CSV export using authenticated fetch.

### Signer profiles
Now use real backend adapter listing + owner-scoped deletion.

### Evaluation
Only displays persisted evidence.
Unmeasured benchmark metrics are explicitly unavailable rather than fabricated.

### Calibration
Uses browser-side MediaPipe capture and submits synchronized pose/face landmarks to backend fitting.

### Live translation
Browser-side MediaPipe produces:
```text
pose [T,132]
face [T,1404]
```
and the frontend sends those to `/api/v1/translate`.

### Frontend API client
Implemented:
- configurable API endpoint;
- 20-second default request timeout;
- correct composition of timeout abort + caller abort;
- shared authenticated request path for live translation.

### CI syntax checks
Every frontend `.js` file is checked using:
```text
node --check
```

---

# 8. SECURITY / DEPLOYMENT DIARY

### CORS
Restricted to required methods/headers.

### Response security headers
Conservative security headers added.

### Current auth storage caveat
Browser bearer token remains in localStorage.
Known risk:
- same-origin JavaScript can access the token if XSS exists.

Production follow-up:
- consider HttpOnly secure cookie/BFF session strategy;
- add deliberate CSRF design.

### Rate limiting
Not implemented yet.

### Database
Current persistence:
```text
SQLAlchemy + SQLite
```

Render free-tier local persistence is ephemeral.
Therefore:
```text
demo/disposable deployment = acceptable
production durable persistence = NOT READY
```

A managed external database is required for durable production state.

---

# 9. ARCHITECTURAL BOUNDARY

The backend translation API expects pre-extracted keypoints:
```text
POST /api/v1/translate
```
with pose + face arrays.

It does NOT currently accept arbitrary raw video and run MediaPipe server-side.

The intended production pipeline remains:
```text
camera
 -> browser MediaPipe
 -> keypoints
 -> backend
 -> model
```

---

# 10. TEST / CI DIARY

Current GitHub workflow checks:
1. install backend dependencies;
2. compile Python app/tests;
3. run pytest;
4. run frontend JS syntax checks.

Historical Claude report:
```text
52/52 tests passing
```
plus Python/JS/YAML/evaluation/overfit checks.

Status of that evidence:
```text
CLAUDE-REPORTED
```
not independently reproduced by this agent.

Latest workflow status after current commits:
- no workflow run was available for the inspected recent commits through the connector;
- therefore current CI is `NOT VERIFIED`.

---

# 11. CURRENT MODEL STATE / CRITICAL LESSON

The most recent observed checkpoint failed semantic validation like this:
```text
TRAIN -> ' '
VAL   -> ' '
space ratio -> 1.0
meaningful tokens -> 0
```

Do NOT interpret:
```text
loss reduction
```
as sufficient evidence.

Do NOT interpret:
```text
non-blank token exists
```
as sufficient evidence.

The model must demonstrate actual character-level recovery on real data.

---

# 12. REQUIRED ACCEPTANCE GATES

## Gate A — Dataset integrity
Must prove:
```text
metadata exists
pose exists
face exists
UIDs unique
pose = [T,132]
face = [T,1404]
T aligned
finite values
valid targets
CTC alignment valid
```

## Gate B — Semantic CTC overfit
Must prove all of:
```text
finite loss
meaningful loss reduction
not blank-only
not space-only
multiple meaningful tokens
acceptable CER
```

The tiny sanity gate is intended to answer:
```text
Can the current model + preprocessing + target encoding + CTC + optimizer
memorize at least a tiny real-data sample?
```

## Gate C — Full clean training
Only after Gate B passes.

## Gate D — Train + held-out semantic acceptance
At minimum:
- multiple training examples;
- multiple held-out examples;
- meaningful predictions;
- no collapse;
- vocabulary/checkpoint compatibility.

## Gate E — Real-video validation
The actual validation notebook must use real ISL videos and report:
- ground truth;
- prediction;
- confidence;
- blank ratio;
- CER;
- frame count;
- feature shapes;
- model readiness.

## Gate F — Application regression
Must verify:
- auth;
- dashboard;
- translation;
- calibration;
- adapter ownership;
- history;
- CSV export;
- evaluation;
- readiness.

---

# 13. KNOWN ERROR CATALOG

Keep these examples searchable for future agents:

```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```
-> MediaPipe runtime mismatch.

```text
ModuleNotFoundError: No module named 'mediapipe.python'
```
-> incompatible MediaPipe package/API combination.

```text
ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value
```
-> isolated MediaPipe runtime inherited Colab matplotlib backend.

```text
NotebookMissingRequiredFieldsError: nbformat_minor
```
-> invalid notebook structure.

```text
Missing metadata CSV: /content/VisionBridge/data/processed/isltranslate/ISLTranslate.csv
```
-> validation notebook assumed unavailable processed training data.

```text
no video file at .../fever (2).mp4
```
-> extension case mismatch with `.MP4` source.

```text
FileNotFoundError: ... good (3).npy
```
-> extraction failed before producing feature artifact; never load output blindly after a failed extraction.

```text
Author identity unknown
```
-> configure Git name/email in Colab before local commit.

```text
Unexpected file staged/changed: data/model_check/
```
-> model push whitelist correctly detected an unrelated directory.

```text
PREDICTED: (no sign detected)
BLANK RATIO: 1.0
```
-> historical CTC blank-collapse checkpoint.

```text
PREDICTED: ' '
SPACE RATIO: 1.0
```
-> latest observed CTC space-token collapse.

---

# 14. IMPORTANT REPOSITORY FILE ROLES

## `backend/app/models/base_model.py`
Defines the model and inference-time frozen loader.

Important:
```text
frozen inference loader != training model
```

## `backend/app/training/train_base_model.py`
Actual base-model training entry point.

## `backend/app/training/overfit_sanity.py`
Pre-training semantic real-data optimization gate.

## `backend/app/training/isltranslate.py`
Dataset/tokenizer/collation and contract enforcement.

## `backend/services / routes`
Runtime inference/calibration/auth/etc.

## `notebooks/train_base_model_colab.ipynb`
Canonical one-click Colab training workflow.

## `notebooks/validate_base_model_colab.ipynb`
Canonical real-video checkpoint validation workflow.

## `notebooks/train_base_model_lightning.ipynb`
Persistent/alternate training workflow.

## `AGENTS.md`
Shared memory / diary / coordination contract.

---

# 15. MULTI-AGENT WORK PROTOCOL

## Before editing
1. Read this file.
2. Inspect the relevant source files.
3. Identify the exact root cause.
4. State what is NOT the root cause when evidence exists.

## During editing
1. Modify only required files.
2. Preserve public contracts unless intentionally changing them.
3. Add/adjust tests for changed behavior.
4. Update this diary immediately after each meaningful change or error.

## After editing
1. Compile / syntax-check.
2. Run focused tests.
3. Run broader regression tests where feasible.
4. Re-read changed code.
5. Record exact runtime evidence.
6. Commit only the intended files.
7. Never claim runtime verification based solely on static inspection.

## If an agent encounters an error
Record:
```text
SYMPTOM
ROOT CAUSE
FIX
FILES CHANGED
TEST PERFORMED
RESULT
NEXT BLOCKER
```

## Commit rule
Commit messages should describe the actual change, e.g.:
```text
fix: reject CTC space collapse in semantic gate
fix: harden training gradient checks
fix: rebuild Colab training pipeline safely
docs: update multi-agent project diary
```

---

# 16. CURRENT REPOSITORY STATE

Branch:
```text
main
```

GitHub read/write:
```text
AVAILABLE
```

Current latest known main commit from the last update sequence:
```text
192f6f2096e32bfd727cd8f8eac4d0054d4236c4
```

Important training/gate commits from the recent pass:
```text
271598dd1d04dc0130d7c63a3c88cffada0dd0bd
0e2e6777fc6a5b18258750f859103b0a7d122b23
b898fa1f5c05406ee82d71a80f0c130b0fbbf386
e0d9eecf7dfee9b80a4f8b73adb6d307b319269e
192f6f2096e32bfd727cd8f8eac4d0054d4236c4
```

The exact branch head should always be rechecked before another agent pushes.

---

# 17. CURRENT BLOCKER BOARD

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
```

---

# 18. CURRENT NEXT STEPS FOR THE NEXT AGENT / USER

### Step 1
Start a fresh Colab runtime and use the current `main`.

### Step 2
Open:
```text
notebooks/train_base_model_colab.ipynb
```

### Step 3
Run from cell 1 to the end.

Expected first gates:
```text
GPU detected
MediaPipe 0.10.21
real dataset downloaded
collision-safe processed data
DATASET INTEGRITY: PASS
```

### Step 4
Run the semantic overfit gate.

It must NOT produce:
```text
prediction = ' '
space ratio = 1.0
```

It must produce meaningful non-space characters and acceptable CER.

### Step 5
Only after the semantic gate passes, run full training.

### Step 6
Post-training acceptance must use multiple train + held-out samples and reject trivial collapse.

### Step 7
Push the checkpoint + vocabulary only after semantic acceptance.

### Step 8
Open:
```text
notebooks/validate_base_model_colab.ipynb
```

Run several real unseen ISL videos, not just one cherry-picked clip.

### Step 9
Run backend regression tests and frontend syntax checks.

### Step 10
Verify deployed `/api/v1/health` and `/api/v1/ready`.

### Step 11
Verify real browser:
```text
auth
 -> dashboard
 -> translation
 -> history
 -> calibration
 -> adapter profile
 -> evaluation
```

### Step 12
Only after the base model and application are proven should the project scale the training data from ~8.5 GB toward 40 GB.

---

# 19. SCALING NOTE: 8.5 GB -> 40 GB

The project can be designed to train on substantially more than 8.5 GB, but scaling storage alone does not fix a collapsed model.

Required design for larger data:
```text
stream/extract incrementally
 -> collision-safe persistent IDs
 -> shard processed features
 -> bounded-memory DataLoader
 -> pinned memory
 -> multiple workers
 -> resumable checkpoints
 -> deterministic split manifest
 -> metrics/early stopping
 -> held-out semantic evaluation
```

Do not simply load all raw video/features into RAM.

40 GB should be a scaling phase only after the smaller pipeline proves semantic correctness.

---

# 20. RELEASE VERDICT

## Current status

# NOT READY

Why:
- the historical checkpoint exhibited complete CTC collapse;
- the latest observed checkpoint exhibited 100% space-token collapse;
- the semantic gate has now been strengthened but still requires a fresh Colab runtime proof;
- current CI/E2E claims are not independently verified;
- durable production persistence, rate limiting, and production-grade browser credential storage remain open.

The codebase is substantially hardened, but the central product promise remains unproven until:
```text
REAL ISL VIDEO
 -> REAL KEYPOINTS
 -> TRAINED CHECKPOINT
 -> NON-TRIVIAL DECODING
 -> CORRECT ENGLISH
```
is demonstrated on real data.

---

# 21. GOLDEN RULES FOR ALL FUTURE AGENTS

1. Never trust a `PASS` string without understanding the assertion behind it.
2. Never confuse inference-time freezing with training-time trainability.
3. Never treat lower CTC loss as proof of semantic learning.
4. Never treat a non-blank token as semantic success.
5. Never push a checkpoint that has not passed real-data semantic validation.
6. Never assume a file exists just because a previous cell was supposed to create it.
7. Never hard-code case-sensitive assumptions for dataset video extensions.
8. Never silently discard unsupported target characters.
9. Never let temporary Colab artifacts enter the repository.
10. Never modify unrelated project files.
11. Never erase user work using destructive Git commands.
12. Record every meaningful event in this file.
13. Prefer reproducible diagnostics over repeated full retraining.
14. Separate `CODE FIXED`, `TESTED`, `RUNTIME VERIFIED`, and `CLAUDE-REPORTED` states.
15. When uncertain, stop before pushing another half-baked checkpoint.

---

# 22. MOST RECENT DIARY ENTRY

## 2026-08-27 — Full-history consolidation

Request:
- consolidate the complete history from the beginning of the ChatGPT project-debugging conversation into `AGENTS.md`;
- preserve failures, fixes, experiments, current model diagnosis, training rules, notebook state, and next steps for other agents.

Action:
- replaced the shorter diary with this consolidated master record;
- retained the existing engineering contract and expanded it into a chronological failure/fix record;
- explicitly recorded the MediaPipe, notebook, dataset, UID, CTC blank, CTC space-collapse, inference-freeze confusion, training-path hardening, Git/push, frontend/backend, security, CI, and scaling history;
- recorded current blockers and the required next execution order.

Status:
```text
AGENTS.md UPDATED
RUNTIME VERIFICATION OF THE NEW DIARY FILE: COMMIT SUCCESS IS THE ONLY LOCAL WRITE EVIDENCE
MODEL QUALITY: NOT VERIFIED
RETRAINING: BLOCKED UNTIL SEMANTIC GATE PASSES
```

Future agent instruction:
```text
READ THIS FILE FIRST.
DO NOT REPEAT ALREADY RESOLVED INVESTIGATIONS WITHOUT NEW EVIDENCE.
```
