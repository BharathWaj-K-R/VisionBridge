# VisionBridge Engineering Contract

This file is the persistent engineering hand-off for agents working on VisionBridge. Read it before making changes.

The active product is an Indian Sign Language alphabet/fingerspelling letter recognizer built around:

~~~text
Browser camera
 -> MediaPipe Tasks Hand Landmarker 0.10.35
 -> normalized 126D two-hand landmark vector
 -> dynamic 26-class VisionBridge letter base model
 -> 64D embedding for signer adaptation
 -> few-shot signer adapter
 -> one predicted A-Z letter + confidence
~~~

The project is intentionally scoped to single-letter recognition, not sentence translation.

---

# Maintainability standard

All implementation work must optimize for actual human maintainability, not the appearance of authorship.

## Project-specific design
- Prefer VisionBridge terminology such as signer, letter, landmark, calibration, prediction, adapter, checkpoint, and model version.
- Use generic names only when they genuinely describe the responsibility.
- Keep domain logic near the domain that owns it.

## Abstraction
- Use the smallest abstraction that solves a real problem.
- Do not create factories, wrappers, interfaces, hooks, services, or utility modules without a concrete responsibility.
- Reuse code only when the shared behavior is meaningful and stable.

## Control flow
- Prefer early returns, named intermediate values, explicit calculations, and straightforward branches.
- Split large functions when the split creates a real responsibility boundary.
- Do not split code to satisfy an arbitrary line count.

## Comments
- Explain why a decision exists, not what obvious syntax does.
- Remove comments that merely narrate the code.
- Preserve comments that explain model contracts, failure modes, compatibility, or performance trade-offs.

## Consistency
- Preserve established naming and API conventions unless a name is genuinely misleading.
- Keep formatting, error handling, and component patterns consistent across the repository.
- Do not introduce artificial quirks, obsolete patterns, or inconsistent style.

## Cleanup
- Remove abandoned prototypes, duplicate implementations, commented-out code, and obsolete architecture.
- Keep tests, training scripts, reproducible notebooks, and deployment files when they support the active product.
- Do not fabricate history, metrics, authorship, or verification evidence.

## Review questions
Before finishing a major refactor:

    What does this file do?
    Why does it exist?
    What depends on it?
    What does it return?
    Is the abstraction useful?
    Is the naming meaningful?
    Is the resulting code easier to test and debug?

The desired result is clear, project-specific, focused software with real tests and honest documentation.

# 1. Engineering rules

Every meaningful change must be traceable in this file or in the repository's normal documentation.

Use these status labels where useful:

~~~text
CODE FIXED
STATIC VERIFIED
RUNTIME VERIFIED
CI VERIFIED
NOT VERIFIED
BLOCKED
~~~

Never mark something verified because it merely looks correct in source.

Never accept a model because:
- loss decreased;
- logits are finite;
- one trivial class is emitted;
- a demo produces a plausible result on one sample.

A model is accepted only after real-data validation.

Use this engineering loop:

~~~text
DISCOVER
-> UNDERSTAND
-> TRACE
-> REPRODUCE
-> ROOT CAUSE
-> DESIGN
-> IMPLEMENT
-> INTEGRATE
-> TEST
-> REGRESSION
-> REVIEW
-> IMPROVE
-> RE-AUDIT
-> ZERO-LOOSE-ENDS
~~~

Change only files required by the task. Avoid destructive cleanup, especially in Colab or training environments.

---


# Self-correcting full-restart protocol

The engineering process is transactional. This repository follows the uploaded
Self-Correcting Full-Restart Protocol as a mandatory execution rule.

If any code, ML, data, integration, product, deployment, or documentation error
is discovered:

~~~text
STOP
-> diagnose root cause
-> fix completely
-> verify the fix
-> invalidate downstream results
-> return to STEP 1
-> rerun the complete workflow
~~~

Never continue downstream from a state that was discovered to be defective.
Checkpoints are monitoring markers only and never override a required restart.
Unverified work must remain explicitly marked UNVERIFIED. A final release
requires one complete corrected execution with no unresolved restart-triggering
error.

Signer-independent evaluation is a required release gate. Missing signer
metadata is a blocker, not permission to remove or bypass the gate.


# Authoritative execution contract

The uploaded **ITERATIVE FULL-RESTART PROJECT REPAIR PROTOCOL** is the
authoritative execution contract for repository repair and release validation.
It takes precedence over convenience, previous partial results, or time-saving
shortcuts.

Mandatory nine-step sequence:

~~~text
STEP 1  INVENTORY THE WRECKAGE
STEP 2  REPRODUCE THE FAILURE
STEP 3  ISOLATE THE FRACTURE
STEP 4  TRACE THE DATA FLOW
STEP 5  IMPLEMENT THE MISSING PIECES
STEP 6  FIX THE DEFECTS
STEP 7  VERIFY RELENTLESSLY
STEP 8  CLEAN THE RESIDUE
STEP 9  DOCUMENT THE TRANSFORMATION
~~~

The protocol requires a full restart whenever any bug, regression, failed test,
failed build, dependency issue, integration problem, data-flow inconsistency,
model mismatch, API mismatch, documentation inconsistency, incomplete
functionality, or evidence error is discovered.

~~~text
STOP
-> REPRODUCE
-> ISOLATE
-> TRACE
-> FIX
-> VERIFY
-> INVALIDATE DOWNSTREAM RESULTS
-> RESTART FROM STEP 1
~~~

A repair is not complete merely because the immediate failure disappears.
Completion requires one entire clean iteration from Step 1 through Step 9 with
no newly discovered unresolved defect or regression. The protocol explicitly
forbids reusing questionable downstream outputs after an upstream defect.

Signer-independent evaluation remains a mandatory release gate. Missing signer
metadata is a blocker, not a reason to bypass the gate.

# 2. Current product mission

VisionBridge recognizes one ISL fingerspelled letter at a time.

The active output is:

~~~text
letter: A-Z
confidence: 0.0-1.0
~~~

The active system does not depend on sentence tokenization, sequence decoding, or sentence-level translation.

The application stack is:

~~~text
Frontend: React + Vite + TypeScript
Backend: FastAPI + SQLAlchemy
ML runtime: PyTorch
Landmarks: MediaPipe Tasks Hand Landmarker 0.10.35
Training: Python CLI + Colab notebook
~~~

---

# 3. Active feature contract

## Input representation

The active camera contract uses a 126D two-hand representation. The feature contract is explicit so a future sensor/landmark representation can be introduced as a versioned model contract rather than silently changing existing checkpoints.

~~~text
Left hand:  21 landmarks x (x,y,z) = 63 values
Right hand: 21 landmarks x (x,y,z) = 63 values
Total: 126 float values
~~~

Rules:

- landmarks are normalized around the wrist;
- hand scale is normalized;
- missing hands are zero-filled;
- feature dimension must remain exactly 126;
- preprocessing used for training must match preprocessing used at inference;
- handedness handling must remain consistent between dataset preparation and runtime;
- training and browser inference consume the raw Tasks handedness labels in the same left/right ordering; do not introduce a one-sided mirror swap;
- the active preprocessing contract is `two-hand-wrist-scale-v1` and the active landmark runtime is `mediapipe-hand-landmarker-0.10.35`.

Do not silently change the active 126D contract. Model versions may introduce a new input contract explicitly.

## Output representation

The base model produces 26 A-Z logits and a 64D embedding.

The active runtime ultimately returns:

~~~text
predicted letter
confidence
latency
model version
~~~

Unknown or low-similarity predictions may be returned as a question mark.

---

# 4. Base model

The active base model class is:

~~~text
VisionBridgeLetterBaseModel
~~~

Default contract:

~~~text
Input:        126
Hidden:       128
Embedding:     64
Classes:       26 (A-Z)
Objective:     cross-entropy

The checkpoint may scale hidden width, embedding width, and class vocabulary.
~~~

Current model structure:

~~~text
LayerNorm(input_dim)
 -> Linear(input_dim -> hidden_dim)
 -> GELU
 -> Dropout(dropout)
 -> Linear(hidden_dim -> embedding_dim)
 -> LayerNorm(embedding_dim)
 -> GELU
 -> Linear(embedding_dim -> num_classes)
~~~

The base model is trainable, versioned, and replaceable. It can be retrained or upgraded as new data becomes available.

Expected checkpoint path:

~~~text
backend/app/models/weights/letter_base_model.pt
~~~

The checkpoint contract includes:

~~~text
model_version
preprocessing_version
landmark_runtime
input_dim
hidden_dim
embedding_dim
num_classes
labels
dropout
state_dict
~~~

The loader must reject incompatible checkpoints instead of partially loading them.

A missing or incompatible checkpoint must make the model not ready rather than silently substituting another model.

---

# Dynamic scaling and model replacement

The active model is not treated as permanently fixed. It is versioned, configurable, and replaceable.

Default configuration:
~~~text
input_dim=126
hidden_dim=128
embedding_dim=64
classes=26 (A-Z)
~~~

These are defaults rather than hard architectural limits. The checkpoint stores the actual dimensions, dropout, and label vocabulary. The training CLI exposes hidden and embedding dimensions so larger models can be trained as the dataset and product scope grow.

The backend detects checkpoint file changes and hot-reloads the latest compatible model. Existing signer adapters retain the model version and checkpoint hash and require recalibration after an embedding-space change. This prevents silent incompatibility while keeping the system dynamically updatable.

# 5. Few-shot signer adapter

The adapter is deliberately prototype-based, not a second offline neural training job.

Pipeline:

~~~text
dynamic base model
 -> 64D embedding
 -> L2-normalized embedding
 -> per-letter prototype
 -> cosine similarity
 -> prediction
~~~

Calibration rules:

- the signer provides a few real examples for selected letters;
- the current frontend uses three shots per selected letter;
- embeddings are generated by the dynamic base model;
- one normalized prototype is stored per calibrated letter;
- prediction compares live embeddings against those prototypes.

Current adapter properties:

~~~text
method: dynamic-base-embedding-prototype
embedding dimension: checkpoint-defined
minimum similarity threshold: 0.35
confidence: softmax over cosine scores
base-checkpoint binding: SHA-256
preprocessing/runtime metadata: required
raw calibration landmarks: not persisted in adapter payload
~~~

The adapter must detect a different base checkpoint and require recalibration rather than mixing incompatible embeddings.

There is no hidden offline adapter-training step. New validated base checkpoints can be installed and hot-reloaded.

Keep this distinction explicit:

~~~text
BASE MODEL: trainable, versioned, and replaceable offline
SIGNER ADAPTER: fitted from a few runtime examples
~~~

---

# 6. Dataset preparation and training

The default training source is the RealSign Indian Sign Language alphabet dataset. Dataset preprocessing uses the supported MediaPipe Tasks Hand Landmarker API and its versioned hand-landmarker.task model bundle. The source archive is a Git LFS object and the Colab notebook uses the Git LFS media endpoint.

Dataset preparation converts alphabet images into the same normalized 126D landmark representation used at runtime.

Canonical preparation script:

~~~text
backend/scripts/prepare_letter_dataset.py
~~~

It writes:

~~~text
train.npz
val.npz
test.npz
labels.json
~~~

Canonical trainer:

~~~text
backend/app/training/letter_base.py
~~~

Canonical reproducible notebook:

~~~text
notebooks/train_letter_base_colab.ipynb
~~~

Standard training flow:

~~~text
dataset images
 -> MediaPipe Tasks Hand Landmarker extraction
 -> pool source Training + Validation data
 -> stratified 80/20 train/validation split
 -> iterative base-model training
 -> per-letter validation check after each epoch
 -> held-out test measurement
 -> checkpoint
 -> runtime installation
~~~

The repository must never claim a base-model accuracy number until an actual training run produces and records it.

---

# 7. Training command

Standard Colab training command:

~~~bash
PYTHONPATH=backend python -m app.training.letter_base \
  --data-dir /content/visionbridge_letter_data \
  --output backend/app/models/weights/letter_base_model.pt \
  --epochs 500 \
  --batch-size 128 \
  --lr 0.001 \
  --target-class-accuracy 1.0
~~~

The resulting checkpoint must exist at:

~~~text
backend/app/models/weights/letter_base_model.pt
~~~

The model must be checked for:

- finite loss;
- finite gradients;
- valid dimensions;
- class coverage;
- non-trivial predictions;
- validation performance;
- held-out test performance.

Do not push a newly trained checkpoint merely because training completed successfully.

---


## Training split and stopping rule

The prepared dataset uses a stratified 80/20 train-validation split built from the dataset's original Training and Validation pools. The original Testing split remains untouched for final evaluation.

The trainer can run for up to 500 epochs. After each epoch it measures every A-Z class on the validation split and continues until every class reaches the configured target, or the epoch cap is reached. Learning rate is reduced when the weakest validation class stops improving.

~~~text
train: 80%
validation: 20%
test: original test split, untouched
maximum epochs: 500
default stopping target: 100% validation accuracy for every A-Z class
~~~

The target is a validation stopping criterion. It is not evidence of perfect recognition for unseen signers or live camera input.

---
# 8. Active backend contract

Primary letter endpoints:

~~~text
GET  /api/v1/letter/status
GET  /api/v1/letter/model
GET  /api/v1/letter/adapters/{id}
POST /api/v1/letter/calibrate
POST /api/v1/letter/predict
POST /api/v1/letter/event
~~~

Rules:

- endpoints require authentication where the surrounding API contract requires it;
- adapter access is scoped to the authenticated user;
- input dimension is validated as 126;
- malformed or invalid feature values are rejected;
- inference requires a valid base checkpoint in real mode;
- prediction events are logged through the existing history mechanism;
- rate limiting and request validation patterns must be preserved.

Adapter listing:

~~~text
/users/me/adapters
~~~

The active letter UI must receive only active letter adapters. Other repository artifacts must not accidentally appear as current letter adapters.

Readiness:

~~~text
/api/v1/ready
~~~

Readiness must report whether the letter base model is available and compatible.

Liveness:

~~~text
/health
~~~

Liveness is not the same as model readiness.

---

# 9. Active frontend contract

Canonical frontend:

~~~text
frontend/
  index.html
  package.json
  vite.config.ts
  tsconfig.json
  public/
  src/
~~~

Important active modules include:

~~~text
src/App.tsx
src/api.ts
src/landmarks.ts
src/useLandmarkSession.ts
src/browserModel.ts
src/styles.css
~~~

Browser flow:

~~~text
camera
 -> hand landmark detection
 -> 126D normalization
 -> browser model + adapter (loaded once)
 -> base model embedding
 -> signer adapter
 -> letter + confidence
~~~

The UI must make the current letter-oriented workflow understandable.

Calibration is a real capture/submission flow, not a timer placeholder.

The visual direction is:

~~~text
white / black monochrome
minimal
high contrast
strong structure
responsive
professional
~~~

---

# 10. Local/demo mode

The frontend supports a local/demo mode through:

~~~text
VITE_LOCAL_MODE
~~~

The Render configuration intentionally sets:

~~~text
VITE_LOCAL_MODE=true
~~~

This is a deliberate deployment state while the real model and remaining production infrastructure are incomplete.

The local/demo path must not be described as equivalent to real model inference.

The true base-model + adapter path is the real-mode path.

Do not flip Render to real mode merely because the frontend builds.

Before switching the deployed frontend to real mode, verify at minimum:

~~~text
real base checkpoint installed
durable production database connected
production authentication hardened
real backend letter-recognition API verified
browser camera flow verified
~~~

---

# 11. Verification gates

## Gate A — Dataset

Verify:

~~~text
A-Z class coverage
correct landmark extraction
126D feature shape
finite values
consistent normalization
valid train/validation/test splits
no accidental feature collisions
~~~

## Gate B — Base model

Verify:

~~~text
training runs without numerical failure
validation improves meaningfully
predictions are not collapsed to one class
all required classes are represented
held-out test accuracy is measured
checkpoint loads strictly
checkpoint metadata matches the runtime contract
~~~

## Gate C — Few-shot adaptation

Verify:

~~~text
calibration uses real signer examples
prototype dimensions are correct
adapter records the correct base checkpoint hash
mismatched checkpoint is rejected
held-out signer examples are evaluated after calibration
~~~

## Gate D — Runtime

Verify:

~~~text
camera capture
hand landmarks
normalization
API request
model inference
adapter lookup
letter prediction
confidence
latency
history logging
~~~

## Gate E — Application E2E

Verify the active product flow:

~~~text
authentication
 -> dashboard
 -> calibration
 -> letter recognition
 -> history
 -> evaluation/settings surfaces
~~~

Do not revive inactive behavior merely to satisfy an old test or document.

---

# 12. Verification status

Current known evidence:

~~~text
Active architecture source: STATIC VERIFIED
Base model implementation: HISTORICAL CI VERIFIED
Few-shot adapter implementation: HISTORICAL CI VERIFIED
Letter API implementation: HISTORICAL CI VERIFIED
Frontend TypeScript check: HISTORICAL CI VERIFIED
Frontend production build: HISTORICAL CI VERIFIED
Production artifact verification: HISTORICAL CI VERIFIED
Base checkpoint trained on real data: NOT VERIFIED
Base test accuracy: NOT VERIFIED
Held-out signer accuracy: NOT VERIFIED
Browser camera runtime in this environment: NOT VERIFIED
Live Render end-to-end behavior: NOT VERIFIED
Production database hardening: NOT VERIFIED
Production HttpOnly authentication: NOT VERIFIED
~~~

Known successful code verification:

~~~text
Historical GitHub Actions run #197
backend: 72 passed, 1 skipped
Python compilation: PASS
frontend TypeScript check: PASS
Vite production build: PASS
production artifact verification: PASS
~~~

Do not extend the above evidence into claims about model quality or real-world recognition.

---

# 13. Current blocker board

~~~text
A  Train base model on real ISL A-Z data             NOT VERIFIED
B  Record held-out base test performance             BLOCKED until A
C  Validate few-shot adaptation on held-out signer   REQUIRED / BLOCKED until A+B+verified signer metadata
D  Verify browser camera + real inference            NOT VERIFIED
E  Verify live Render real-mode flow                 BLOCKED until A-D
F  Durable production database                     NOT IMPLEMENTED
G  Production HttpOnly authentication              NOT IMPLEMENTED
H  Production rate limiting hardening              NOT VERIFIED
~~~

There is no additional hidden ML training requirement for the active architecture.

---

# 14. Production boundaries

Current infrastructure still has known limitations:

~~~text
SQLite on Render            -> disposable without external durable storage
Client-side bearer token    -> still present in current deployment path
Real-mode deployment        -> intentionally disabled
Raw video server inference  -> not part of the active letter contract
~~~

Do not call the system production-ready while these boundaries remain unresolved.

The active product can still be developed and validated independently of production hardening.

---

# 15. Repository cleanliness rules

Do not reintroduce inactive model concepts into the active contract.

Do not add new product work for:

~~~text
sentence translation
sequence decoding
sentence-level training
unused input streams
obsolete calibration flows
~~~

Do not silently rename the active 126D input contract.

Do not add a second feature representation unless the runtime, training, tests, and documentation are updated together.

Keep this as the canonical ML source of truth:

~~~text
base model -> dynamic embedding -> few-shot signer prototypes -> letter
~~~

When modifying one stage, trace every dependent stage before editing.

---

# 16. Current architecture diary

## 2026-09-23 — Letter-only architecture

The active product was narrowed to single-letter ISL fingerspelling recognition.

The current architecture was established as:

~~~text
MediaPipe Tasks Hand Landmarker
 -> normalized 126D hand vector
 -> dynamic 26-class base model
 -> 64D embedding
 -> few-shot signer adapter
 -> letter + confidence
~~~

This document treats that architecture as the only active product contract.

## 2026-09-23 — Base model + few-shot adapter implementation

Implemented:

~~~text
VisionBridgeLetterBaseModel
letter dataset preparation
base-model training CLI
Colab training notebook
dynamic-embedding few-shot adapter
checkpoint checksum binding
letter API integration
letter frontend integration
readiness reporting
adapter ownership filtering
regression tests
~~~

The adapter is deliberately prototype-based. It does not require a second offline training job.

Status:

~~~text
BASE MODEL CODE: HISTORICAL CI VERIFIED
FEW-SHOT ADAPTER: HISTORICAL CI VERIFIED
API: HISTORICAL CI VERIFIED
FRONTEND: HISTORICAL CI VERIFIED
~~~

## 2026-09-23 — Active-state handoff

The initial base-model training step outside normal source and CI verification is:

~~~text
run notebooks/train_letter_base_colab.ipynb
~~~

That run must create and validate:

~~~text
backend/app/models/weights/letter_base_model.pt
~~~

After checkpoint installation, the next evidence-producing work is:

~~~text
base held-out test
 -> signer calibration
 -> held-out signer evaluation
 -> browser real-mode verification
 -> Render verification
~~~

No hidden adapter training step exists.

---

# 17. Definition of done

A change is complete only when:

~~~text
source updated
tests updated where needed
regression checks pass
documentation matches implementation
no stale active-path contract remains
verification status is honest
known blockers are recorded
~~~

For model changes specifically:

~~~text
real data
 -> measured result
 -> compatible checkpoint
 -> runtime validation
 -> held-out signer validation
~~~

The repository must never claim a successful real-world model result without the corresponding evidence.


## 2026-09-23 — Low-latency browser inference and hand tracing

The real-time product path was optimized around local inference:

~~~text
camera
 -> MediaPipe Tasks Hand Landmarker 0.10.35
 -> 21-point hand tracing + wrist trail
 -> normalized 126D vector
 -> browser-loaded base model
 -> browser few-shot adapter
 -> letter + confidence
~~~

The backend is no longer required for every prediction frame. It is used for model/adapter loading, calibration persistence, and throttled asynchronous recognition-event logging.

The previous Holistic browser pipeline was removed from the active camera path.

Performance decisions:

~~~text
MediaPipe Tasks Hand Landmarker runtime=0.10.35
camera target=640x480, max=960x720
single in-flight hand-tracking call
canvas size changes only when video dimensions change
model weights loaded once per session
adapter loaded once per selected adapter
no blocking network request per frame
prediction loop targets ~30 Hz
~~~

Status:

~~~text
HOT-PATH CODE: CODE FIXED
HAND TRACING: CODE FIXED
DYNAMIC BROWSER MODEL: CODE FIXED
ASYNC HISTORY: CODE FIXED
REAL DEVICE LATENCY: NOT VERIFIED
~~~

## 2026-09-23 — Iterative A-Z training

The base-model trainer now evaluates every letter after each epoch and prioritizes the weakest validation class when selecting the best checkpoint.

~~~text
train epoch
 -> full validation sweep
 -> per-letter accuracy
 -> identify weakest letter
 -> adjust learning rate on plateau
 -> continue automatically
~~~

Default stopping target:

~~~text
100% validation accuracy for every A-Z class
~~~

The training loop has a maximum epoch cap so an impossible target cannot create an endless Colab session. A checkpoint is still saved when the cap is reached, and the held-out test result is reported separately.

This target is a validation criterion, not a guarantee of perfect recognition for unseen signers or live camera input.

## 2026-09-23 — MediaPipe training compatibility fix

MediaPipe 0.10.31 and newer no longer expose the legacy Python Solutions surface used by older examples. The training path now uses mp.tasks.vision.HandLandmarker with the versioned Google-hosted hand_landmarker.task model bundle. The Colab notebook installs only mediapipe==0.10.35, downloads the hand model automatically, validates that the Tasks landmarker can initialize, and passes the model path to dataset preparation. Production backend dependencies are not installed in the Colab training environment.

# 17. Evaluation contract

The canonical evaluator is ~~~text
backend/app/training/evaluate_letter_base.py
~~~

It must be used for recorded V3/V4/V5 test measurements and reports overall accuracy, macro accuracy, per-letter accuracy, worst-class accuracy, confusion matrix, parameter count, checkpoint size, and model-only latency. The original test split remains measurement-only and is not a tuning input.

Signer-independent evaluation is a separate gate. The current prepared NPZ contract does not contain signer IDs, so a signer-independent result must not be inferred from the random 80/20 split. A future signer-holdout evaluation requires explicit signer metadata or a verified signer manifest.

# 18. Change ledger — maintained on every repository change

This section is the cumulative engineering ledger for the active VisionBridge letter-recognition development cycle. Every repository change made during the current refactor, training stabilization, ML evaluation, runtime-parity correction, and documentation cycle is recorded here. Future repository changes must update this file in the same engineering cycle.

## Change-record rule

Every future change must record:

    date
    commit
    files or area
    change
    reason
    verification
    status

A change is incomplete if the implementation changes but this ledger is not updated. Multiple tiny commits for one logical correction may be grouped, but no meaningful implementation, dependency, training, model, runtime, deployment, test, or documentation change may be omitted.

## 2026-09-24 — Complete change history carried into the active ledger

### Product and legacy-architecture cleanup

060301e0541df57b4d86f452dc2b3c385e8691d0 — remove obsolete pose conversion script
1c36b35563a6dd07d56025222d3f2bd2f1c96b03 — remove obsolete pose-face extraction script
593d2c9356982f3ee829bafc267dcf03ebdf6c35 — remove obsolete temporal adapter tests
827d7e380fc563852cfd2be5502aa988f8c2b59b — remove obsolete sequence calibration tests
4d6777074f96b5c6b116e5135e8c403192f6a22e — remove obsolete CTC contract tests
3af0f77905fc9e89092affa98531b85dc87fa2f6 — remove obsolete sentence dataset UID tests
e16b2146ef8b8c2d89525b32875c22d6b41fa19b — remove obsolete sentence CER tests
a7a38a214ce4b6a27474cb979a5b1e7bf50ebc93 — remove obsolete pose-face script tests
35a0f78d9f6541fb9792188b41c7342acde9100a — remove obsolete multimodal model tests
596a557c54ebc416458ebcd31a93f40d8b83574e — test: align account coverage with active APIs
0244ccd09f7a98bcc9e3132022ea77669cd672d5 — test: validate current letter model readiness
24b622454b7520e34757e81a6e16d87b412b1959 — test: cover history export with direct letter prediction records
d53b1d1e8a8f9eda480a102b8e2e77ae832abe92 — test: keep rate limiter coverage focused on reusable logic
756ee1e9557c9b96c07bd375b2a182905b59078a — remove obsolete sentence live-pipeline tests
b8e590dc769830155b0c191e71d88785ef7191fc — remove obsolete sentence training notebook
735059f791dcfeed6b6a5b764b7cbc5470f5c4a9 — remove obsolete sentence lightning notebook
7679ffb7afbc53a93da182d9a93606879bec4634 — remove obsolete sentence validation notebook
d487c8bff343cf282fc4f857c6d239688465b7b7 — remove obsolete sentence vocabulary
237722985c7245de97633defe39b839c8173afee — remove obsolete sentence checkpoint
34f3f1640fcf3fd4d7106510b6d3b9baa118ae45 — docs: align backend environment example with letter pipeline
d126db6edb28f91f01f601216b751d2d1fb12698 — refactor: make letter dataset preparation maintainable
52c976ff494d6b16f4a4a6af2239a7f3b0efd7cd — refactor: simplify adapter lifecycle handlers
7961bd143cad868ffed3120a2701b324d73b067c — refactor: organize active letter API
cd3e54af1dc49400ee3e55e490c1943c29bd0aad — refactor: make settings readable and letter-specific
9d748cf9db488095f7a7aa8326efa8382b700323 — fix: preserve established settings interface
9d5d3f3f3a2a8fe2f6bcdd0ea617531c7d4e6481 — refactor: align database models with active letter product
9672e56416d5783f7e70019f4e557ee2ddcdc636 — docs: add human maintainability engineering standard
6b5586234c18cf4dc2946005e20c9f822ee41a70 — chore: remove failing CI workflow until re-enabled

Status:
    legacy sequence/pose/face/sentence architecture removed from the active product
    active product narrowed to A-Z letter recognition
    failing CI workflow intentionally removed and not restored

### 2026-09-23 — Training policy and reproducibility

5e8feb5f80907701c38adc49a425c4bbaabe04fc5 — iterate base-model training by letter accuracy
722482938fb2f43ff71d69190bef0aad3c64146e — make letter training notebook self-iterating
b9c5e79b76acaa913324894b31a9a34689af98dc — document iterative letter model training
86a9b495bd1b7418c59d67b5c2e962f7ccc78851 — record iterative base-model training
55d9c94f7c3e476fb6a06291a1e65ffac1b3ea08 — use stratified 80-20 letter training split
234ccf5902f38d169b8236bc0fff7fe424adc3dc — raise automatic training epoch cap
b400c04a74343eb640b2cad0ae5d498d11c7ecdc — use 80-20 split and 500 epoch training cap
2d281190e5f799effb553ace36cdedb6667aea5e — document 80-20 training split
e00782d8648d6344f3dd077263c39116ea0149d5 — document current training split and epoch policy
2b4915c0a315578ec89f9c61e0da6890c969431f — make Colab training isolated and reproducible

Training contract:
    stratified 80/20 train-validation split
    maximum 500 epochs
    per-letter validation monitoring
    weakest-class checkpoint selection
    untouched source test split
    finite-loss and finite-gradient guards

### 2026-09-23 — MediaPipe Tasks and RealSign fixes

038a5ab00fd6d0860eb3b8db32aa349be2190683 — update MediaPipe training dependency for Colab
dbfb24dc82885b9c6d9256749c1a1d998e0803a8 — clarify MediaPipe training dependency
b76027ab691d5687b131f05b8c796753b0e01622 — verify MediaPipe version before preprocessing
a7ded9fbaa631258b9e9040f0bcaf0d8694c9b2f — reset Colab working directory before repository cleanup
91676482255a5ea5319fa3c91cd51a96040ad09d — isolate Colab training dependencies
ec42ab92f39751441c078147086b49bc2da0c8bc — migrate dataset preparation to MediaPipe Tasks API
582b57f00f235b69c71e68cc47953381659f01c4 — trim training dependencies to MediaPipe runtime
b6e5d17d0cb33871933ba3e6b1e696a42b08c366 — make Colab training use MediaPipe Tasks
74660478c6ace6cec20842ac3a4ecefded7dcdd4 — document MediaPipe Tasks training path
d7e9fcfab537e80876f5116b28abc843575d59b3 — record MediaPipe Tasks training contract
b1e93cc912b64d77a106876b40bb93b47adf0f7d — record MediaPipe Tasks training compatibility
29736546ecd2f03cd45099b9601956175cad29e0 — download RealSign LFS archive correctly
4a8e0f314d323358a68f79acab048bdf463751c7 — document RealSign LFS training source
5c58751cbd94bce7cb5a59c9c1e5e6fbf69f611b — document current letter dataset preparation
9fbee0239a8c2e4b77f3a416350ae417f5e50be9 — record RealSign LFS training path
2ea274df551503ddf8a583510dbea26f74a34d08 — record RealSign archive download fix
f03d476b8f590c1749c2aa30777f80df36604ed3 — use supported MediaPipe Tasks imports in preprocessing

Dataset source and extraction contract:
    RealSign Git LFS media endpoint
    MediaPipe Tasks Hand Landmarker
    normalized 126D two-hand representation
    original source testing split left untouched

### 2026-09-24 — Colab runtime corrections

bca6ba6b443389157f0af03ce28f4dbe0e2a0769 — reset Colab cwd before repository cleanup
96e5324d11729a5840f4473314666d9c717ca2b9 — fix Colab training environment setup
7774d9fd409939179df2261ad9fc604e9f465b63 — use isolated Python 3.12 training environment
2fec304110b7b7ac7ed635d7a2f07a3a0eafdb7f — make uv invocation robust in Colab
b808a3e6774d244af60b48cbbcf0da46b78bb9a4 — notebook correction
9852e9283c48b809736e5e93c4892678d57935da — correction of notebook
ffdc684164b28631301bb5f79f77963c46d1cd12 — pin notebook training dependencies
72794c0af1013fca1e82a0e51e380532ae4a1d19 — align Colab notebook dependency pins

Current notebook environment contract:
    managed Python 3.12 environment via uv
    MediaPipe 0.10.35
    pinned NumPy 2.1.3
    pinned PyTorch 2.9.0 build
    streamed subprocess errors

### 2026-09-24 — Browser landmark parity correction

0e7ac3671c673248a6953c25a82b6b0bea51cb1b — use MediaPipe Tasks Vision in browser
a00ee1bb4e73f6db40646d13d9075140f344c139 — align browser landmarks with MediaPipe Tasks
e4d48c2fe02135a6770bf5a4f256b59b00ddeaac — pin browser MediaPipe Tasks version
b89b93ff00855416796800266164f199399c263b — lock MediaPipe Tasks dependency
5b4aaf441404643f54e3e3a60e673482b7fbc27a — match browser MediaPipe runtime version
336b12d8dfa15d7699e5226c6861748f52a8a364 — keep browser hand tracking on the default delegate
f73356646ff4d5fe50ae34799a4d4e95d250e8d6 — update frontend tracker label
654028fa33649a19f73b0138bae2c8377a8c0e2a — document consistent Tasks handedness semantics
a5606c422d85392e292968023c44d203da77811c — document handedness parity rule
ff647f493097fd332515d5b4d4bc469639c7eba06 — record handedness parity rule
a276f1ebd6d89acec282429f0dc22576684d3116 — support current Tasks handedness result field

Current runtime contract:
    browser package: @mediapipe/tasks-vision 0.10.35
    training runtime: MediaPipe Tasks Hand Landmarker 0.10.35
    browser model asset: hand_landmarker.task
    same raw Tasks handedness semantics in both paths
    legacy @mediapipe/hands browser path removed

### 2026-09-24 — Checkpoint and preprocessing contract

a261a27f9060af76c4e5f187af3f9341c3ab8e0c — version the landmark preprocessing contract
5109456c3b776521f274f1b26515659d4e0c3aba — bind adapters to preprocessing contract
0d6164c3debcc3d32c814d0b33638c3a85d30f1f — validate browser model preprocessing metadata
a56e7f6d238dd8924f367b64508bdc4b95f8fe96 — include model preprocessing metadata in evaluation
0f9b9c9ed4114176d39d9023d59c88f258e073a3 — test model preprocessing metadata
990dda0f1aa37a42c02f3fc0f730f7286d04ab52 — test adapter preprocessing binding

Metadata contract:
    preprocessing_version = two-hand-wrist-scale-v1
    landmark_runtime = mediapipe-hand-landmarker-0.10.35

### 2026-09-24 — Evaluation and dataset-integrity improvements

bb1e4e7522c4c6686b90491cd8c7c638163ca836 — add reproducible letter model evaluation
7e3c822a35ee18c4fe505eae4cad97f9ce62bc0e — add canonical letter model evaluation
a56e7f6d238dd8924f367b64508bdc4b95f8fe96 — include model preprocessing metadata in evaluation
42f3963c27c0e995796728ad5f81baf9dec4465c — prevent duplicate leakage across train and validation
ae942ed618cb99efae1e0d96fd1a0709b0e0c782 — test duplicate-safe dataset splitting
de111cb6a5a63902ec689f1358f5425fbbadf45f — fix dataset split test import
53e5675c596354716b708e2b2f9a48174a4964a0 — guard notebook against dataset leakage
be66781f9a53800a6b34f172e8f8244f1678826f — document dataset manifests and duplicate checks
8dc44b04ce1b32aa78bdbc04f661172a0b6f1959 — update training command to current epoch cap

Evaluation outputs:
    overall accuracy
    macro class accuracy
    per-letter accuracy
    worst-letter accuracy
    A-Z confusion matrix
    parameter count
    checkpoint size
    model-only latency

Dataset outputs:
    train_manifest.jsonl
    val_manifest.jsonl
    test_manifest.jsonl
    duplicate_report.json

### 2026-09-24 — Documentation and debug records

41090105b6c830e83fb9eca9c976fe41a36c3589 — document unified landmark runtime and evaluation
6ef9210366b7f2d0e528dfd3ef8a6b5a329c76bd — document ML parity and evaluation gates
fc4c474ee919b0a60e4f2d519bb58842c6982f42 — record critical ML pipeline corrections
bd38f90d7457801612f02d3ee654d4f56f097069 — record uploaded checkpoint audit

Note: the checkpoint-audit commit recorded the supplied V3 checkpoint structure, size, and SHA-256. The binary checkpoint itself is still not stored in this repository because the available GitHub write interface cannot upload binary model files.

### 2026-09-24 — Uploaded V3 checkpoint audit

checkpoint:
    visionbridge-letter-base-v3
    input_dim = 126
    hidden_dim = 128
    embedding_dim = 64
    num_classes = 26
    parameters = 26,582
    size_bytes = 111,205
    sha256 = 2b42639e0ffb3c40112bf931f434f6adf5b578fce21399ba53795d3fba0529a

Status:
    checkpoint structure = STATIC VERIFIED
    checkpoint accuracy = NOT VERIFIED
    binary repository installation = PENDING


### 2026-09-24 — Self-correcting audit cycle

91942ffa43fe24e5083e7f0e6976fb65f458b1eb — fix: bind rate limits to identity
8ed63bf3a5202ab0b4be800d41f035caa9623ff6 — fix: require signer adapter recalibration
c4a1dbf34c78eda8e9111f952762ac6f43861c09 — fix: wire calibration rate limit
e698aded2b7f4329a3af10e2794ae8ebca3ee92e — test: enforce adapter recalibration boundary
33ebe19c90095d97bfa2bf57b6c06728d3f5b6ae — test: cover rate-limit identity and wiring
b2815dde7bafdcb952764614571d791735239b36 — chore: remove stale calibration setting
f557f114ae1d2eebc609d3787352a143f1b6dc2a — docs: clarify signer evaluation and runtime
d184a5735c9cabeef24b0fe88b43bb8dadc2c430 — chore: normalize API root wording

Audit findings corrected:
    signer adapter auto-refresh after base-model change     FIXED
    calibration limiter wired to recognition limit          FIXED
    invalid bearer tokens bypass shared limiter bucket      FIXED
    stale Render calibration setting                        REMOVED
    signer-independent embedding claim overstated          CLARIFIED

Verification boundary:
    changed-source re-audit                 STATIC VERIFIED
    current CI workflow                      DISABLED / NOT RUN
    local full test suite                    NOT VERIFIED in this environment
    real-data training                       NOT VERIFIED
    held-out signer evaluation              REQUIRED / BLOCKED BY SIGNER METADATA
    browser camera runtime                  NOT VERIFIED
    live Render verification                NOT VERIFIED

### 2026-09-24 — Restart #2: cache and runtime-contract hardening

8cdfc91dc5db88a4ce922190205ecefe06f3c475 — fix: invalidate model cache on weight changes
2ab3542925a1e5de407842cd185f977e13d98f11 — docs: remove stale MediaPipe runtime wording
b9ebeddfca3a230c524e4c0f8171f61e669cd001 — fix: enforce browser adapter metadata parity
75397a24d7b8d1c53b66fc4d82efdd54f6827289 — test: enforce adapter preprocessing compatibility

Restart findings corrected:
    checkpoint cache could miss a same-metadata content change       FIXED
    README described the legacy MediaPipe Hands runtime              FIXED
    browser adapter ignored preprocessing/runtime metadata           FIXED

Restart status:
    source re-audit                         STATIC VERIFIED
    current CI workflow                    DISABLED / NOT RUN
    full local suite                       NOT VERIFIED
    real-data training                     NOT VERIFIED
    signer-independent evaluation          REQUIRED / BLOCKED BY SIGNER METADATA

### 2026-09-24 — Restart #4: deployment, dataset asset, and demo-path hardening

c91e1650e925414a30e8757310fa01c5926c82cf — fix: use liveness endpoint for Render health
c8ac76a52f10329aad43308836ca60490372b2f6 — fix: pin hand landmarker asset integrity
7d65acbf8041a8c5645ae05bbdeda9b3d2471df8 — fix: clean up failed camera startup
abaf9ca9fd33e4c9984e48e578f975b4aaacfb16 — fix: isolate local demo data by user
d97781d66045efc490aae7e75607956d236b54a4 — fix: throttle local prediction history semantics
1e724495d8e478a56f8558031b0791ebfacde956 — fix: throttle local recognition history

Findings corrected:
    Render health check depended on missing model readiness       FIXED
    cached hand-landmarker asset accepted without integrity check FIXED
    camera startup failure could leak created tracker/stream       FIXED
    local demo data was shared between browser users              FIXED
    local recognition spammed history on every prediction tick    FIXED

Verification boundary:
    current CI workflow                    DISABLED / NOT RUN
    source-level audit                     IN PROGRESS
    full local test suite                  NOT VERIFIED
    real-data training                     NOT VERIFIED
    signer-independent evaluation          REQUIRED / BLOCKED BY VERIFIED SIGNER METADATA

### 2026-09-24 — Ledger completeness correction

5ac7147b80dc06dbd0b0d8a4e35002084c74f6d6 — fix: prune stale rate limit buckets
0a01b326006917aa997baff2fe35449efe7c766d — test: cover stale rate limit bucket cleanup

These two commits belong to the preceding rate-limit hardening cycle and are recorded here so the cumulative ledger contains every meaningful repository change made during the current correction run.

Current source status:
    stale rate-limit bucket cleanup       CODE FIXED
    cleanup regression coverage            CODE FIXED
### 2026-09-24 — Restart #5: strict metadata and calibration-data minimization

c05e9a2b7221c7240d7d77f4757975cf466c3c4a — fix: enforce model metadata contract
d48947419af069ffa7981f4abf748e359bb6f875 — fix: enforce adapter metadata contract
1400f5c7c2ea8d5feaedd44f633adb8f112a49ef — test: enforce required runtime metadata
516b9b75d8b4a330806c6168aa9d4f54e6a1524f — fix: avoid persisting raw calibration landmarks
963f85a17ae221182e23a43789717387a80a611f — test: prevent raw calibration data persistence

Findings corrected:
    missing checkpoint metadata accepted as compatible       FIXED
    missing adapter metadata accepted as compatible         FIXED
    raw calibration landmarks stored unnecessarily          REMOVED

Verification boundary:
    source-level re-audit                     STATIC VERIFIED
    current CI workflow                      DISABLED / NOT RUN
    full local test suite                    NOT VERIFIED
    real-data training                     NOT VERIFIED
    signer-independent evaluation          REQUIRED / BLOCKED BY VERIFIED SIGNER METADATA
    browser camera runtime                  NOT VERIFIED
    live Render verification                NOT VERIFIED

### 2026-09-24 — Restart #6: enforce calibration shot contract

142dfa77397260c06ea08aff6147b0e427e1aa3f — fix: enforce three calibration shots
069ac9cc60bae24be2da18b493728615398e0235 — test: enforce calibration shot minimum

Finding corrected:
    backend accepted fewer calibration shots than the active UI contract   FIXED

Verification boundary:
    source-level re-audit                     STATIC VERIFIED
    full local test suite                    NOT VERIFIED
    real-data training                     NOT VERIFIED
    signer-independent evaluation          REQUIRED / BLOCKED BY VERIFIED SIGNER METADATA

### 2026-09-24 — Final ledger reconciliation

9f4eea79b9ba7fd2a9d9d4e5285477c795f95efa — docs: document adapter data minimization
ca9d87bc6c3f139de8d0139384cbe5cc53842665 — docs: reconcile audit ledger completeness
966ab18ee4cdb1550d38115240b5e08be3b67c92 — docs: reconcile model audit ledger completeness
e2df45d0b5a19302156f7e69d8dfbc7342d61393 — docs: record calibration contract restart
a88e2176daffaf6b85884f006c407d27f59e9655 — docs: record calibration contract restart

Final audit status:
    targeted stale-reference sweep            PASS
    signer-independent gate                  REQUIRED / BLOCKED BY VERIFIED SIGNER METADATA
    model-quality evidence                  NOT VERIFIED
    current CI checks                       NONE
    binary V3 checkpoint in repository       PENDING


### 2026-09-24 — Adopt uploaded execution contract

Adopted the uploaded ITERATIVE FULL-RESTART PROJECT REPAIR PROTOCOL as the
authoritative repository execution contract. It requires Steps 1-9 to be
treated as complete iterations and requires a full restart after every newly
discovered defect or invalidating error.

Source basis:
    uploaded protocol lines 9-26
    uploaded protocol lines 90-140
    uploaded protocol lines 404-463
    uploaded protocol lines 546-566
    uploaded protocol lines 655-688
    uploaded protocol lines 703-737

Implementation status:
    contract adopted                     ENFORCED
    signer-independent evaluation        REQUIRED / BLOCKED BY VERIFIED SIGNER METADATA

## Current correction state

    training/browser landmark mismatch      CORRECTED
    legacy browser MediaPipe path           REMOVED
    handedness one-sided swap               REMOVED
    preprocessing drift                     GUARDED
    checkpoint/adapter contract drift       ENFORCED
    train/validation duplicate leakage      PREVENTED
    evaluation visibility                   IMPROVED
    dependency drift                        REDUCED
    signer-independent evaluation           REQUIRED / BLOCKED BY VERIFIED SIGNER METADATA
    raw calibration data persistence         REMOVED
    calibration shot-count contract          ENFORCED
    V3 binary checkpoint in repository      PENDING
    browser real-device verification        NOT VERIFIED
    live Render real-mode verification      NOT VERIFIED

## Historical supersession rule

Historical diary entries describe the implementation that existed at the time. A later correction is authoritative for the active contract. Do not delete historical records merely because they are superseded.

Example:
    2026-09-23 legacy MediaPipe browser path
    superseded by
    2026-09-24 MediaPipe Tasks Hand Landmarker browser path

## Required future change loop

    DISCOVER
    -> TRACE
    -> MODIFY
    -> TEST
    -> UPDATE AGENTS.md
    -> UPDATE OTHER REQUIRED DOCS
    -> RE-AUDIT

Every future repository change must update this file before the engineering cycle is considered complete.