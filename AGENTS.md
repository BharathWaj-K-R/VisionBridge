# VisionBridge Engineering Contract

This file is the persistent engineering hand-off for agents working on VisionBridge. Read it before making changes.

The active product is an Indian Sign Language alphabet/fingerspelling letter recognizer built around:

~~~text
Browser camera
 -> MediaPipe Hands
 -> normalized 126D two-hand landmark vector
 -> dynamic 26-class VisionBridge letter base model
 -> 64D signer-independent embedding
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
Landmarks: MediaPipe Hands
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
- handedness handling must remain consistent between dataset preparation and runtime.

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
real backend translation verified
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
Base model implementation: CI VERIFIED
Few-shot adapter implementation: CI VERIFIED
Letter API implementation: CI VERIFIED
Frontend TypeScript check: CI VERIFIED
Frontend production build: CI VERIFIED
Production artifact verification: CI VERIFIED
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
GitHub Actions run #197
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
A  Train base model on real ISL A-Z data            NOT VERIFIED
B  Record held-out base test performance            BLOCKED until A
C  Validate few-shot adaptation on held-out signer  BLOCKED until A/B
D  Verify browser camera + real inference           NOT VERIFIED
E  Verify live Render real-mode flow               BLOCKED until A-D
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
BASE MODEL CODE: CI VERIFIED
FEW-SHOT ADAPTER: CI VERIFIED
API: CI VERIFIED
FRONTEND: CI VERIFIED
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
 -> MediaPipe Hands only
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
MediaPipe Hands modelComplexity=0
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
