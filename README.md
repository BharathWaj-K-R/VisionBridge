# VisionBridge

**Few-shot signer-adaptive Indian Sign Language letter recognition.**

VisionBridge's active product is a two-stage fingerspelling recognizer:

~~~text
Browser camera
  -> MediaPipe Tasks Hand Landmarker 0.10.35 hand landmarks
  -> normalized 126D two-hand vector
  -> dynamic 26-class ISL letter base model
  -> 64D embedding for signer adaptation
  -> few-shot signer adapter
  -> one predicted letter + confidence
~~~

## Base model + few-shot adapter

### Base model

The base model is a configurable MLP trained on general ISL A-Z data and can be retrained or scaled as new data becomes available.

| Contract | Value |
|---|---:|
| Left hand | 63 XYZ values |
| Right hand | 63 XYZ values |
| Combined input | 126 |
| Default embedding | 64 |
| Default output classes | 26 (A-Z) |
| Loss | Cross-entropy |

The base model remains trainable, versioned, hot-reloadable, and replaceable after each validated training cycle.

### Dynamic scaling

The default model uses 126 input features, a 128-unit hidden layer, a 64D embedding, and 26 A-Z outputs. These are defaults, not immutable architecture limits.

Hidden width, embedding width, and dropout are checkpoint-defined within the active V3 input/output contract. The active V3 runtime remains 126D input with the A-Z vocabulary. A future input or vocabulary change requires a new model version, updated preprocessing/runtime contracts, and a fresh validation cycle. The backend hot-reloads compatible checkpoints, while existing adapters record the model version and checkpoint hash and require recalibration when the embedding space changes.

## Few-shot signer adapter

The signer adapter operates on the current model's configured embedding. The signer captures a few examples for each letter they want to recognize. The adapter stores a normalized prototype for each calibrated letter and predicts by cosine similarity.

The adapter records the exact base-model version and checkpoint SHA-256. Replacing the base checkpoint requires recalibration rather than silently mixing incompatible representations.
Raw calibration landmarks are not stored in the active adapter payload; the adapter stores the derived letter prototypes and compatibility metadata instead.

## Do I need to train anything?

**Yes, an initial base-model training job is required. Future model upgrades use the same training pipeline; no separate offline adapter-training job is required.**

You do **not** need to train the few-shot adapter offline.

The lifecycle is:

~~~text
1. Prepare ISL alphabet landmarks
2. Train or upgrade base model
3. Validate base model on its held-out test split
4. Install the small base checkpoint
5. New signer captures 3 examples/letter
6. Fit signer adapter at runtime
7. Recognize unseen examples with the current validated base + adapter
~~~

The previous continuous sentence-level architecture is no longer part of the critical path.

## Dataset

The default training source is the public RealSign Indian Sign Language alphabet dataset. Its repository documents 26 ISL alphabet classes and separate training, testing, and validation folders.

RealSign ISL alphabet dataset: https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset

The repository preparation script and browser runtime both use MediaPipe Tasks Hand Landmarker 0.10.35 with the same 21-point normalized landmark contract. The browser uses the version-pinned Tasks Vision package and the same hand-landmarker.task model bundle. The training notebook downloads the RealSign Git LFS archive through the media endpoint because the normal GitHub file endpoint returns the LFS pointer.

## Active API

~~~text
GET  /api/v1/letter/status
GET  /api/v1/letter/model
GET  /api/v1/letter/adapters/{id}
POST /api/v1/letter/calibrate
POST /api/v1/letter/predict
POST /api/v1/letter/event
~~~

The active endpoints validate the hand contract, require authentication, enforce adapter ownership, rate-limit the expensive operations, and bind adapters to the base-model checksum.

## Training notebook

Run:

~~~text
notebooks/train_letter_base_colab.ipynb
~~~

The notebook downloads the RealSign dataset, downloads the versioned MediaPipe Hand Landmarker task model, extracts the two-hand landmarks through the supported MediaPipe Tasks API, builds a stratified 80/20 train-validation split from the dataset's training + validation pools, and automatically iterates base-model training. After each epoch it measures accuracy for every A-Z class on the full validation split. Training can run for up to 500 epochs and stops when every letter reaches the configured target. The dataset's original testing split is kept untouched and measured separately. The best checkpoint is always saved.

Equivalent CLI training command:

~~~bash
PYTHONPATH=backend python -m app.training.letter_base \
  --data-dir /content/visionbridge_letter_data \
  --output backend/app/models/weights/letter_base_model.pt \
  --epochs 500 \
  --target-class-accuracy 1.0
~~~

After the training run, commit the generated small checkpoint:

~~~text
backend/app/models/weights/letter_base_model.pt
~~~

The file is intentionally allowed by .gitignore because the production API needs the trained base model locally.

### Migrating a legacy V3 checkpoint

Some V3 checkpoints created before the current metadata contract may contain valid
learned weights but lack the preprocessing/runtime metadata now required by the
strict loader. Migrate such a checkpoint without changing its learned tensors:

~~~bash
PYTHONPATH=backend python -m scripts.migrate_v3_checkpoint   --input /path/to/legacy_v3.pt   --output backend/app/models/weights/letter_base_model.pt
~~~

The migration utility accepts only the active V3 architecture and A-Z vocabulary,
adds the versioned preprocessing/runtime metadata, and refuses incompatible or
non-finite checkpoints. The migrated checkpoint must still pass the canonical
held-out test evaluation before it is accepted for deployment.

## Real-time architecture

Recognition is optimized so the per-frame hot path stays inside the browser:

~~~text
camera
  -> MediaPipe Tasks Hand Landmarker 0.10.35
  -> 126D normalization
  -> browser base-model inference
  -> browser few-shot adapter
  -> letter + confidence
~~~

The backend is used for authentication, one-time model/adapter loading, calibration persistence, and asynchronous recognition-event history. The real-time loop does not wait for FastAPI on every frame.

The camera also renders a live 21-point hand skeleton plus a short wrist-motion trail.

The browser tracker uses the version-pinned MediaPipe Tasks Hand Landmarker 0.10.35 configuration documented above.

This targets near-zero network latency, not literal zero milliseconds. The actual result depends on the user's camera, browser, hardware, and tracking workload.

## Run locally

### 1. Train the base model

Run `notebooks/train_letter_base_colab.ipynb`.

Install the resulting checkpoint at:

~~~text
backend/app/models/weights/letter_base_model.pt
~~~

### 2. Start the backend

~~~bash
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
~~~

### 3. Start the frontend

In a second terminal:

~~~bash
cd frontend
npm install
npm run dev
~~~

For the real model path, set:

~~~text
VITE_LOCAL_MODE=false
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
~~~

The browser fetches the current model once, fetches the selected signer adapter once, and performs frame-by-frame prediction locally. The backend prediction endpoint remains available for diagnostics and fallback.

## Frontend

The active UI is:

~~~text
/dashboard   status and recent letter events
/calibration capture few-shot signer examples
/translate   live letter recognition
/history     letter prediction history
/settings    signer adapter lifecycle
~~~

The old sentence translation screens are no longer the active product flow.

## Verification boundary

The repository's GitHub Actions regression workflow is currently disabled. Historical CI runs verified the application code, frontend build, and backend tests, but no current CI result should be treated as active verification. Training still reports held-out validation and test measurements directly from the notebook.

Two separate measurements matter:

1. Base-model accuracy on the held-out dataset test split.
2. Few-shot signer accuracy on held-out examples from a signer not used during adapter calibration.

No accuracy percentage is claimed here until those runs produce actual measurements.

Signer-independent evaluation is a required release gate, not a feature that can be removed because the current dataset metadata is insufficient. The current RealSign class folders do not expose verified signer IDs, so this gate remains blocked until an explicit signer-labeled manifest or equivalent verified metadata is available.

## Evaluation

Use the canonical evaluator after preparing the dataset and installing a checkpoint:

~~~bash
PYTHONPATH=backend python -m app.training.evaluate_letter_base \
  --checkpoint backend/app/models/weights/letter_base_model.pt \
  --data-dir /content/visionbridge_letter_data \
  --split test \
  --output-json /content/visionbridge_letter_evaluation.json
~~~

The evaluator reports overall accuracy, macro class accuracy, per-letter accuracy, the weakest class, the full A-Z confusion matrix, checkpoint size, parameter count, and model-only latency. Test results are for measurement only and must not be used repeatedly to tune training settings.
