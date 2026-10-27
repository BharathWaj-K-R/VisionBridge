# VisionBridge

**VisionBridge is a browser-based communication portal with Indian Sign Language (ISL) A–Z fingerspelling recognition and optional signer-specific calibration.** It combines a live camera tracker, a compact browser inference path, calibration profiles, prediction history, speech input, and a communication word bank.

> **Scope matters:** the verified ML task is isolated A–Z letter recognition. The portal also contains a speech-to-sentence and sign-token resolution prototype, but it is not a validated, general-purpose sentence-level ISL translator. See [Current limitations](#current-limitations).

## What the portal currently includes

| Area | Current behavior |
|---|---|
| **Camera Translate** | Uses the browser camera and MediaPipe Tasks Hand Landmarker to detect up to two hands, draw live landmarks, and recognize A–Z letters. |
| **Cybernetic glove** | Shows a procedural Three.js visualization synchronized to detected hand landmarks. It is a visual tracker, not a second recognition model. |
| **Signer calibration** | Captures examples for selected letters and fits a prototype adapter in the base model's 64D embedding space. Each calibrated letter needs at least three examples. |
| **Current message** | Shows committed letters and provides Add space, Backspace, and Clear controls. The current implementation is a short rolling buffer, not a long-form text editor. |
| **History** | Lists saved recognition events and supports CSV export. |
| **Speak** | Uses browser speech-recognition capabilities to capture text, group it into sentences, and prepare a sequence of sign-reference tokens. Browser support and microphone permission are required. |
| **Word Bank** | Provides built-in communication vocabulary, custom phrases, and assignment to Quick Access. |
| **Quick Access and profiles** | Supports ten Quick Access slots, profile-specific communication preferences, signing speed, voice preferences, and avatar settings. |
| **Settings and accounts** | Provides registration/sign-in, theme preferences, account controls, adapter management, and browser permission guidance. |

## Application routes

| Route | Purpose |
|---|---|
| ``/``, ``/login`` | Sign in or create an account |
| ``/dashboard`` | Home and recent recognition activity |
| ``/translate`` | Live camera recognition and letter composition |
| ``/calibration`` | Capture examples and fit a signer adapter |
| ``/voice-to-sign`` | Speech input, sentence queue, and sign-token resolution prototype |
| ``/word-bank`` | Vocabulary, custom phrases, and Quick Access assignment |
| ``/personalization`` | Profiles, avatar preferences, voice/speed settings, and saved phrases |
| ``/history`` | Review, clear, and export recognition history |
| ``/settings`` | Appearance, account, permissions, and adapter lifecycle |

## Architecture

### Camera recognition path

```text
Browser camera
  -> MediaPipe Tasks Hand Landmarker 0.10.35
  -> 21 landmarks per detected hand (x, y, z)
  -> left-hand slot (63 values) + right-hand slot (63 values)
  -> independent wrist-centering and scale normalization
  -> 126D input vector
  -> V3 base classifier (26 A-Z logits and 64D embedding)
  -> optional signer adapter (normalized prototypes + cosine similarity)
  -> letter + confidence
  -> stability/confidence gate
  -> on-screen message buffer and optional history event
```

The per-frame model path runs in the browser in real mode. The backend supplies the model and adapter payloads, handles calibration persistence and authentication, and records history events. Normal live inference does not need a FastAPI request for every camera frame.

The hand-tracking overlay and cybernetic glove use live MediaPipe landmarks. Neither changes the model's prediction. If a hand is absent, its feature slot is represented by zeros; the normalizer rejects an input when both slots are empty.

### V3 model contract

| Property | Current contract |
|---|---:|
| Left-hand landmarks | 21 × XYZ = 63 values |
| Right-hand landmarks | 21 × XYZ = 63 values |
| Combined input | 126 values |
| Hidden layer | 128 units by default |
| Embedding | 64 dimensions by default |
| Output vocabulary | 26 classes, A–Z |
| Default loss | Cross-entropy |
| Preprocessing version | ``two-hand-wrist-scale-v1`` |
| Landmark runtime | ``mediapipe-hand-landmarker-0.10.35`` |
| Model version | ``visionbridge-letter-base-v3`` |

The base model is a small, configurable MLP: LayerNorm, Linear, GELU, Dropout, Linear, LayerNorm, GELU, followed by a 26-class output head. Its checkpoint records model, preprocessing, runtime, dimension, vocabulary, and dropout metadata. Incompatible checkpoints and adapters are rejected. Adapters are tied to the base checkpoint SHA-256 and need recalibration if the base model changes.

### Speech and sign-token path

The Speak page uses the browser Web Speech API for speech recognition, sentence buffering/reconciliation, and playback controls. Sentence text is resolved against the vocabulary and the sign-asset registry. The current registry contains A–Z letter-atlas entries, while the word and phrase asset maps are empty. Consequently, this path is a prototype for speech-to-sentence handling and sign-token resolution, **not** proof of accurate semantic ISL sentence translation. See [Current limitations](#current-limitations).

## Current recorded model evaluation

The repository contains a recorded test evaluation in [``visionbridge_letter_evaluation.json``](visionbridge_letter_evaluation.json), bound to checkpoint SHA-256:

``f903872e96f269c5fbee6ea897f0027b91e162f2fc45b4c9a115e8ba8e38c917``

| Metric | Recorded result |
|---|---:|
| Prepared test samples evaluated | 4,633 |
| Overall accuracy | 98.2733% |
| Macro class accuracy | 98.2913% |
| Weakest class | W |
| W accuracy | 78.7710% |
| Parameters | 26,582 |
| Checkpoint size | 111,429 bytes |

These are **stored dataset-evaluation results**, not a new run performed by reading this README, and not a live-webcam accuracy guarantee. Per-letter sample counts range from 137 to 200 in the stored report because landmark extraction can skip unusable images. The headline accuracy therefore describes the samples that reached the prepared test set, not every source image. The report includes per-letter counts and the full confusion matrix.

The most important recorded confusion is W, often predicted as M, N, or O. Do not hide that class weakness behind the overall score.

## Dataset and reproducible preparation

The training source is the public [RealSign Indian Sign Language alphabet dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset). The dataset archive is downloaded during training; it is **not stored in this repository**, and Git LFS is not required for the dataset.

The training notebook pins a RealSign source commit and verifies the archive's expected size and SHA-256 before extracting it. It also downloads and verifies the MediaPipe Hand Landmarker ``.task`` asset. Dataset preparation is implemented in ``backend/scripts/prepare_letter_dataset.py`` and produces:

```text
visionbridge_letter_data/
├── train.npz
├── val.npz
├── test.npz
├── labels.json
├── train_manifest.jsonl
├── val_manifest.jsonl
├── test_manifest.jsonl
└── duplicate_report.json
```

The pipeline:
- Extracts two-hand landmarks with MediaPipe Tasks Hand Landmarker 0.10.35.
- Keeps the source testing split separate from training and validation.
- Creates a stratified 80/20 train/validation split from the source training and validation pools.
- Records source path, source split, class label, and exact image SHA-256 in manifests.
- Groups exact duplicates during the generated train/validation split and checks exact-hash overlap against the test split.
- Saves class counts and records images skipped when valid landmarks cannot be extracted.

Exact-hash checks do not prove signer independence or detect every near-duplicate frame. The current source folder structure does not expose verified signer IDs, so signer-independent evaluation remains **blocked**.

## Run the application locally

### Prerequisites

- Python 3.11 for the backend.
- Node.js 20.x and npm for the frontend.
- A modern browser with camera access. Camera access on deployed websites requires HTTPS; localhost is suitable for local development.
- Internet access to load the pinned MediaPipe Tasks Vision runtime, hand-landmarker asset, and (in real mode) the model payload from the backend.

### 1. Start the backend

From the repository root:

```bash
cd backend
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the backend dependencies and start FastAPI:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The current checkpoint is committed at ``backend/app/models/weights/letter_base_model.pt``, so you do not need to train the model just to run the application. The backend defaults to a local SQLite database for development. Interactive API documentation is available at ``http://127.0.0.1:8000/docs``; health and readiness endpoints are under ``/api/v1/health`` and ``/api/v1/ready``.

### 2. Start the frontend

In a second terminal:

```bash
cd frontend
npm ci
```

The checked-in ``.env.example`` documents the frontend variables. To use the full local backend, create ``frontend/.env`` (this file is ignored by Git) with:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1
VITE_LOCAL_MODE=false
```

Then start Vite:

```bash
npm run dev
```

Open the URL Vite prints, normally ``http://localhost:5173``, and sign in or register. Grant camera permission for Translate and microphone permission for Speak when the browser asks.

### Local demo mode versus real mode

- **Local demo mode:** ``VITE_LOCAL_MODE=true``. The frontend's demo data, account-like state, adapter prototypes, history, word bank, profiles, and communication settings are stored in browser ``localStorage``. This is convenient for UI exploration but is not server-backed persistence or an end-to-end production test.
- **Real mode:** ``VITE_LOCAL_MODE=false``. The frontend uses the configured FastAPI API and the committed base-model checkpoint. Calibration, authentication, adapters, and history use the backend/database path. Set ``VITE_API_BASE_URL`` to the matching API base URL.

Do not deploy with local demo mode when you expect server authentication and persistent account data.

## Training and evaluation

Training is optional when using the committed model. Retrain only when preparing a candidate checkpoint for review.

### Recommended reproducible training workflow

Run the notebook in Colab:

- [Train and compare accuracy candidates](https://colab.research.google.com/github/BharathWaj-K-R/VisionBridge/blob/main/notebooks/train_letter_base_colab.ipynb)
- [Evaluate the currently committed V3 checkpoint](https://colab.research.google.com/github/BharathWaj-K-R/VisionBridge/blob/main/notebooks/evaluate_letter_base_v3_colab.ipynb)

The training notebook creates an isolated Python 3.12 environment, verifies the pinned dataset and MediaPipe assets, prepares landmarks, trains controlled candidates, selects by validation metrics, and evaluates the selected candidate on the untouched test split. A GPU is recommended for training; dataset download and landmark extraction can still take time.

The GitHub Actions workflow **VisionBridge Accuracy Optimization** is manually triggered. It runs the training notebook, uploads reports, and conditionally commits a candidate checkpoint and evidence to ``main`` only when the configured selection, held-out test, browser/PyTorch parity, and backend test gates pass. Normal commits do not automatically retrain the model.

The underlying training module is ``backend/app/training/letter_base.py``. It supports configurable hidden/embedding widths, dropout, cross-entropy or focal loss, class-focused loss weights, and mild landmark jitter/rotation. Model selection should use validation data; do not repeatedly use the test set to tune hyperparameters.

### Evaluate a prepared checkpoint

Install training dependencies only when preparing data or running training locally:

```bash
python -m pip install -r backend/requirements-training.txt
```

From the repository root, run the canonical evaluator against a prepared dataset:

```bash
PYTHONPATH=backend python -m app.training.evaluate_letter_base --checkpoint backend/app/models/weights/letter_base_model.pt --data-dir /content/visionbridge_letter_data --split test --output-json /content/visionbridge_letter_evaluation.json
```

The evaluator reports overall and macro accuracy, per-letter accuracy, class counts, the confusion matrix, checkpoint hash/size, parameter count, and model-only latency. Test evaluation is for final measurement; use ``--split val`` when comparing candidate checkpoints.

### Checkpoint compatibility

A future change to input features, preprocessing, landmark runtime, or label vocabulary needs a new compatible model/runtime contract and fresh evaluation. The migration utility ``backend/scripts/migrate_v3_checkpoint.py`` is only for eligible legacy V3 checkpoints missing required metadata. It does not retrain or improve the learned weights, and migrated weights must still pass the evaluator and release checks.

## Backend API overview

The API version prefix is ``/api/v1``.

| Area | Routes |
|---|---|
| Health | ``GET /health``, ``GET /ready`` |
| Authentication | ``POST /auth/register``, ``POST /auth/login``, ``GET /auth/csrf``, ``POST /auth/logout`` |
| Current user and adapters | ``GET /users/me``, ``GET /users/me/adapters``, ``DELETE /users/me/adapters/{id}`` |
| Letter model | ``GET /letter/status``, ``GET /letter/model``, ``GET /letter/adapters/{id}`` |
| Calibration and recognition | ``POST /letter/calibrate``, ``POST /letter/predict``, ``POST /letter/event`` |
| Dashboard and history | ``GET /dashboard``, ``GET /history``, ``DELETE /history``, ``GET /history/export.csv`` |
| Words and Quick Access | ``GET/POST /communication/words``, ``PUT/DELETE /communication/words/{id}``, ``GET/PUT /communication/quick-access`` |
| Profiles and usage | ``GET/POST /communication/profiles``, ``PUT/DELETE /communication/profiles/{id}``, ``POST /communication/usage``, ``GET /communication/most-used`` |

Protected routes require authentication. The API validates the active model/adapter contract, scopes user-owned records, and applies request limits to authentication, calibration, and recognition endpoints.

## Configuration and deployment

- ``render.yaml`` defines the Render Python API and static frontend services.
- The frontend production configuration must use ``VITE_LOCAL_MODE=false`` and a valid ``VITE_API_BASE_URL``.
- Production backend configuration requires a strong non-default ``SECRET_KEY``, a durable PostgreSQL ``DATABASE_URL`` (SQLite is rejected in production), correct ``ALLOWED_ORIGINS``, and valid secure-cookie settings.
- The backend uses FastAPI JWT/HttpOnly-cookie authentication, a CSRF token/header for browser state-changing requests, password validation/hashing, security response headers, and in-process rate limiters.
- Supabase SQL migrations define the custom ``visionbridge_app`` database role and owner-scoped row-level security for application tables. Review/apply the required migrations to the target database before relying on those controls. The release records include runtime RLS verification; a dedicated pgTAP suite is still listed as maintenance work.

Relevant configuration examples are in ``backend/.env.example`` and ``frontend/.env.example``. Treat them as examples, not production secrets. The backend reads environment variables from its process environment; it does not load a ``.env`` file automatically.

## Repository map

```text
.
├── frontend/
│   ├── src/App.tsx                         # Routes, auth, application shell
│   ├── src/pages/                          # Dashboard, Translate, calibration, Speak, Words, history, settings, profiles
│   ├── src/landmarks.ts                    # MediaPipe setup, normalization, tracking overlay
│   ├── src/useLandmarkSession.ts            # Camera/landmark lifecycle
│   ├── src/browserModel.ts                 # Browser-side V3 model and adapter inference
│   ├── src/components/CyberneticHandGlove.tsx # Live procedural 3D hand display
│   ├── src/api.ts                          # API client and local demo mode
│   ├── src/signing/                        # Sign asset registry and token resolution
│   └── public/signs/sign-atlas.webp        # A–Z reference atlas
├── backend/
│   ├── app/api/                            # Auth, letters, history, dashboard, communication, users, health
│   ├── app/core/                           # Configuration, security, CSRF, rate limits
│   ├── app/db/                             # SQLAlchemy models, sessions, PostgreSQL RLS context
│   ├── app/models/letter_model.py          # V3 architecture and checkpoint contract
│   ├── app/services/letter_fewshot.py       # Normalization and few-shot signer adapter
│   ├── app/training/                       # Training and evaluation
│   ├── scripts/                            # Dataset preparation, checkpoint migration, parity verification
│   └── tests/                              # Backend, release, adapter, and training tests
├── notebooks/                              # Colab training and evaluation notebooks
├── data/README.md                          # Dataset source and prepared-data contract
├── docs/release/                           # Release gates, accuracy experiments, parity and RLS evidence
├── supabase/migrations/                    # PostgreSQL schema and RLS migrations
├── visionbridge_letter_evaluation.json     # Stored test metrics and confusion matrix
└── visionbridge_v3_release_evidence.json   # Checkpoint-bound release evidence
```

## Tests and CI

The [Product Validation workflow](.github/workflows/product-validation.yml) runs on pushes and pull requests and includes:

1. Backend regression tests: ``pytest -q`` in ``backend/``.
2. Browser/PyTorch numerical parity: 32 generated input vectors, with a default maximum absolute error tolerance of ``1e-5``.
3. Frontend TypeScript and Vite production build with ``VITE_LOCAL_MODE=false``.
4. Production-artifact verification using ``frontend/scripts/verify-production-build.mjs``.

Run the same checks locally:

```bash
# Backend tests
cd backend
python -m pip install -r requirements.txt
pytest -q

# Frontend build (in a second terminal, from the repository root)
cd frontend
npm ci
npm run build
node scripts/verify-production-build.mjs
```

The automated checks validate code contracts and builds; they do not replace testing camera/microphone permissions, live device behavior, signer-held-out accuracy, or a complete production deployment on a real browser/device.

> **Release gate: BLOCKED.** The committed checkpoint has recorded same-source held-out test metrics, but signer-independent evaluation remains a required gate and is blocked because the active dataset metadata does not expose verified signer IDs. A passing model test or CI run does not satisfy that gate.

## Current limitations

1. **A–Z isolated-letter scope.** The base model predicts a single alphabet letter from a frame. It does not perform continuous ISL grammar or sentence-level sign-language translation.
2. **Voice-to-sign remains a prototype.** Browser speech recognition and sentence/token processing exist, but ``WORD_SIGN_ASSETS`` and ``PHRASE_SIGN_ASSETS`` are empty. Do not advertise the current flow as a validated semantic sign translator.
3. **Short rolling camera buffer.** On committing a new letter, the current message buffer retains at most the latest six tokens. It needs a buffer behavior change before it can act as a long-form sentence composer.
4. **Dataset coverage.** The recorded prepared test set has 4,633 samples with uneven per-letter support. Images that fail landmark preparation are skipped, so end-to-end failure rates across every source image are not represented by classification accuracy alone.
5. **Signer generalization.** Verified signer IDs are absent from the active dataset metadata; signer-independent evaluation is blocked.
6. **W-class weakness.** W is the weakest class in the current recorded test report at 78.77%; it is commonly confused with M, N, and O.
7. **Live-device verification.** Automated CI does not exercise a real camera/microphone or prove accuracy in varying lighting, angles, users, and hardware.
8. **Sign asset coverage.** The repository includes a static A–Z atlas. Word/phrase assets need to be added and validated before claiming broad visual ISL phrase coverage.
9. **Database migration tests.** Runtime RLS evidence is recorded, but a dedicated pgTAP suite is not yet part of CI.

Keep those limitations visible when publishing the project or reporting accuracy.