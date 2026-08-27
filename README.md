# VisionBridge

**Hand-aware, few-shot signer-adaptive continuous Indian Sign Language recognition workspace.**

VisionBridge is being rebuilt around one coherent stack:

```text
Browser React/Vite app
  -> MediaPipe Holistic
  -> pose + face + left-hand + right-hand landmarks
  -> FastAPI API
  -> hand-aware PyTorch temporal model
  -> CTC decoding
  -> optional BridgeAdapter personalization
```

## Stack

| Layer | Choice |
|---|---|
| Web app | React + Vite + TypeScript |
| Landmark extraction | MediaPipe Holistic |
| API | FastAPI |
| Persistence | SQLAlchemy / SQLite for demo; managed DB required for durable production |
| ML | PyTorch temporal Transformer + local temporal convolution |
| Training | Colab + Lightning notebooks |
| Deployment | Render |

## Model contract

| Stream | Features/frame |
|---|---:|
| Pose | 132 = 33 × (x,y,z,visibility) |
| Face | 1404 = 468 × (x,y,z) |
| Left hand | 63 = 21 × (x,y,z) |
| Right hand | 63 = 21 × (x,y,z) |
| CTC blank | 0 |
| Vocabulary | 49 in current tokenizer |
| Maximum sequence | 1024 frames |

Hands are first-class model inputs. Every training and inference sample is expected to contain synchronized pose, face, left-hand, and right-hand streams. Missing hands can be represented by zero-filled 63D frames during dataset compatibility handling, but the new production API requires both hand streams explicitly.

## Web experience

The browser application uses a monochrome, minimal design system and real data flows. The live translation surface displays the camera feed with a 21-point skeleton overlay for both hands and sends synchronized landmark windows to the FastAPI translation endpoint.

Routes:

```text
/dashboard
/translate
/calibration
/history
/evaluation
/settings
```

Authentication uses the existing FastAPI bearer-token flow. The web client reads `VITE_API_BASE_URL` at build time.

## Training

Canonical notebooks:

- `notebooks/train_base_model_colab.ipynb`
- `notebooks/train_base_model_lightning.ipynb`
- `notebooks/validate_base_model_colab.ipynb`

The Colab flow must be run on a fresh GPU runtime. It reconstructs real dataset features, validates all four modalities, runs a semantic CTC overfit gate, performs training only after that gate passes, and must not push a checkpoint that is blank/space/trivial output.

## Important model-history status

A previous checkpoint was observed to produce a literal space token on 100% of frames for both train and validation examples. The earlier acceptance test incorrectly treated any non-blank output as success. The acceptance gate was hardened to reject empty, whitespace-only, space-collapsed, low-diversity, and high-CER predictions.

A fresh hand-aware training run is therefore required before the new checkpoint can be considered usable. The old pose+face checkpoint is not a valid substitute for the new hand-aware architecture.

## Development

### Backend

```bash
cd backend
python -m venv .venv
# activate using your platform's standard command
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Production build:

```bash
npm run build
```

Set:

```text
VITE_API_BASE_URL=https://<your-backend>/api/v1
```

## Deployment

`render.yaml` builds the React app into `frontend/dist` and rewrites SPA routes to `/index.html`. The backend readiness endpoint is `/api/v1/ready`.

SQLite on Render free service remains disposable local state. Use an external managed database before claiming durable production persistence.

## Verification policy

A feature is not considered complete until the relevant UI/API/model path is integrated and runtime-tested. Model quality is explicitly **not verified** until a fresh real-data training + semantic validation run passes. CI results are only reported when an actual workflow run is available.
