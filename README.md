# VisionBridge

**Few-Shot Signer-Adaptive Continuous Indian Sign Language Translation**

Real-time ISL-to-text translation using a pose + facial-expression fusion transformer as a frozen base model, plus a lightweight few-shot adapter (**BridgeAdapter**) intended to personalize to a new signer's style from calibration data.

## Repo layout

```text
visionbridge/
├── backend/          FastAPI + SQLite backend, model code, adapters
│   ├── app/
│   │   ├── api/       route handlers (auth, translate, calibration, dashboard, history, users, evaluation, health)
│   │   ├── core/      config + security
│   │   ├── db/        SQLAlchemy session + ORM models
│   │   ├── models/    frozen base model + BridgeAdapter
│   │   ├── schemas/   Pydantic request/response contracts
│   │   └── services/  inference + calibration orchestration
│   ├── tests/
│   └── requirements.txt
├── frontend/          Static HTML/CSS/JS UI with live API integration
├── data/              dataset preparation area
├── notebooks/         Colab/Lightning training and validation notebooks
├── .github/workflows/ backend + frontend regression CI
└── render.yaml        Render deployment config for backend + frontend
```

## Model contract

| Contract | Current value |
|---|---:|
| Pose features / frame | 132 |
| Face features / frame | 1404 |
| Maximum inference sequence | 1024 frames |
| CTC blank token | 0 |
| Calibration minimum | 300 seconds |
| Calibration fitting cap | 256 frames |

These are implementation contracts, not benchmark claims.

## Datasets

The training pipeline is wired for real ISL keypoint datasets. See the Colab notebook and `backend/app/training/isltranslate.py` for the expected processed layout.

## Training notebooks

- `notebooks/train_base_model_colab.ipynb` — real-video extraction, clean UID rebuilding, CTC sanity gate, deterministic training, and acceptance checks.
- `notebooks/train_base_model_lightning.ipynb` — persistent training workflow.
- `notebooks/validate_base_model_colab.ipynb` — real-video checkpoint validation.

## Local development

### Backend

```bash
cd backend
python -m venv .venv
# activate the environment using your platform's standard command
pip install -r requirements.txt
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

### Frontend

Serve `frontend/` with any static HTTP server, for example:

```bash
npx serve frontend
```

The browser performs MediaPipe Holistic keypoint extraction for the live translation and calibration flows, then sends pose/face keypoints to the configured backend API.

## Authentication and account workflows

- `/api/v1/auth/register` creates an account.
- `/api/v1/auth/login` returns a bearer token.
- Dashboard, calibration, history, signer-adapter management, and evaluation require authentication.
- Anonymous base-model translation remains supported by `/api/v1/translate` when a compatible base checkpoint is installed.

## Health and readiness

- `GET /api/v1/health` is a lightweight process-liveness endpoint and does not load the model.
- `GET /api/v1/ready` validates the checkpoint/vocabulary contract and returns HTTP 503 until the model is ready.
- Render uses `/api/v1/ready` as its health check so an alive-but-unready service is not reported as healthy.

## Deployment (Render)

`render.yaml` defines:
- `visionbridge-backend` — FastAPI service.
- `visionbridge-frontend` — static site.

The current Render configuration uses SQLite on service-local storage. That is suitable only for a disposable demo because the storage is ephemeral on the free service plan. Use persistent storage or an external database before treating the deployment as durable production infrastructure.

Update `ALLOWED_ORIGINS` and the frontend API endpoint to the actual deployed service URLs.

## Verification status

The repository contains automated backend tests and GitHub Actions checks for Python compilation, backend tests, and frontend JavaScript syntax. The current model checkpoint and clean-data retraining are **not considered quality-verified until a real Colab/CI run demonstrates non-blank predictions on real ISL data**.

Adapter deletion removes both the database record and its on-disk weights, and calibration now fails closed if the adapter would exceed the documented parameter budget.

The evaluation UI intentionally reports benchmark metrics as **not measured** unless evidence is persisted. It does not display fabricated accuracy, BLEU, WER, or memory numbers.
