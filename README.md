# VisionBridge

**Few-shot signer-adaptive Indian Sign Language letter recognition.**

VisionBridge has been deliberately downscoped from continuous sentence translation to a small, demonstrable fingerspelling product:

```text
Browser camera
  -> MediaPipe hand landmarks
  -> 21-point left + 21-point right hand vectors
  -> wrist/scale normalization
  -> signer-specific few-shot prototypes
  -> cosine similarity
  -> one predicted letter + confidence
```

## Why the scope changed

The previous project used a four-stream temporal CTC model for sentence-level recognition. That path required a fresh hand-aware training run and real-data semantic validation. The current product does not depend on that checkpoint.

The active letter adapter is a **prototype-based few-shot signer adapter**. The signer supplies a few live examples for each letter they want to recognize. VisionBridge stores one normalized prototype per calibrated letter and compares new hand shapes against those prototypes.

This keeps the core research/demo idea — signer adaptation from very little data — while removing sentence decoding, pose, face, CTC, and large model training from the critical demo path.

## Feature contract

| Input | Size |
|---|---:|
| Left hand | 21 × 3 = 63 |
| Right hand | 21 × 3 = 63 |
| Combined | 126 floats |
| Output | One letter + confidence |

Each hand is translated so the wrist is the origin and scaled by the maximum wrist-relative landmark distance. Missing hands are represented as zeros.

## Product flow

1. Sign in.
2. Open **Calibrate**.
3. Start the camera and capture three examples for each letter you want to recognize.
4. Calibrate at least two letters.
5. Fit the signer adapter.
6. Open **Recognize** and test unseen hand shapes live.

The frontend runs the same prototype logic locally when `VITE_LOCAL_MODE=true`, so the demo does not require a trained checkpoint or external database.

## Active API

```text
POST /api/v1/letter/calibrate
POST /api/v1/letter/predict
```

Both endpoints require authentication, validate the 126-value hand contract, preserve signer ownership checks, and use the existing rate-limiting pattern.

## Legacy code

The original sentence-level training, CTC, and multimodal translation modules remain in the repository as legacy/regression material. They are not part of the active letter-recognition UI.

## Development

### Backend

```bash
cd backend
python -m venv .venv
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests -q
```

### Frontend

```bash
cd frontend
npm install
npm run check
npm run build
```

## Verification boundary

The new letter path is covered by the repository regression suite and frontend CI build checks. Recognition accuracy is **not** claimed here because no held-out signer benchmark was run as part of this time-constrained downscope.

The production Render configuration intentionally keeps `VITE_LOCAL_MODE=true` until the project has a validated trained model, durable persistence, and production-grade authentication.
