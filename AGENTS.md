# VisionBridge Multi-Agent Engineering Contract

## State-update rule
Every meaningful code/notebook change, error, test, training result, validation result, root-cause discovery, workaround, decision, blocked state, and milestone must be recorded here. Never mark VERIFIED without runtime evidence.

## Mission
VisionBridge: continuous Indian Sign Language -> English translation.

```text
Real video -> MediaPipe Holistic -> pose(132) + face(1404) -> VisionBridgeBaseModel -> CTC decode -> English -> optional BridgeAdapter
```

## Current state
Branch: `main`.
GitHub connector read/write: AVAILABLE.
Direct shell clone/runtime: BLOCKED in this agent environment because `github.com` DNS cannot be resolved, so local pytest/build/browser execution after the latest changes is NOT AVAILABLE.
Historical backend result: `30 passed, 0 failures`, but it predates this audit.
Current checkpoint exists but is NOT VERIFIED as quality-ready.

## Model contract
- pose: 132 values/frame
- face: 1404 values/frame
- max sequence: 1024
- CTC blank: 0
- model: pose encoder + face encoder + cross-modal fusion + shared Transformer + char head

## Historical root cause: duplicate UIDs
Symptom: repeated filename stems such as `fever (2).MP4` occur under different sentence folders.
Root cause: old notebook used filename stem as global UID.
Impact: feature `.npy` files could overwrite and labels/features could be mismatched.
Fix: `train_base_model_colab.ipynb` rebuilds processed data and creates collision-safe IDs from sentence label + stem + relative-path hash.
Status: IMPLEMENTED, NOT VERIFIED by clean retraining.

## Historical model failure
Known real clips previously produced all-blank output even when used as training examples:
- `you are good` -> pose (69,132), face (69,1404), blank ratio 1.0.
- `i am suffering from fever` -> pose (72,132), face (72,1404), blank ratio 1.0.
Old checkpoint is untrusted.

## ML/data hardening
`backend/app/training/isltranslate.py` now rejects:
- duplicate UIDs
- wrong pose/face shapes
- misaligned or empty streams
- non-finite values
- unsupported target characters
- impossible CTC targets, including repeated-label minimum length
- invalid batch feature dimensions

Training uses deterministic split seed 42, AdamW, CTCLoss(blank=0, zero_infinity=True), gradient clipping, resumable checkpoints.
Status: FIXED, NOT RUNTIME-VERIFIED.

## Decoder hardening
Evaluator/inference load the saved vocabulary, enforce token 0 as blank, and fail closed on missing/malformed/mismatched checkpoint vocabulary.
Status: FIXED, NOT RUNTIME-VERIFIED.

## Extractor/converter hardening
`backend/scripts/extract_keypoints.py` now resolves video by stem with case-insensitive `.mp4/.avi/.mov/.mkv` extensions.
`backend/scripts/convert_isign_pose.py` now strictly requires pose 132 / face 1404, aligned non-empty frames, finite values, and fails incompatible 1434-face data.
Status: FIXED, NOT RUNTIME-VERIFIED.

## Security hardening
Added `backend/app/api/deps.py` bearer auth helpers.
- anonymous base-model translation remains allowed for public demo
- user-scoped translation requires matching authenticated user
- adapter use requires authenticated owner
- calibration requires authentication and matching user
- calibration duration must be >= `CALIBRATION_MIN_SECONDS`
- calibration target labels must satisfy true CTC alignment
Status: FIXED, NOT RUNTIME-VERIFIED.

## Frontend integrity hardening
Removed fake translation fallback and obsolete fake ticker.
Removed unverified benchmark/user-count claims.
Clarified that raw camera frames stay browser-side while extracted keypoints are sent to the backend.
Status: FIXED, NOT RUNTIME-VERIFIED.

## Production API boundary
`/api/v1/translate` accepts pre-extracted keypoints, not raw video.
Browser flow is `camera -> browser MediaPipe -> keypoints -> backend -> model`.
Raw-video upload API is not implemented.
Status: PARTIAL.

## Deployment
Render config uses Python 3.11.9, FastAPI/Uvicorn, SQLite, free tier, and explicitly notes ephemeral SQLite storage. Actual remote Render health/CORS/persistence was not live-tested.
Frontend non-local default API URL is `https://visionbridge-backend.onrender.com/api/v1`; remote availability was not live-tested.

## Regression tests added/updated
- `backend/tests/test_feature_contract_scripts.py`
- `backend/tests/test_ctc_contract.py`
- `backend/tests/test_translate_live_pipeline.py` (adapter auth boundary + decoder contract)
- `backend/tests/test_account_workflows.py`
- `.github/workflows/backend-tests.yml`
Runtime status: NOT VERIFIED because local execution is blocked.

## New feature completion pass
### Account workflows
Added authenticated account UX and protected private pages.
Implemented real dashboard, history, profile/adapters, and evaluation API routes backed by SQLite/SQLAlchemy.
Anonymous translation remains available; calibration/history/profile/evaluation require bearer authentication.
Status: IMPLEMENTED, NOT RUNTIME-VERIFIED.

### Dashboard
Replaced hardcoded usage/model/activity metrics with persisted translation logs, current model readiness, average confidence/latency, and current signer adapter metadata.
Status: IMPLEMENTED, NOT RUNTIME-VERIFIED.

### Translation history
Replaced static cards with authenticated server-side history querying, search, date-range selection, sorting, and CSV export.
Status: IMPLEMENTED, NOT RUNTIME-VERIFIED.

### Signer profiles
Replaced hardcoded people/accuracy cards with authenticated adapter listing and owner-scoped adapter deletion.
Status: IMPLEMENTED, NOT RUNTIME-VERIFIED.

### Evaluation
Replaced fabricated accuracy/BLEU/WER/memory numbers with an evidence-backed status page. Benchmark metrics are explicitly shown as unmeasured until a real benchmark run is persisted.
Status: IMPLEMENTED, NOT RUNTIME-VERIFIED.

### Calibration
Replaced the simulated calibration progress UI with a real browser MediaPipe capture flow. The browser samples synchronized 132/1404 keypoints at about 3 fps over the configured five-minute session and submits human-readable target text to the authenticated calibration API. Backend converts target text with the saved vocabulary and downsamples calibration inputs to `CALIBRATION_MAX_FRAMES` before fitting the adapter.
Status: IMPLEMENTED, NOT RUNTIME-VERIFIED.

### Authentication UI
Added `frontend/pages/auth.html` and `frontend/assets/js/auth.js` for real registration/login and bearer-token persistence. Private pages now require the token.
Status: IMPLEMENTED, NOT RUNTIME-VERIFIED.

## Performance hardening
Added `CALIBRATION_MAX_FRAMES` with default 256 and runtime validation against `MAX_INFERENCE_FRAMES` to prevent the five-minute calibration capture from forcing a 900-frame Transformer training pass on the CPU deployment target.
Status: IMPLEMENTED, NOT RUNTIME-VERIFIED.

## Documentation synchronization
README now reflects the implemented browser keypoint extraction/calibration/account workflows and explicitly states that model quality and benchmark metrics remain unverified until real-data runtime evidence exists.
Status: UPDATED.

## Acceptance gates
A: clean dataset integrity -> NOT VERIFIED.
B: real-data CTC overfit / non-blank + useful target overlap -> NOT VERIFIED.
C: full training -> BLOCKED by B.
D: multi-video real validation -> BLOCKED by C.
E: BridgeAdapter -> BLOCKED by D.
F: full application runtime/API/browser regression -> NOT VERIFIED.

## Issue diary
### 2026-08-25 duplicate UID corruption
FIXED-NOT-VERIFIED. Old filename-stem IDs were not globally unique. Collision-safe notebook IDs implemented.

### 2026-08-25 real-video blank collapse
FAILED evidence. Old checkpoint produced 100% blank on known real clips. Checkpoint rejected.

### 2026-08-25 deterministic split mismatch
FIXED-NOT-VERIFIED. Training and notebook acceptance now share seed 42.

### 2026-08-25 evaluator decoder vocabulary state
FIXED-NOT-VERIFIED. Saved vocab is loaded/validated before decode.

### 2026-08-25 132/1404 feature-contract enforcement
FIXED-NOT-VERIFIED. Dataset, extractor, converter, API guards hardened.

### 2026-08-25 uppercase `.MP4` extractor failure
FIXED-NOT-VERIFIED. Case-insensitive stem+extension resolution added.

### 2026-08-25 stale 1434-face converter
FIXED-NOT-VERIFIED. Strict 1404 converter contract added; incompatible arrays rejected.

### 2026-08-25 adapter authorization gap
FIXED-NOT-VERIFIED. Bearer auth added; adapter ownership and calibration user ownership enforced.

### 2026-08-25 CTC calibration contract gap
FIXED-NOT-VERIFIED. True CTC minimum length and 300-second minimum calibration duration enforced.

### 2026-08-25 silent target-character dropping
FIXED-NOT-VERIFIED. Tokenizer now fails instead of dropping unsupported characters.

### 2026-08-25 fake frontend translation and false metrics
FIXED-NOT-VERIFIED. Fake ticker removed and claims replaced with runtime/evidence language.

### 2026-08-25 static product workflows discovered
FIXED-NOT-VERIFIED. Dashboard, history, signer profiles, and evaluation pages were converted from hardcoded demo state to real authenticated APIs and persisted data flows.

### 2026-08-25 calibration mock discovered
FIXED-NOT-VERIFIED. Calibration page now captures real browser landmarks, submits a real target text, and persists an adapter through the existing backend service.

### 2026-08-25 CI gap discovered
IMPLEMENTED. Added GitHub Actions backend regression workflow. No workflow run is available yet for the latest commit, so CI is NOT VERIFIED.

## Current next steps
1. Fresh Colab runtime.
2. Run `notebooks/train_base_model_colab.ipynb` from cell 1.
3. Require `DATASET INTEGRITY: PASS`.
4. Require `OVERFIT SANITY: PASS`.
5. Only then full training.
6. Require train + held-out acceptance before checkpoint push.
7. Run validation notebook on multiple real videos.
8. Run `python -m pytest backend/tests` in Colab/CI.
9. Run the new GitHub Actions workflow and fix any regression failures.
10. Verify remote Render health/CORS/auth/calibration/browser flow.
11. Only then resume and evaluate BridgeAdapter on held-out real signers.

## Final verdict
**NOT READY.**
Reason: clean retraining not yet runtime-verified; current checkpoint quality not trusted; current full test suite not run after latest changes; browser/Render behavior not live-tested; raw-video production API boundary remains incomplete; the new CI workflow has not yet produced a successful run.

## Golden rule
Static correctness is not runtime proof. Only an actual test or real-data experiment earns VERIFIED.
