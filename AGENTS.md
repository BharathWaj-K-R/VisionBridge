# VisionBridge Multi-Agent Engineering Contract

## State-update rule
Every meaningful code/notebook change, error, test, training result, validation result, root-cause discovery, workaround, decision, blocked state, and milestone must be recorded here. Never mark VERIFIED without runtime evidence.

## Mission
Continuous Indian Sign Language -> English translation.

```text
Real video -> MediaPipe Holistic -> pose(132) + face(1404) -> VisionBridgeBaseModel -> CTC decode -> English -> optional BridgeAdapter
```

## Current repository state
- Branch: `main`.
- GitHub read/write: available.
- Direct shell clone/runtime in this agent: blocked because `github.com` DNS cannot be resolved. Latest backend tests, browser E2E, and live Render checks are therefore NOT VERIFIED here.
- Historical backend result: `30 passed, 0 failures`, but that predates the latest audit passes.
- `base_model.pt` and its vocabulary are present in the repository, but the checkpoint is NOT quality-verified.

## Core model contract
- Pose: 132 values/frame.
- Face: 1404 values/frame.
- Maximum inference sequence: 1024 frames.
- CTC blank: 0.
- Backbone: pose encoder + face encoder + cross-modal fusion + shared Transformer + character head.

## Historical ML failure diary
### Duplicate UID corruption
Problem: sentence-level dataset paths can contain repeated filenames such as `fever (2).MP4`.
Root cause: old training notebook used filename stem as the global UID.
Impact: `.npy` feature files could overwrite each other or become mismatched with text labels.
Fix: Colab training rebuilds processed data and generates collision-safe IDs from sentence label + filename + relative-path hash.
Status: FIXED IN CODE, CLEAN RETRAIN NOT VERIFIED.

### Real-video blank-collapse checkpoint
Evidence from prior runtime:
- `you are good`: pose `(69,132)`, face `(69,1404)`, predicted `(no sign detected)`, blank ratio `1.0`.
- `i am suffering from fever`: pose `(72,132)`, face `(72,1404)`, predicted `(no sign detected)`, blank ratio `1.0`.
The previous checkpoint is rejected as quality evidence.

### Deterministic split mismatch
Root cause: training split and notebook acceptance could use different split orderings.
Fix: deterministic seed 42 is shared between training and acceptance.
Status: FIXED IN CODE, NOT RUNTIME-VERIFIED.

## ML/data hardening diary
`backend/app/training/isltranslate.py` rejects duplicate IDs, wrong pose/face shape, frame mismatch, empty/non-finite streams, unsupported target characters, impossible CTC targets, and invalid batch feature dimensions.

CTC contract now accounts for adjacent repeated labels: minimum required input length is `target_length + repeated_adjacent_labels`.

Tokenizer no longer silently drops unsupported target characters.

`backend/scripts/extract_keypoints.py` resolves video files case-insensitively across `.mp4/.MP4/.avi/.AVI/.mov/.MOV/.mkv/.MKV` style extensions.

`backend/scripts/convert_isign_pose.py` strictly enforces pose 132 / face 1404, aligned non-empty finite arrays, and rejects stale 1434-face data.

Evaluator/inference load the saved vocabulary, reserve token 0 for CTC blank, and fail closed on checkpoint/vocabulary mismatch.

## Backend/API hardening diary
- Added bearer-auth dependency helpers.
- Anonymous base-model translation remains allowed for the public demo.
- Any user-scoped request requires a matching authenticated user.
- Adapter use requires authenticated ownership.
- Calibration requires authentication and matching user.
- Translation validates frame count, dimensions, finite values, and maximum frame count before PyTorch.
- Calibration validates target text/labels, true CTC minimum length, feature dimensions, finite values, and minimum duration.
- Calibration uses `CALIBRATION_MAX_FRAMES=256` by default and evenly downsamples long sessions before adapter fitting.
- Adapter parameter budget is now enforced before saving; out-of-budget adapters are rejected.
- Adapter deletion is owner-scoped and removes the stored file as well as the DB row using a reversible tombstone step around the DB transaction.

## Health/readiness diary
Problem: old `/health` returned HTTP 200 even when model readiness was degraded, so deployment infrastructure could consider a model-unavailable service healthy.
Fix:
- `/api/v1/health` = lightweight process liveness.
- `/api/v1/ready` = model/vocabulary readiness, returns HTTP 503 when not ready.
- Render `healthCheckPath` now points to `/api/v1/ready`.

Readiness metadata is cached by model/vocab file mtime+size so repeated deployment probes do not reload the 30 MB checkpoint every time.
Status: FIXED IN CODE, NOT RUNTIME-VERIFIED.

## Feature completion diary
### Authentication
Real registration/login UI and JWT bearer authentication added. Private pages redirect to authentication when no token is present.

### Dashboard
Hardcoded usage/model/activity cards replaced with authenticated backend data derived from translation logs, model readiness, and the latest signer adapter.

### History
Static history cards replaced with authenticated persistent history, search, date ranges, sorting, and CSV export.

### Signer profiles/adapters
Static cards replaced with real adapter listing and owner-scoped deletion. Adapter file lifecycle is now cleaned up with the database record.

### Evaluation
Fabricated accuracy/BLEU/WER/memory claims removed. Evaluation reports only persisted evidence and explicitly says when benchmark data is unavailable.

### Calibration
Simulated calibration animation replaced with real browser MediaPipe capture, target-text submission, backend vocabulary encoding, CTC validation, frame capping/downsampling, and adapter persistence.

### Live translation
Fake/demo translation fallback removed. Browser MediaPipe produces 132/1404 keypoints and sends real inference requests to `/api/v1/translate`.

## Frontend integrity diary
- Removed fake translation ticker and hardcoded model benchmark claims.
- Dashboard/history/users/evaluation now render server-backed state.
- Calibration page explicitly describes browser extraction and backend adapter fitting.
- Frontend JS syntax is now covered by CI using `node --check` on every `.js` file.
- API endpoint remains configurable through browser settings/local storage.

## Database/deployment diary
Current persistence is SQLAlchemy + SQLite.
Render free-tier storage is ephemeral, so the deployment is suitable for a disposable demo, not durable production persistence.
No production external DB has been wired yet.

## CI/testing diary
`.github/workflows/backend-tests.yml` now runs:
1. Python dependency installation.
2. Python `compileall` for `app` and `tests`.
3. `pytest tests -q`.
4. Node `--check` on every frontend JS file.

Regression tests cover CTC contracts, dataset UID integrity, feature contracts, model availability/padding, translation validation/auth, account workflows, evaluation helpers, calibration contracts, and adapter file lifecycle.

Latest workflow execution is NOT VERIFIED because no workflow run is available from the GitHub connector for the latest commit.

## Known architectural boundary
`/api/v1/translate` accepts pre-extracted pose/face keypoints. It does not decode arbitrary uploaded video server-side.
Production path is intentionally:
`camera -> browser MediaPipe -> keypoints -> backend -> model`.
Raw-video backend upload is not implemented.

## Remaining blockers / acceptance gates
A. Clean dataset rebuild -> NOT VERIFIED.
B. Real-data CTC sanity with non-blank useful target overlap -> NOT VERIFIED.
C. Full clean model retraining -> BLOCKED by B.
D. Multi-video unseen real validation -> BLOCKED by C.
E. BridgeAdapter quality validation on held-out real signers -> BLOCKED by D.
F. Latest full backend test run -> NOT VERIFIED.
G. Frontend browser/E2E runtime -> NOT VERIFIED.
H. Live Render health/CORS/auth/calibration/persistence -> NOT VERIFIED.
I. Durable production database -> NOT IMPLEMENTED.

## Issue diary timeline
### 2026-08-25
- Duplicate filename UID collision discovered -> collision-safe training IDs implemented.
- Real ISL clips produced 100% CTC blanks -> old checkpoint rejected.
- Deterministic train/validation split mismatch found -> seed 42 alignment fixed.
- Decoder vocabulary state bug found -> saved vocabulary validated before decode.
- 132/1404 dimension mismatches found -> dataset/API/converter guards added.
- Uppercase `.MP4` extractor failure found -> case-insensitive extension resolution fixed.
- Stale 1434-face converter contract found -> strict 1404 validation added.
- Adapter authorization gap found -> bearer auth + ownership checks added.
- Calibration CTC minimum-length and minimum-duration gaps found -> validation added.
- Silent tokenizer target-character dropping found -> strict target validation added.
- Fake frontend translations and unverified benchmark metrics found -> removed.
- Dashboard/history/users/evaluation were demo/static -> converted to backend/database workflows.
- Calibration was a mock -> converted to real browser capture + backend fitting.
- CI only covered backend tests -> Python compile + frontend JS syntax checks added.
- `/health` readiness semantics were unsafe -> liveness/readiness split added.
- Adapter DB delete orphaned weight files -> staged/reversible file lifecycle added.
- Calibration could train ~900 frames -> `CALIBRATION_MAX_FRAMES=256` cap and synchronized downsampling added.
- Adapter budget was only reported, not enforced -> calibration now fails closed before save.
- Readiness repeatedly reloaded checkpoint metadata -> readiness signature cache added.
- Synthetic inference test incorrectly demanded non-empty semantics from random features -> assertion corrected.

## Current next steps
1. Fresh Colab runtime.
2. Pull latest `main`.
3. Run `notebooks/train_base_model_colab.ipynb` from cell 1.
4. Require `DATASET INTEGRITY: PASS`.
5. Require `OVERFIT SANITY: PASS` with non-blank useful target overlap.
6. Run full training only after the gate passes.
7. Require train + held-out acceptance before pushing a new checkpoint.
8. Run the validation notebook on multiple real unseen videos.
9. Run `python -m pytest backend/tests` in a clean runtime.
10. Verify the latest GitHub Actions workflow is green.
11. Verify `/api/v1/health` and `/api/v1/ready` on the deployed backend.
12. Verify live frontend CORS/auth/history/calibration/translation journeys.
13. Move durable deployment persistence to a managed database before production.
14. Only then resume BridgeAdapter quality evaluation on held-out signers.

## Final verdict
**NOT READY.**
Reason: the engineering codebase has been substantially hardened, but the latest clean retraining, real-video model quality, full regression runtime, browser E2E, and live deployment are still unverified. Static correctness is not runtime proof.
