# VisionBridge Multi-Agent Engineering Contract + Project Diary

## Operating rules

This is the persistent hand-off record for ChatGPT/Codex, Claude, and future agents. Read it before editing. Every meaningful change, failure, test result, design decision, runtime blocker, and deployment finding must be recorded here. Distinguish `CODE FIXED`, `STATIC VERIFIED`, `RUNTIME VERIFIED`, `CI VERIFIED`, `CLAUDE-REPORTED`, and `NOT VERIFIED`.

Change only files required by the current task. Never use destructive Git cleanup in Colab. Never push a model because loss decreased, logits are finite, or a trivial token is emitted. A model checkpoint is acceptable only after real-data semantic validation.

Engineering loop:
```text
DISCOVER -> UNDERSTAND -> TRACE -> REPRODUCE -> ROOT CAUSE
-> DESIGN -> IMPLEMENT -> INTEGRATE -> TEST -> REGRESSION
-> REVIEW -> IMPROVE -> RE-AUDIT -> ZERO-LOOSE-ENDS
```

The project follows the uploaded autonomous engineering protocol. The protocol requires whole-repository reconnaissance, end-to-end feature completion, security/performance/UX reviews, regression testing, and a final evidence-based release verdict.

---

# Mission

Continuous Indian Sign Language -> English translation with few-shot signer adaptation.

Current intended pipeline:
```text
Camera / real video
 -> MediaPipe Holistic
 -> pose + face + left-hand + right-hand skeletons
 -> hand-aware temporal PyTorch model
 -> character CTC decoder
 -> English text
 -> optional BridgeAdapter signer personalization
```

Frontend is standardized on React + Vite + TypeScript. Backend remains FastAPI + SQLAlchemy. Training remains PyTorch with Colab/Lightning orchestration. Visual language: monochrome, high-contrast, neat, stylish, minimalistic.

---

# Current multimodal contract

| Stream | Features/frame |
|---|---:|
| Pose | 132 = 33 * (x,y,z,visibility) |
| Face | 1404 = 468 * (x,y,z) |
| Left hand | 63 = 21 * (x,y,z) |
| Right hand | 63 = 21 * (x,y,z) |
| CTC blank | 0 |
| Vocabulary | 49 |
| Maximum sequence | 1024 frames |

Every trained hand-aware sample must carry four synchronized streams.

---

# Hand-aware model and training status

The old pose+face checkpoint is invalid for the current four-stream production model. The production inference loader deliberately rejects the legacy artifact instead of partially loading it.

Current architecture:
```text
Pose encoder
Face encoder
Left-hand encoder
Right-hand encoder
        |
        v
Learned gated multimodal fusion
        |
        v
Temporal feature extractor
        |
        v
Bidirectional recurrent/temporal encoder
        |
        v
49-class character CTC head
```

Training protections currently include:
- trainable-parameter checks;
- first-batch gradient checks;
- finite-gradient validation;
- semantic overfit sanity gate;
- rejection of blank/space collapse;
- meaningful-token diversity checks;
- CER checks;
- resumable checkpoints;
- final checkpoint artifact verification.

Status:
```text
HAND-AWARE CODE: CODE FIXED
LEGACY CHECKPOINT: INVALID FOR NEW MODEL
NEW HAND-AWARE CHECKPOINT: NOT TRAINED / NOT ACCEPTED
MODEL QUALITY: NOT VERIFIED
```

---

# Frontend current state

## Stack
```text
React + Vite + TypeScript
```

Canonical frontend structure:
```text
frontend/
  index.html
  package.json
  vite.config.ts
  tsconfig.json
  public/
    favicon.svg
  src/
    main.tsx
    App.tsx
    api.ts
    landmarks.ts
    useLandmarkSession.ts
    styles.css
    vite-env.d.ts
```

Application routes:
```text
/dashboard
/translate
/calibration
/history
/evaluation
/settings
```

The live translation page uses the real browser camera, MediaPipe Holistic, synchronized pose/face/left-hand/right-hand frames, visible hand skeleton overlays, and API-backed prediction/confidence/latency/error reporting.

Calibration captures synchronized multimodal frames and submits the captured sequence for adapter fitting rather than using a timer-only placeholder.

The obsolete duplicate static frontend was removed. React is the canonical frontend source.

## Visual system

The frontend now uses a deliberate monochrome visual system:
- white primary canvas;
- black structural/sidebar elements;
- grayscale secondary text;
- strong black borders;
- restrained editorial serif headings;
- subtle grid texture;
- offset black/gray shadows;
- responsive layouts retained;
- no backend/theme changes.

The browser page metadata is also aligned to the white theme and references the VisionBridge favicon. The favicon is a monochrome VisionBridge mark.

## 404 handling

The SPA contains a first-class not-found page using the same frontend visual system. Unknown application routes do not fall back to an unrelated legacy page.

## Authentication presentation

The frontend authentication flow remains a lightweight client-side presentation layer while the database-backed authentication integration is deferred. No new backend authentication behavior was introduced during the frontend polish work.

---

# Render/Vite deployment recovery

A live Render error exposed that the static service was serving the source Vite `index.html`, which requested `/src/main.tsx` and caused:
```text
Failed to load module script: Expected a JavaScript-or-Wasm module script but the server responded with a MIME type of binary/octet-stream.
```

A secondary compatibility probe for `/dist/deploy-marker.txt` returned 404, proving the live service was not exposing the expected nested `dist` path.

Deployment recovery therefore moved toward publishing the compiled Vite artifact at the service root when required, rather than relying on the source tree being the published directory.

Relevant deployment work includes:
- pinned frontend runtime dependencies;
- explicit Vite production artifact verification;
- deterministic production artifact marker;
- Render-safe artifact path handling;
- explicit static publish configuration;
- root-level compiled artifact fallback for an existing misconfigured static service.

The live Render site could not be independently fetched from this environment, so live browser deployment remains `NOT VERIFIED` until checked in the actual browser/Render service.

---

# Verification ledger

## Verified through GitHub source inspection
```text
React/Vite/TypeScript frontend structure        STATIC VERIFIED
Monochrome frontend styling                     STATIC VERIFIED
Favicon reference and asset                     STATIC VERIFIED
404 route/page implementation                   STATIC VERIFIED
Four-stream API contract                        STATIC VERIFIED
Hand-aware training protections                 STATIC VERIFIED
Render/Vite build configuration                  STATIC VERIFIED
```

## GitHub Actions

The repository has a regression workflow that checks backend Python compilation/tests and frontend TypeScript/Vite production build plus artifact verification.

Known successful historical runs include the post-migration frontend dependency/build fixes, including the published Vite plugin version and Render build-path changes.

The latest frontend-theme commit triggered GitHub Actions run #144, but at the last inspection it was still queued. Therefore that specific run was not yet a final CI verdict.

Status:
```text
REPOSITORY CI: HISTORICALLY VERIFIED FOR PRIOR FIXES
LATEST COMMIT CI: NOT VERIFIED UNTIL RUN COMPLETES
```

## Runtime

Local execution from this environment was previously blocked because the container could not resolve `github.com`, so local browser/runtime execution is not treated as verification.

Render live verification is also not complete from this environment.

---

# Known production gaps

```text
SQLite on Render              -> ephemeral unless external DB is connected
Client-side bearer storage    -> still present
Rate limiting                 -> not implemented
Raw video server inference    -> not implemented
Production database wiring   -> pending user DB connection
```

These are tracked engineering gaps, not reasons to claim a production-ready backend when the evidence does not support it.

---

# Required acceptance gates

## Gate A — Dataset
```text
metadata
pose
face
left_hand
right_hand
unique UID
aligned frame counts
finite features
valid targets
valid CTC alignment
```

## Gate B — Semantic overfit
Must demonstrate finite loss, meaningful reduction, non-trivial output, meaningful-token diversity, and acceptable CER.

## Gate C — Full training
Only after Gate B.

## Gate D — Multi-sample train/held-out acceptance
Use several train and validation examples and reject trivial output.

## Gate E — Real-video validation
Record ground truth, prediction, confidence, CER, blank ratio, space ratio, frame count, and all stream shapes.

## Gate F — Application E2E
Verify:
```text
auth
-> dashboard
-> live translation
-> hand skeleton overlay
-> calibration
-> adapter
-> history
-> evaluation
-> settings
```

---

# Current blocker board

```text
A  fresh hand-aware Colab extraction             NOT VERIFIED
B  hand-aware semantic overfit                   NOT VERIFIED
C  full hand-aware training                      BLOCKED until B
D  new hand-aware checkpoint                     BLOCKED until C
E  multi-video real validation                   BLOCKED until D
F  adapter calibration on new checkpoint         BLOCKED until E
G  latest GitHub Actions run                     NOT VERIFIED
H  local frontend build                          NOT VERIFIED here
I  browser camera + hand overlay                 NOT VERIFIED here
J  Render E2E after Vite migration               NOT VERIFIED here
K  durable production DB                         NOT IMPLEMENTED
L  production-grade HttpOnly auth                NOT IMPLEMENTED
M  production rate limiting                      NOT IMPLEMENTED
N  40 GB scale                                   BLOCKED until correctness
```

---

# Chronological diary

## 2026-08-27 — Full protocol redesign + hand-aware migration

Actions:
```text
1. Reconstructed the ML model as a hand-aware four-stream temporal architecture.
2. Extended extraction to left/right MediaPipe hand skeletons.
3. Extended dataset loading/collation to four synchronized streams.
4. Extended training and semantic overfit gate to four streams.
5. Extended translation/calibration APIs and services.
6. Changed model readiness so the old legacy checkpoint is unavailable until retraining.
7. Added React/Vite/TypeScript application structure.
8. Added shared landmark/session modules and visible 21-point hand skeletons.
9. Replaced timer-only calibration with real capture/submission flow.
10. Migrated the Render frontend to Vite build/dist + SPA routing.
11. Removed duplicate legacy static pages and assets so React is the single frontend source.
12. Updated canonical Colab and validation workflows for hand-aware data.
13. Rebuilt Lightning notebook around the canonical repository trainer.
14. Added hand model/collation regression coverage.
```

Status:
```text
SOURCE STRUCTURE: STATIC VERIFIED
MODEL QUALITY: NOT VERIFIED
RUNTIME: NOT VERIFIED
```

## 2026-09-10 — Frontend build/deployment recovery

Observed Render browser error:
```text
main.tsx:1 Failed to load module script
Expected a JavaScript-or-Wasm module script but the server responded with MIME type binary/octet-stream
```

Root cause:
```text
Live Render service was serving source frontend/index.html
instead of the compiled Vite artifact.
```

Actions:
```text
1. Pinned frontend dependencies to known published versions.
2. Hardened the production build verification script.
3. Added deterministic production artifact marking.
4. Made Vite artifact paths deployment-safe.
5. Made Render static publish configuration explicit.
6. Added a fallback deployment strategy for an existing service publishing frontend/ instead of frontend/dist.
```

Status:
```text
CODE FIXED: YES
SOURCE CONFIG: STATIC VERIFIED
LIVE RENDER: NOT VERIFIED HERE
```

## 2026-09-10 — Monochrome frontend polish

Actions:
```text
1. Converted the browser theme to a white/black monochrome system.
2. Preserved application functionality and route structure.
3. Added the VisionBridge favicon.
4. Added the application 404 page within the same visual system.
5. Kept backend code untouched.
```

Status:
```text
FRONTEND THEME: CODE FIXED
FAVICON: CODE FIXED
404 PAGE: CODE FIXED
BACKEND: UNCHANGED FOR THIS POLISH
```

## 2026-09-10 — Current engineering state

The source tree now reflects the monochrome frontend and the updated deployment configuration. The project diary has been refreshed to reflect the current frontend structure, favicon/404 additions, deployment recovery work, and current verification boundaries.

Do not mark Render or browser runtime as verified without observing the deployed service and browser behavior directly.
Do not mark the new hand-aware model as accepted until the real-data semantic gates pass.

---

# Required next execution

1. Confirm GitHub Actions on the newest commit reaches a completed conclusion.
2. Verify the Render frontend serves compiled assets from the service root and no longer requests `/src/main.tsx`.
3. Verify the favicon loads with status 200.
4. Verify an invalid route renders the application 404 page.
5. Verify the frontend authentication presentation works without changing backend behavior.
6. Connect the production database when ready and replace the deferred client-side auth layer with the backend auth flow.
7. Separately execute the hand-aware Colab semantic overfit gate before any full training or model push.
8. Run multi-video acceptance and then full application E2E.
9. Only after correctness is proven consider larger 40 GB scale training.

---

# Final release verdict

## NOT READY

Reason:
- the current hand-aware model has not yet passed the required real-data semantic gates;
- the previous checkpoint remains invalid for the new model;
- live Render/browser verification is incomplete from this environment;
- production database/auth hardening remains pending.

The project can move toward `READY WITH MINOR ISSUES` only after the relevant evidence-based gates pass.