# VisionBridge Model Debug Log

This file records model-specific failures, fixes, experiments, validation evidence, and unresolved ML blockers for the active letter-recognition architecture.

ACTIVE PIPELINE

    MediaPipe Tasks Hand Landmarker 0.10.35
     -> normalized 126D two-hand landmarks
     -> dynamic VisionBridgeLetterBaseModel
     -> 64D embedding
     -> few-shot signer adapter
     -> A-Z letter + confidence

This is the only active ML architecture described here. Model weights are versioned and replaceable; no component is treated as permanently immutable.

---


---

# 0A. 2026-09-24 — Self-correcting audit restart

RESTART #1
FAILED STEP: full repository / data / ML contract audit
ERRORS FOUND:
    1. Backend adapter loader silently rebuilt prototypes after a base-checkpoint change.
    2. Calibration used the recognition rate-limit setting instead of the dedicated calibration limit.
    3. Rate limiting keyed bearer traffic by raw token, allowing fabricated tokens to bypass a shared identity bucket.
    4. Render retained an unused calibration-minimum environment setting.
    5. Documentation described the 64D embedding as signer-independent before signer-holdout evidence existed.
ROOT CAUSE:
    Adapter lifecycle did not enforce the same strict checkpoint binding as the browser adapter.
    Rate-limit wiring and identity bucketing were inconsistent with their documented contracts.
    Documentation overstated an unverified property.
CORRECTION:
    Require exact base-model version, SHA-256, embedding dimension, label vocabulary, and adapter method/version.
    Require explicit calibration throttling and use user identity for valid bearer tokens with IP fallback for invalid tokens.
    Remove stale deployment configuration and describe signer independence only as a required evaluation gate.
FIX VERIFIED:
    Static source re-audit confirms the offending auto-refresh path is removed and the dedicated limiter wiring is present.
DOWNSTREAM RESULTS INVALIDATED:
    No real-data model metrics existed, so no model-quality result was reused.
FULL RESTART COMPLETED:
    Static repository re-audit completed after corrections.
FINAL STATUS:
    Code corrections recorded. Full runtime, real-data, held-out signer, browser, and Render gates remain NOT VERIFIED or BLOCKED as documented.


---

# 0B. 2026-09-24 — Restart #2

FAILED STEP: post-fix full re-audit
ERRORS FOUND:
    1. Base-model cache invalidation depended only on mtime, size, and inode.
    2. README still contained legacy MediaPipe Hands runtime wording.
    3. Browser adapter loading did not validate preprocessing or landmark-runtime metadata.
ROOT CAUSE:
    Checkpoint identity did not include content hash in the in-memory cache key.
    Documentation and browser adapter validation had not been updated with the complete Tasks/runtime contract.
CORRECTION:
    Add checkpoint SHA-256 to the hot-reload signature.
    Align README runtime wording with MediaPipe Tasks Hand Landmarker 0.10.35.
    Enforce browser-side adapter preprocessing and landmark-runtime metadata compatibility.
FIX VERIFIED:
    Static re-audit confirms the new cache signature and adapter metadata guards are present.
DOWNSTREAM RESULTS INVALIDATED:
    No real-data model-quality results existed to invalidate.
FULL RESTART COMPLETED:
    Static repository re-audit completed again after these corrections.
FINAL STATUS:
    Source-level hardening is complete for the discovered issues; runtime and ML evidence remain explicitly unverified or blocked.


---

# 0C. 2026-09-24 — Restart #3

FAILED STEP: post-hardening documentation re-audit
ERROR FOUND:
    AGENTS.md retained an obsolete MediaPipe Hands model-complexity description inside the current architecture diary.
ROOT CAUSE:
    The earlier historical performance entry was not rewritten when the browser tracker moved to the version-pinned MediaPipe Tasks runtime.
CORRECTION:
    Align the diary's executable architecture and performance contract with MediaPipe Tasks Hand Landmarker 0.10.35.
FIX VERIFIED:
    Static source re-audit confirms no active MediaPipe Hands configuration remains in the repository documentation checked in this cycle.
DOWNSTREAM RESULTS INVALIDATED:
    No model metrics or runtime evidence depended on the stale documentation.
FULL RESTART COMPLETED:
    Documentation re-audited after correction.
FINAL STATUS:
    Documentation consistency gate passes for the corrected runtime terminology; model, signer, browser, and deployment evidence remain unverified or blocked.


---

# 0D. 2026-09-24 — Restart #4

FAILED STEP: post-hardening application/data/deployment audit
ERRORS FOUND:
    1. Render used model readiness as process health while the real checkpoint remained intentionally absent.
    2. Dataset preparation accepted any existing hand-landmarker.task without checking the pinned asset identity.
    3. Failed camera startup could leave tracker/stream resources alive.
    4. Local demo storage was global rather than scoped to the active local user.
    5. Local prediction mode persisted every poll instead of applying the same event throttling as real mode.
CORRECTION:
    Separate /health liveness from /ready model readiness.
    Pin the hand-landmarker asset by size and SHA-256 in dataset preparation.
    Register tracker and camera resources before operations that can fail so shared cleanup handles them.
    Scope local adapters/history by local username and throttle local history events.
FIX VERIFIED:
    Static source re-audit confirms each corresponding correction is present.
DOWNSTREAM RESULTS INVALIDATED:
    No model-quality metrics depended on these application/deployment fixes.
FULL RESTART COMPLETED:
    Source-level repository re-audit completed after corrections.
FINAL STATUS:
    Corrected code paths are recorded; full runtime/build/ML/deployment evidence is still bounded by the unavailable local dependency checkout and disabled CI.


---

# 0E. 2026-09-24 — Restart #5

FAILED STEP: post-hardening strict-contract audit
ERRORS FOUND:
    1. Checkpoint loaders accepted missing preprocessing/runtime metadata.
    2. Adapter loaders accepted missing preprocessing/runtime metadata.
    3. Adapter payloads persisted raw calibration landmarks after automatic re-embedding had already been removed.
CORRECTION:
    Require exact preprocessing_version and landmark_runtime on checkpoints and adapters.
    Stop persisting raw calibration landmarks; require explicit recalibration after base-model changes.
FIX VERIFIED:
    Static source review confirms exact metadata equality checks and no calibration_samples field in fitted adapters.
DOWNSTREAM RESULTS INVALIDATED:
    No real-data model-quality result existed to invalidate.
FULL RESTART COMPLETED:
    Source-level repository re-audit completed after corrections.
FINAL STATUS:
    Strict model-contract and calibration-data minimization gates are corrected; real-data, signer-holdout, browser, and deployment evidence remain unverified or blocked.


---

# 0F. 2026-09-24 — Ledger completeness correction

Additional commits from the preceding rate-limit hardening cycle:
    5ac7147b80dc06dbd0b0d8a4e35002084c74f6d6 — prune stale rate limit buckets
    0a01b326006917aa997baff2fe35449efe7c766d — cover stale rate limit bucket cleanup

These changes were already implemented; this entry closes the documentation ledger gap identified during re-audit.


---

# 10. 2026-09-24 — Restart #6

FAILED STEP: application contract audit
ERROR FOUND:
    The frontend enforced three calibration shots per letter, but the backend schema accepted fewer.
ROOT CAUSE:
    The shot-count rule existed only in the browser workflow instead of the API contract.
CORRECTION:
    Add a Pydantic request validator requiring at least three examples for every calibrated letter.
FIX VERIFIED:
    Schema regression test covers both rejection of insufficient shots and acceptance of a valid request.
DOWNSTREAM RESULTS INVALIDATED:
    No ML metrics depended on the request-schema correction.
FULL RESTART COMPLETED:
    Source-level re-audit completed after the correction.
FINAL STATUS:
    Calibration shot-count contract is enforced; real-data and signer-holdout evidence remain NOT VERIFIED / BLOCKED.


---

# 11. 2026-09-24 — Authoritative-protocol execution iteration 1 (FAILED)

STEP 1-4 execution uncovered multiple contract defects and one environment execution limit.

External execution limit:
    Local container could not clone GitHub because DNS/network access to github.com is unavailable.
    This is an environment limitation, not evidence that the repository clone is broken.

Defects fixed before restart:
    camera permissions policy disabled getUserMedia                     FIXED
    V3 checkpoint loader accepted wrong input/class contract             FIXED
    non-finite checkpoint tensors were not rejected                     FIXED
    train/validation split could leave a class on one side only         FIXED
    train/test exact duplicates were only warned about                  FIXED
    notebook hand-model asset check was incomplete                      FIXED
    local demo calibration bypassed three-shot requirement               FIXED

Verification boundary:
    source-level checks                      STATIC VERIFIED
    local full test suite                    NOT VERIFIED
    local build                             NOT VERIFIED
    real-data preparation                   NOT RUN
    real-data training                      NOT RUN
    signer-independent evaluation           REQUIRED / BLOCKED BY VERIFIED SIGNER METADATA
    browser camera runtime                  NOT VERIFIED
    Render runtime                          NOT VERIFIED

Protocol result:
    RESULT: FAILED
    RESTART REQUIRED: YES
    downstream evidence invalidated: YES


---

# 12. 2026-09-24 — Authoritative-protocol execution iteration 2 (FAILED)

During the second full pass, legitimate model/data integrity defects were corrected.

One discovered diagnosis was later proven incorrect:
    The FastAPI Permissions-Policy header was interpreted as governing the separately served frontend page.
    That assumption was wrong for this deployment shape, so the camera-policy change was reverted.

Reverted commits:
    87f94197ca08e75984cebba070a161c3efc61d3c — allow camera under permissions policy
    05cda46694eddd051059552eab9169c40403c678 — camera policy assertion

Retained corrections:
    active V3 input/class contract enforcement
    finite checkpoint state validation
    A-Z training vocabulary enforcement
    train/validation split coverage guard
    source-test duplicate exclusion from training/validation
    notebook hand-model asset integrity verification
    local three-shot calibration parity

Protocol result:
    RESULT: FAILED
    RESTART REQUIRED: YES
    downstream evidence invalidated: YES


---

# 13. 2026-09-24 — Authoritative-protocol execution iteration 3 (FAILED)

REPRODUCTION:
    The supplied V3 checkpoint was loaded through the current strict model loader and failed on missing preprocessing/runtime metadata.

ARTIFACT INSPECTION:
    model_version = visionbridge-letter-base-v3
    input_dim = 126
    hidden_dim = 128
    embedding_dim = 64
    num_classes = 26
    labels = A-Z
    parameters = 26,582
    all learned tensors finite
    supplied SHA-256 = 2b42639e0ffb3c40112bf931f434f6adf5b578fce21399ba53795d3fba0529a

ROOT CAUSE:
    The supplied checkpoint predates the metadata-bound checkpoint envelope.

FIX:
    Added backend/scripts/migrate_v3_checkpoint.py to validate the legacy V3 structure and add only the current preprocessing/runtime metadata.

VERIFICATION:
    migrated artifact loaded with the current strict loader
    tensor values unchanged
    model output shape = [16,26]
    embedding shape = [16,64]
    logits and embeddings finite

PROTOCOL RESULT:
    RESULT: FAILED
    RESTART REQUIRED: YES
    downstream evidence invalidated: YES


---

# 14. 2026-09-24 — Authoritative-protocol execution iteration 4 (FAILED)

FAILED STEP: Step 4 / Step 6 camera runtime data-flow audit

Problem:
    Async camera startup could outlive an explicit stop/unmount during getUserMedia() or video.play().

Root cause:
    Startup had no generation-based cancellation boundary after every externally awaiting operation.

Fix:
    Add startingRef guard.
    Add startGenerationRef token.
    Check generation before getUserMedia, after getUserMedia, and after video.play().
    Explicitly stop acquired stream tracks and close the tracker on failed/cancelled startup.

Verification:
    current source inspection confirms the generation checks and cleanup path.
    real browser lifecycle execution remains NOT VERIFIED.

PROTOCOL RESULT:
    RESULT: FAILED
    RESTART REQUIRED: YES
    downstream evidence invalidated: YES


---

# 15. 2026-09-24 — Authoritative-protocol execution iteration 5 (FAILED)

FAILED STEP: Step 1 / Step 9 consistency audit

Findings:
    1. The notebook declared a fixed hand-landmarker SHA-256 but initially did not enforce it.
    2. The active V3 model loader now fixes 126D/A-Z, while README/AGENTS still described input and vocabulary as freely scalable.

Fixes:
    Notebook now computes SHA-256 and exact size after download.
    Documentation now states that active V3 fixes the 126D input and A-Z vocabulary; future input/vocabulary changes require a new model version and complete validation cycle.

Verification:
    notebook Cell 2 contains exact size and SHA checks
    README and AGENTS describe the active V3 contract consistently

PROTOCOL RESULT:
    RESULT: FAILED
    RESTART REQUIRED: YES
    downstream evidence invalidated: YES

# 1. Active model contract

    Input:          126 normalized landmark values
    Embedding:      64 dimensions
    Output classes: 26 (A-Z)
    Loss:           CrossEntropyLoss
    Base model:     VisionBridgeLetterBaseModel
    Adapter:        dynamic-base-embedding-prototype

Expected checkpoint:

    backend/app/models/weights/letter_base_model.pt

The base model remains trainable, versioned, and replaceable after each validated training cycle.

The few-shot adapter is fitted from signer calibration examples. It does not require a separate offline training job.

---

# 2. Landmark and normalization contract

Each sample contains:

    left hand:  63 values
    right hand: 63 values
    total:     126 values

Preprocessing:

    MediaPipe Tasks Hand Landmarker 0.10.35
     -> raw Tasks handedness left/right placement
     -> wrist-relative coordinates
     -> scale normalization
     -> missing-hand zero fill
     -> 126D vector

Training and runtime must use the same representation.

Any preprocessing change requires:

    dataset regression
     -> base-model validation
     -> adapter validation

Do not compare a checkpoint trained with one feature contract against inference features from another contract.

---

# 3. Dataset integrity lesson

A previous VisionBridge training path exposed a general dataset-preparation failure mode: identifiers derived only from filenames can collide when different physical samples reuse the same filename.

Failure pattern:

    two physical samples
     -> same identifier
     -> same processed output path
     -> later sample overwrites earlier sample
     -> labels and features become misaligned
     -> training metrics become misleading

The active letter preparation path therefore requires:

    globally unique sample identifiers
    duplicate-ID detection
    clean runtime feature output
    explicit train/validation/test artifacts

This lesson remains active because it protects the new letter training pipeline.

---

# 4. Base model implementation

Current model:

    VisionBridgeLetterBaseModel

Architecture:

    LayerNorm(input_dim)
     -> Linear(input_dim -> hidden_dim)
     -> GELU
     -> Dropout(dropout)
     -> Linear(hidden_dim -> embedding_dim)
     -> LayerNorm(embedding_dim)
     -> GELU
     -> Linear(embedding_dim -> num_classes)

The model must expose the embedding width and class vocabulary declared by its checkpoint.

The base model remains trainable, versioned, and replaceable during signer adaptation; a changed model triggers adapter recalibration.

---

# 5. Few-shot adapter implementation

The active adapter operates in base-model embedding space:

    126D landmarks
     -> base model
     -> 64D embedding
     -> normalization
     -> per-letter prototype
     -> cosine similarity
     -> confidence
     -> A-Z or ?

Current calibration behavior:

    3 shots per selected letter
    one normalized prototype per letter
    base checkpoint SHA-256 stored with adapter
    adapter rejected when checkpoint hash does not match
    minimum similarity threshold = 0.35

This is a prototype-based few-shot adapter.

Do not describe it as a separately trained neural adapter.

---

# 6. Debugging rules

When a model failure occurs, capture evidence before changing architecture.

For every training or debugging run record:

    dataset source
    dataset version/reference
    class count
    sample count
    feature shape
    missing-hand distribution
    NaN/Inf count
    class distribution
    random seed
    hyperparameters
    epoch count
    best validation metric
    held-out test metric
    prediction distribution
    checkpoint path
    checkpoint version/hash

For numerical failures also record:

    first loss
    last valid loss
    gradient finite status
    parameter-update status
    logit finite status
    embedding finite status

For recognition failures also record:

    predicted letter
    expected letter
    confidence
    similarity scores
    calibration letters
    shot count
    base checkpoint hash

Do not diagnose a model from a single successful prediction.

---

# 7. Base-model validation gate

A newly trained base checkpoint is not accepted until all of the following are checked:

    A-Z labels are correct
    126D inputs are valid
    embedding size matches checkpoint metadata
    logits have shape [batch, configured class count]
    loss is finite
    gradients are finite
    parameters update during training
    validation performance improves
    predictions are not collapsed
    held-out test performance is measured
    checkpoint reload succeeds
    checkpoint metadata is correct

A low loss with collapsed predictions is a failure, not a success.

---

# 8. Few-shot adapter validation gate

After base-model validation:

    select calibration letters
    capture signer examples
    fit prototypes
    save adapter
    reload adapter
    validate base checkpoint hash
    predict held-out signer samples

The adapter must be evaluated on examples that were not used to create its prototypes.

Report at minimum:

    calibration shot count
    calibrated letters
    held-out sample count
    correct predictions
    accuracy
    unknown/rejected count
    mean confidence
    per-letter failures

Do not claim signer adaptation works solely because calibration completes without an exception.

---

# 9. Real-time debugging gate

When browser inference is enabled, trace the complete path:

    camera frame
     -> MediaPipe Tasks Hand Landmarker 0.10.35
     -> handedness
     -> normalization
     -> 126D vector
     -> API payload
     -> base checkpoint
     -> 64D embedding
     -> adapter lookup
     -> similarity scores
     -> letter
     -> confidence

For a failing prediction, determine the first stage where values become invalid or semantically wrong.

Useful evidence:

    hands detected: yes/no
    left landmarks: present/absent
    right landmarks: present/absent
    feature norm
    embedding norm
    top prototype scores
    similarity gap
    confidence
    latency

Do not tune confidence thresholds before verifying feature and embedding correctness.

---

# Real-time latency contract

The active real-time hot path is:

    camera
     -> MediaPipe Tasks Hand Landmarker 0.10.35
     -> hand tracing
     -> normalized 126D vector
     -> browser model
     -> browser adapter
     -> prediction

Do not put FastAPI inference inside the per-frame hot path.

Measure these separately:

    hand-tracker latency
    feature-normalization latency
    browser model latency
    adapter similarity latency
    UI update cadence
    end-to-end prediction latency

A network request is acceptable for:

    initial model load
    adapter load
    asynchronous history/event persistence

A network request is not acceptable for every camera frame.

The hand tracing overlay must remain lightweight: 21-point skeleton + short wrist trail, with no canvas resize on every frame.

---

# 10. Current training state

Known source status:

    Base model code:                 HISTORICAL CI VERIFIED
    Dataset preparation code:        HISTORICAL CI VERIFIED
    Training CLI:                    HISTORICAL CI VERIFIED
    Colab notebook:                  STATIC VERIFIED
    Few-shot adapter code:           HISTORICAL CI VERIFIED
    Checkpoint compatibility:        HISTORICAL CI VERIFIED
    Uploaded V3 checkpoint structure: STATIC VERIFIED
    Uploaded V3 checkpoint accuracy:  NOT VERIFIED
    Repository checkpoint install:    PENDING
    Base held-out accuracy:           NOT VERIFIED
    Signer held-out accuracy:         REQUIRED / BLOCKED pending signer metadata
    Browser real-model inference:     NOT VERIFIED

Known successful code verification:

    GitHub Actions run #197
    backend: 72 passed, 1 skipped
    Python compilation: PASS
    frontend TypeScript check: PASS
    Vite production build: PASS
    production artifact verification: PASS

These results verify source behavior and regression coverage. They do not establish model accuracy.

---

# 11. Current ML blocker

The active base model has not yet been trained and accepted from a verified real-data run.

Required execution:

    notebooks/train_letter_base_colab.ipynb

Expected artifact:

    backend/app/models/weights/letter_base_model.pt

Standard training command:

    PYTHONPATH=backend python -m app.training.letter_base
      --data-dir /content/visionbridge_letter_data
      --output backend/app/models/weights/letter_base_model.pt
      --epochs 500
      --batch-size 128
      --lr 0.001
      --patience 6

After training, record:

    best validation accuracy
    held-out test accuracy
    prediction distribution
    checkpoint hash
    training configuration
    warnings or failures

Do not report the model as trained until the artifact actually exists and reloads successfully.

---

# 12. Debugging a bad base-model result

If test accuracy is poor:

    1. verify labels.json and class mapping
    2. verify train/validation/test class coverage
    3. inspect landmark extraction failures
    4. compare training and inference normalization
    5. inspect missing-hand frequency
    6. inspect class imbalance
    7. inspect prediction collapse
    8. inspect embedding separability
    9. verify checkpoint metadata
    10. only then consider architecture or hyperparameter changes

If training loss does not decrease:

    check feature variance
    check labels
    check optimizer configuration
    check gradient flow
    check learning rate
    check parameter updates
    check for NaN/Inf

If validation is much worse than training:

    check leakage assumptions
    check signer/data distribution differences
    check preprocessing consistency
    check class imbalance
    check overfitting

If live predictions are poor after good base-model test results:

    check camera landmark quality
    check mirrored input and handedness handling
    check live normalization
    check base checkpoint hash
    check calibration samples
    check held-out signer examples

Do not change several variables simultaneously during diagnosis.

---

# 13. Checkpoint safety

A checkpoint is accepted only when:

    architecture matches
    input_dim matches active feature contract
    embedding_dim matches checkpoint metadata
    num_classes matches checkpoint vocabulary
    labels match checkpoint vocabulary
    state_dict loads strictly
    test result is recorded

The runtime must reject missing, malformed, incompatible, or stale checkpoints.

The adapter must remain bound to the exact base checkpoint used to create its prototypes.

The active checkpoint and adapter loaders require exact preprocessing and
landmark-runtime metadata. Missing metadata is incompatible.

Raw calibration landmarks are not persisted in the active adapter payload,
because base-model changes require explicit recalibration rather than silent
re-embedding of stored samples.

No silent fallback to an incompatible model is permitted in real mode.

---

# 14. Current loose ends

    BASE MODEL TRAINING / UPGRADE     NOT VERIFIED
    BASE HELD-OUT TEST               NOT VERIFIED
    FEW-SHOT HELD-OUT SIGNER TEST    NOT VERIFIED
    LIVE CAMERA REAL MODE            NOT VERIFIED
    RENDER REAL MODE                 NOT VERIFIED

There is no second hidden offline ML training task.

Intended lifecycle:

    train or upgrade the base
     -> keep the current model replaceable
     -> calibrate signer with a few shots
     -> evaluate held-out signer examples
     -> deploy/use

---

# 15. Debug-log update rule

Every future model investigation must add a dated entry containing:

    date
    symptom
    reproduction
    evidence
    root cause
    fix
    verification
    remaining uncertainty

Do not rewrite measured historical results.

Do not insert accuracy numbers that were not produced by an actual run.

Do not preserve obsolete architecture descriptions as active instructions.

The active dynamically replaceable model source of truth remains:

    126D hands
     -> dynamic base model
     -> 64D embedding
     -> few-shot signer adapter
     -> letter


---

# 2026-09-23 — Browser fast path

### Problem

The previous real-mode UI sent prediction requests to FastAPI from the browser during live recognition. Network round trips were therefore part of the prediction loop.

### Fix

The active prediction path now loads the current model and signer adapter once and performs inference in the browser:

    camera
     -> MediaPipe Tasks Hand Landmarker 0.10.35
     -> normalized 126D vector
     -> browser base model
     -> browser few-shot adapter
     -> letter + confidence

Backend calls remain for:

    initial model load
    adapter load
    calibration persistence
    throttled recognition-event logging

The UI also renders:

    21-point hand skeleton
    left/right hand labels
    short wrist-motion trace
    tracker FPS
    measured local inference time

### Performance decisions

    MediaPipe Tasks Hand Landmarker 0.10.35
    640x480 ideal camera input
    one in-flight tracker call
    no per-frame network request
    cached browser model
    cached selected adapter
    typed-array model math
    lightweight canvas tracing

### Verification

    GitHub Actions run #197
    backend: success
    frontend: success

This verifies source/test/build integration. Real device latency and real signer accuracy remain NOT VERIFIED.

### Remaining uncertainty

    actual MediaPipe latency varies by device/browser
    real-world signer accuracy requires the trained base checkpoint
    browser fast path requires a trained checkpoint in real mode

## 2026-09-23 — MediaPipe Tasks training compatibility

The dataset-preparation path now uses MediaPipe Tasks Hand Landmarker in image mode with the versioned Google-hosted hand_landmarker.task model. The old mp.solutions API is not used by the active training path.

## 2026-09-23 — RealSign download path

The RealSign Dataset.zip file is stored through Git LFS. The training notebook downloads the actual archive from the Git LFS media endpoint and validates it with zipfile.is_zipfile before extraction. The ordinary raw GitHub file endpoint returns the small LFS pointer text instead of the 656 MB archive.


# 2026-09-24 — Critical ML pipeline corrections

### Findings corrected

1. Browser landmark extraction no longer uses the legacy `@mediapipe/hands` CDN. The browser now uses the same MediaPipe Tasks Vision 0.10.35 family and the same Google-hosted hand_landmarker.task bundle as dataset preparation.

2. The browser no longer swaps left/right handedness. Training and browser inference now place landmarks according to the Tasks result handedness labels using the same 126D ordering.

3. The 126D preprocessing contract is versioned as `two-hand-wrist-scale-v1`, and the active landmark runtime is recorded as `mediapipe-hand-landmarker-0.10.35`. Checkpoints and signer adapters carry this metadata.

4. Training dependencies are pinned for reproducibility:
    mediapipe==0.10.35
    numpy==2.1.3
    torch==2.9.0+cpu

5. The canonical letter evaluator now reports:
    overall accuracy
    macro class accuracy
    per-letter accuracy
    worst-letter accuracy
    confusion matrix
    parameter count
    checkpoint size
    model-only latency

6. Dataset preparation now hashes source images, preserves train/validation/test manifests, and keeps exact duplicate images together during the generated 80/20 train-validation split. Any exact duplicate between the training/validation pools and source test split is reported without modifying the test set.

### Verification status

    CODE FIXED
    STATIC VERIFIED

    Browser MediaPipe Tasks runtime: NOT VERIFIED on a real browser/device
    Base-model test accuracy: NOT VERIFIED in this environment
    Signer-independent evaluation: BLOCKED until verified signer metadata/manifest is available

### Signer-evaluation boundary

The RealSign repository documents four signers and separate class-based Training, Validation, and Testing folders, but its documented folder structure does not provide signer IDs in the prepared class directories. Therefore the generated random 80/20 split is a class-balanced sample split, not a signer-holdout experiment. A signer-independent result must only be reported after verified signer metadata is available.


## 2026-09-24 — Uploaded V3 checkpoint audit

The user-supplied checkpoint was inspected before repository installation.

    model_version: visionbridge-letter-base-v3
    input_dim: 126
    hidden_dim: 128
    embedding_dim: 64
    classes: 26
    size_bytes: 111205
    sha256: 2b42639e0ffb3c40112bf931f434f6adf5b578fce21399ba53795d3fba0529a

Status:

    checkpoint structure: STATIC VERIFIED
    checkpoint accuracy: NOT VERIFIED
    repository installation: PENDING

The checkpoint must not be described as accuracy-validated until it has been evaluated on the untouched test split with the canonical evaluator.


## 2026-09-24 — Protocol iteration 11 failure

The three-shot calibration contract was correctly enforced in the service, but
existing successful adapter tests still contained two-shot fixtures. Commit
9ea571e7426707182453f81496cd604bc86ed9aa aligned those fixtures with the
runtime contract.

During verification, a separate latent contradiction was found: the dynamic
constructor permits custom label vocabularies, but the active checkpoint loader
strictly requires the production A-Z/26-class vocabulary. The existing
checkpoint round-trip test expected a three-class checkpoint to load.

Per the authoritative restart contract, iteration 11 is failed and all
downstream evidence is invalidated. Iteration 12 must restart at Step 1 and
resolve the checkpoint/test contract contradiction before further evidence is
accepted.


## 2026-09-24 — Protocol iteration 12 failure

Iteration 12 resolved the production A-Z checkpoint-test contradiction, tightened
browser compatibility checks, and changed Render to publish only the Vite build
output.

Verification then found two downstream defects:
1. The adapter test fixture used six distinct random exemplars, so the existing
   assertion that the query predicts A was no longer deterministic.
2. Requiring browser compatibility metadata at the TypeScript payload type level
   made the existing local-demo payload fail that type contract.

Per the authoritative restart protocol, iteration 12 is failed and its
downstream evidence is invalidated. Iteration 13 restarts from Step 1.


## 2026-09-24 — Protocol iteration 13 failure

Iteration 13 repaired the deterministic adapter fixture and restored optional
TypeScript metadata fields while keeping strict runtime compatibility checks.

A final verification sweep found stale deployment-layout wording in
frontend/vite.config.ts: the comment still described an older Render publish
layout even after staticPublishPath was changed to dist.

Commit 75992bcad7775cb0c4d3e657978d3d5b27ea7c55 corrected that residue.

Per the authoritative restart protocol, iteration 13 is failed and all
downstream evidence is invalidated. Iteration 14 restarts from Step 1.


## 2026-09-24 — Protocol iteration 14 failure

A real persistence defect was reproduced: with SQLite foreign-key enforcement
enabled, deleting an adapter that still had TranslationLog rows referencing it
raised an IntegrityError.

Commit 8fdc265f40651b39a179b84ef43f0cd8763e8420 fixes deletion by clearing the
adapter reference while preserving the historical log row. A regression test
was added.

The first version of that regression test used a fake weights path outside the
managed adapter directory, which correctly triggered the path-safety guard.
Commit 6f987c20f5af1512a37fefd01a1e2a64e565b984 corrected the fixture.

Iteration 14 is failed under the authoritative restart protocol and its
downstream evidence is invalidated. Iteration 15 restarts from Step 1.


## 2026-09-24 — Protocol iteration 15 failure

Iteration 15 reproduced a real persistence inconsistency in calibration:
the adapter file was written before the database transaction committed, so a
database failure could leave an orphaned adapter JSON.

Commit 2c9dcbef59c50867a63d6cc7b90bc2ef261bffd5 adds rollback cleanup and
commit 3f59709b11eb5c07302ab5b2dc6624490a8338eb adds regression coverage.

The fix was source-verified. Per the authoritative restart protocol, iteration
15 is failed and downstream evidence is invalidated. Iteration 16 restarts
from Step 1.


## 2026-09-24 — Protocol iteration 16 failure

The lifecycle audit reproduced an unhandled promise rejection when the MediaPipe
tracker's asynchronous send() operation failed.

Commit b07a171638869a4960db6f43fe607b43ed1478ae handles tracker errors by
stopping recognition, releasing camera tracks, closing the tracker, and leaving
the UI in a stopped state. A local rejection-path simulation passed.

Per the authoritative restart protocol, iteration 16 is failed and its
downstream evidence is invalidated. Iteration 17 restarts from Step 1.


## 2026-09-24 — Protocol iteration 17 failure

Iteration 17 found a retired frontend route: /evaluation silently redirected to
the dashboard after the old evaluation screen had been removed. Commit
f6829ce0a0ed175f4cd4a2e9afbbd1036ce9b0f6 removed the dead route.

Source search confirmed no remaining frontend /evaluation route references.

Per the authoritative restart protocol, iteration 17 is failed and its
downstream evidence is invalidated. Iteration 18 restarts from Step 1.


## 2026-09-24 — Protocol iteration 18 failure

The camera lifecycle audit found a race in which an in-flight MediaPipe result
could arrive after stop() and repopulate the latest frame, allowing the separate
recognition timer to keep processing despite the stopped camera state.

Commit 0f6bcfecf1cfa2498c5ab3f87a7c17b650184f62 adds start-generation and active
state guards to the result callback and stale error path.

The repaired guards were source-verified. Per the authoritative restart
protocol, iteration 18 is failed and downstream evidence is invalidated.
Iteration 19 restarts from Step 1.


## 2026-09-24 — Protocol iteration 19 failure

The account regression tests introduced during the adapter persistence repair
used dynamic __import__ expressions where explicit imports were available.

Commit 05f2bfdb036eba741ad669e69a1b22ed0caa19d9 replaced them with ordinary
explicit imports. Source verification confirms no __import__ remains in the
test file.

Per the authoritative restart protocol, iteration 19 is failed and downstream
evidence is invalidated. Iteration 20 restarts from Step 1.


## 2026-09-24 — Protocol iteration 20 failure

The adapter deletion audit found a transaction boundary defect: the history
nulling update ran outside the rollback/restore block, so a database failure at
that point could strand the adapter file as a tombstone while leaving the
database record intact.

Commit 932d86c51c35c646dbe6c0df0e4eb92ca4bde1f1 encloses the history update,
adapter delete, and commit in the protected block. Commit
a0c19d6837dd8d907e801eee585c6fc4c1262a25 adds regression coverage.

Per the authoritative restart protocol, iteration 20 is failed and downstream
evidence is invalidated. Iteration 21 restarts from Step 1.


## 2026-09-24 — Protocol iteration 21 clean pass

Iteration 21 restarted from Step 1 after the iteration 20 transaction defect.
The complete corrected repository was re-audited before accepting any
downstream evidence.

Static verification:
- retired architecture references absent
- no placeholder TODO/FIXME/NotImplementedError code
- no retired /evaluation route
- no legacy MediaPipe Hands references
- Render publishes frontend/dist and health-checks /api/v1/health
- current main branch has no active CI workflow/statuses

Targeted runtime verification:
- strict checkpoint migration and load: PASS
- 126 -> 128 -> 64 -> 26 model contract: PASS
- migrated learned tensors unchanged: PASS
- browser/Python numerical inference parity: PASS
- 126D normalization parity: PASS
- adapter deletion with retained prediction history: PASS
- calibration database-failure cleanup: PASS
- tracker failure recovery and stale callback rejection: PASS

Current release boundary:
- repository base checkpoint installation: PENDING
- uploaded checkpoint accuracy: NOT VERIFIED
- signer-independent held-out evaluation: BLOCKED pending verified signer metadata
- browser real-model device flow: NOT VERIFIED
- live Render deployment: NOT VERIFIED
- durable production database: NOT IMPLEMENTED
- real-mode auth token hardening: PENDING

Iteration 21 result:
    CLEAN REPAIR PASS
    no restart-triggering defect discovered
    downstream model-quality and deployment claims remain explicitly
    unverified or blocked.

## 2026-09-24 — Protocol iterations 22–23 — Stitch frontend integration

The uploaded Stitch ZIP was inspected as the design source. The active UI now
uses the ZIP's monochrome editorial/technical visual language while preserving
the real VisionBridge flows: live camera recognition, few-shot signer
calibration, letter history, signer profiles, and local/real runtime behavior.

Commits:
- 82061f6f2ca667150ef4ef9917a4ad2e75280235
- cd31610e8fa2175ad56d04fed5aa987fafdf4d3f
- 1c791853de886a956b4a4ab567d54ecea066d926

A genuine integration defect was found after the first frontend styling pass:
the font @import was appended after existing CSS rules. This was corrected by
moving the import to the top of the stylesheet, and the protocol required a
full restart before acceptance.

A separate camera-permission warning raised by an initial audit probe was
investigated and proven to be a false positive because the header was on the
backend API response rather than the static frontend document. The temporary
change was reverted completely.

Clean source-level checks after restart:
- no retired architecture files
- active routes remain dashboard, translate, calibration, history, settings
- no retired /evaluation route
- Stitch live-recognition and calibration markers are present
- no copied demo metrics such as 98.4%/92.4% or fictitious device telemetry
- @mediapipe/tasks-vision remains locked to 0.10.35
- package.json and package-lock dependency versions match
- strict model-loader input/vocabulary checks remain present
- frontend App.tsx and stylesheet delimiter balance checks pass
- stylesheet font import is first rule
- no current CI workflow is present

Runtime verification remains blocked by the environment/network boundary, so
no Vite build or live browser result is claimed.


## 2026-09-24 — Protocol iterations 24–26 — Stitch integration repair and verification

Scope:
- Integrate the uploaded Stitch VisionBridge design into the existing frontend without modifying backend/model behavior.
- Preserve the existing live recognition, few-shot calibration, history, signer profile, local-runtime, and real-runtime flows.

Iteration 24:
- FAILED at frontend source audit.
- The active `frontend/src/App.tsx` contained duplicated declarations:
  `function Authfunction Auth`, `function Dashboard()function Dashboard()`,
  `const LETTERS =const LETTERS`, and `function History()function History()`.
- Root cause: malformed source edits in the active frontend file.
- Correction: restored the intended single declarations.
- Isolated verification: duplicate markers absent and delimiter balance is zero.

Iteration 25:
- FAILED at CI type-check.
- Error: local `BrowserAdapterPayload` construction in `frontend/src/api.ts`
  omitted required `preprocessing_version` and `landmark_runtime` metadata.
- Root cause: the local demo adapter path had drifted from the browser model contract.
- Correction: supplied the canonical preprocessing and MediaPipe runtime identifiers already enforced by `browserModel.ts`.
- Downstream build evidence from the failed run was invalidated.

Iteration 26:
- PASS.
- `npm ci`: PASS.
- `npm run check`: PASS.
- `npm run build`: PASS.
- Vite production preview startup and HTTP response check: PASS.
- Source audit after restart: PASS.
- Render configuration remains static frontend -> `dist`, with `VITE_LOCAL_MODE=true`; backend/model behavior was not changed.
- The Stitch visual system remains sourced from the existing integrated stylesheet: monochrome editorial layout, Newsreader/Space Mono typography, telemetry strips, camera workstation, classification panel, word buffer, temporal log, and A–Z calibration matrix.
- Temporary/branch verification is isolated in `.github/workflows/stitch-frontend-check.yml` so the main deployment path is not altered.

Final verified change set relative to main:
- `frontend/src/App.tsx`: repaired malformed duplicated declarations.
- `frontend/src/api.ts`: restored local adapter metadata contract.
- `.github/workflows/stitch-frontend-check.yml`: added frontend type-check, production build, and preview-start verification.

Remaining release boundary:
- This branch is not a live Render deployment of the new commit.
- Browser camera/model behavior and production Render E2E remain separate verification stages.
- No claim is made that backend model quality has changed.


## 2026-09-24 — Pipeline scope correction

The GitHub Actions verification workflow added during frontend repair was removed after the repository deployment requirement was clarified: VisionBridge must not introduce a GitHub CI/CD pipeline.

Verification remains documented from the already completed local/hosted check, but no workflow is retained in the repository.


## 2026-09-24 — Protocol iteration 27 — Post-pipeline execution audit

Step 1 — Inventory:
- active branch: `feat/stitch-frontend-integration-v2`
- 66 repository files visible from the branch tree
- 18 frontend files
- no `.github/workflows` files
- no retired `/evaluation` route/path
- no legacy MediaPipe Hands path detected
- Render configuration and frontend lockfile are present

Step 2 — Reproduction:
- no new functional failure reproduced in the active source.
- The previously repaired frontend source and adapter-contract defects remain absent.

Step 3 — Isolation:
- `frontend/src/App.tsx` declaration markers are unique.
- App delimiter counts are balanced.
- active routes remain dashboard, translate, calibration, history, and settings.
- Stitch recognition/calibration UI markers remain present.
- `@mediapipe/tasks-vision` remains pinned to 0.10.35.
- browser preprocessing/runtime identifiers remain aligned with the active contract.
- package.json and package-lock root dependency declarations match.
- Render still publishes `frontend/dist` and keeps `VITE_LOCAL_MODE=true`.

Step 4 — Fix:
- no functional fix was required in iteration 27.
- The previously added GitHub Actions workflow remains removed as required.

Step 5 — Invalidation:
- no new defect was found, so no downstream evidence required invalidation.

Step 6 — Restart:
- full source audit was repeated from Step 1 after the pipeline removal.

Step 7 — Verification:
- repository tree audit: PASS
- no GitHub workflow pipeline: PASS
- source contract audit: PASS
- deployment configuration audit: PASS
- prior hosted type-check/build/preview evidence for the unchanged functional code: PASS
- direct live Render URL probing from this environment: BLOCKED by the web access boundary; no live deployment claim is made.

Iteration 27 result:
    CLEAN AUDIT PASS
    no restart-triggering defect discovered
    no GitHub CI/CD pipeline retained
    live Render E2E remains explicitly unverified from this environment.


## 2026-09-24 — Protocol iteration 28 — Metadata consistency failure

STEP 1 — INVENTORY:
- active branch: `feat/stitch-frontend-integration-v2`
- current PR #6 is open against `main`
- repository deployment requirement remains: no GitHub Actions / CI/CD pipeline
- functional source and Render configuration are unchanged

STEP 2 — REPRODUCTION:
- PR #6 metadata still claimed that frontend CI verification had been added.
- This contradicted the repository's current state because the temporary workflow had already been removed.

STEP 3 — ISOLATION:
- repository tree audit: no `.github/workflows`
- source audit: previously repaired frontend contracts remain intact
- Render audit: static frontend publishes `dist`; `VITE_LOCAL_MODE=true`
- the inconsistency was limited to PR metadata, not application behavior

STEP 4 — FIX:
- corrected PR #6 description to remove the stale CI claim and explicitly state that GitHub Actions / CI/CD is not introduced.

STEP 5 — INVALIDATION:
- no functional or deployment evidence was invalidated because application source and Render configuration were not changed.
- PR documentation claims were corrected before this restart could be accepted.

STEP 6 — RESTART:
- restart from Step 1 required by the execution protocol after the discovered inconsistency.

## 2026-09-24 — Protocol iteration 29 — Post-metadata restart audit

STEP 1 — INVENTORY:
- active branch: `feat/stitch-frontend-integration-v2`
- branch is 9 commits ahead of `main`, 0 behind
- PR #6 remains open and mergeable
- 66 repository files visible from the branch tree
- 18 frontend files
- no `.github/workflows` files
- no retired `/evaluation` route/path detected
- no legacy MediaPipe Hands path detected
- frontend lockfile and Render configuration are present

STEP 2 — REPRODUCTION:
- no new functional failure reproduced from the repository source audit
- previously repaired App.tsx declaration duplication and adapter metadata drift remain absent

STEP 3 — ISOLATION:
- active frontend routes remain dashboard, translate, calibration, history, and settings
- `@mediapipe/tasks-vision` remains pinned to 0.10.35
- browser preprocessing/runtime identifiers remain aligned with the active contract
- package.json and package-lock root dependency declarations match
- Render still publishes `frontend/dist` and keeps `VITE_LOCAL_MODE=true`
- searches for TODO/FIXME/NotImplementedError and retired frontend markers returned no matches
- PR #6 description now matches the actual no-CI repository state

STEP 4 — FIX:
- no application-code fix required in iteration 29
- metadata correction from iteration 28 is retained

STEP 5 — INVALIDATION:
- no new defect discovered; no downstream functional evidence invalidated

STEP 6 — RESTART:
- full source/deployment/metadata audit repeated from Step 1 after the iteration 28 correction

STEP 7 — VERIFICATION:
- repository tree audit: PASS
- no GitHub workflow pipeline: PASS
- source contract audit: PASS
- dependency manifest/lockfile audit: PASS
- Render configuration audit: PASS
- PR metadata consistency audit: PASS
- prior npm type-check/build/preview evidence for unchanged functional code remains applicable
- direct live Render probing from this environment: BLOCKED by web access boundary; no live E2E claim is made

Iteration 29 result:
    CLEAN PASS
    no restart-triggering defect discovered
    no GitHub CI/CD pipeline retained
    PR metadata is consistent with repository scope
    live Render E2E remains explicitly unverified


## 2026-09-24 — Protocol iteration 30 — Ledger arithmetic correction and restart

A documentation arithmetic error was discovered during post-write verification: iteration 29 recorded the branch as 8 commits ahead of `main`, while the live GitHub compare result shows 9 commits ahead and 0 behind.

CORRECTION:
- corrected the recorded ahead count from 8 to 9.

RESTART:
- the protocol was restarted from Step 1 after correcting the ledger discrepancy.

FINAL VERIFICATION:
- GitHub compare: 9 ahead / 0 behind
- current head: `e8ea916fb8ba57dc0f7752623b698e2c5c0b882a`
- PR #6 remains open
- no GitHub Actions workflow exists
- no `/evaluation` route/path detected
- no legacy MediaPipe Hands path detected
- Render configuration remains unchanged
- live Render E2E remains unverified from this environment

Iteration 30 result:
    CLEAN LEDGER PASS
    the recorded 9-ahead comparison was accurate at the verification point;
    the ledger correction commit is itself an additional branch commit.
    Future protocol entries must treat branch-ahead counts as point-in-time evidence,
    not as a value that remains constant after documentation commits.

## 2026-09-24 — Protocol iteration 31 — Colab notebook V3 architecture execution hardening

STEP 1 — INVENTORY:
- current source of truth: `main`
- active base-model contract remains V3:
  126D normalized two-hand landmarks -> LayerNorm -> Linear(128) -> GELU -> Dropout(0.10)
  -> Linear(64) -> LayerNorm -> GELU -> Linear(26)
- training notebook: `notebooks/train_letter_base_colab.ipynb`
- no trained production checkpoint is currently committed under `backend/app/models/weights/`
- no GitHub Actions workflow is present

STEP 2 — REPRODUCTION:
- environment-level repository execution was attempted with a shallow Git clone.
- reproduction failed before source execution because the container could not resolve `github.com`:
  `fatal: unable to access 'https://github.com/BharathWaj-K-R/VisionBridge.git/': Could not resolve host: github.com
  `
- therefore local pytest/build execution could not be performed from the container in this iteration.
- GitHub-backed source inspection remained available and was used for contract verification.

STEP 3 — ISOLATION:
- notebook source inspection found two quality defects:
  1. duplicated `nvidia-smi` capability probing in the environment cell
  2. stale Colab metadata name `train_letter_base_colab_fixed.ipynb`
- the active model source and training CLI dimensions were already aligned with V3; no architecture mismatch was reproduced.
- the notebook trained with explicit `--hidden-dim 128`, `--embedding-dim 64`, `--dropout 0.10` and the active 126D/A-Z dataset contract.

STEP 4 — FIX:
- removed the duplicated GPU capability probe.
- renamed Colab metadata to `train_letter_base_colab.ipynb`.
- added an executable V3 architecture-contract cell immediately before training.
- the new check asserts:
  - input_dim = 126
  - hidden_dim = 128
  - embedding_dim = 64
  - num_classes = 26
  - A-Z label vocabulary
  - encoder layer order: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> LayerNorm -> GELU
  - embedding output shape [2, 64]
  - logits output shape [2, 26]
  - finite outputs

STEP 5 — INVALIDATION:
- prior notebook-only claims are superseded by the modified notebook.
- previously established model/source contracts remain applicable because backend model, preprocessing, and evaluation code were not changed.
- no trained-checkpoint accuracy evidence exists and none is claimed.

STEP 6 — RESTART:
- restarted verification from Step 1 after the notebook write.

STEP 7 — VERIFICATION:
- remote notebook fetch after commit: PASS
- notebook JSON parse: PASS
- notebook cell count: 12
- exactly one architecture-contract cell: PASS
- exactly one GPU probe invocation: PASS
- architecture assertions 126/128/64/26: PASS
- encoder sequence assertion: PASS
- output-shape assertions: PASS
- training flags remain hidden=128, embedding=64, dropout=0.10: PASS
- untouched test evaluation cell remains present: PASS
- current production checkpoint still absent: PASS
- commit: `383c5d76f351ca33a555c293017776e711aefdf2`

Iteration 31 result:
    PASS for source/notebook contract hardening
    runtime execution remains BLOCKED by container DNS/network access
    real RealSign training and held-out accuracy remain outstanding



---

# 2026-09-25 — V3 release-hardening restart

ITERATION: 32
RESULT: FAILED
FAILED STEP: Step 1 — inventory / contract audit

FINDING:
    Browser V3 inference accepted a model payload without requiring the active
    V3 model_version identifier. A future same-shaped checkpoint could therefore
    reach the V3 browser implementation without an explicit version guard.

ROOT CAUSE:
    Browser payload validation checked input/output dimensions and preprocessing
    metadata but did not bind the payload to the active V3 model version.

FIX:
    frontend/src/browserModel.ts now requires:
        model_version == visionbridge-letter-base-v3
        model_sha256 is present
        preprocessing_version == two-hand-wrist-scale-v1
        landmark_runtime == mediapipe-hand-landmarker-0.10.35

VERIFICATION:
    Corrected source contains the exact V3 version guard and checksum guard.
    Comparison against the pre-fix head shows only frontend/src/browserModel.ts
    changed in this iteration.

DOWNSTREAM RESULTS INVALIDATED:
    No real-data model-quality evidence existed.

RESTART REQUIRED: YES


# 2026-09-25 — V3 release-evidence restart

ITERATION: 33
RESULT: FAILED
FAILED STEP: Step 1 — inventory / release-evidence audit

FINDING:
    The canonical evaluator did not emit the checkpoint SHA-256 required by the
    model release evidence contract, and the Colab notebook did not generate a
    final machine-readable V3 release-evidence manifest.

ROOT CAUSE:
    Checkpoint integrity was validated at load time, but the evaluator/report
    contract did not persist the content hash. The notebook ended after test
    evaluation without consolidating the training configuration, checkpoint
    hash, metrics, and signer-gate status.

FIX:
    backend/app/training/evaluate_letter_base.py now reports checkpoint_sha256.
    notebooks/train_letter_base_colab.ipynb now generates
    /content/visionbridge_v3_release_evidence.json containing:
        model contract
        training configuration
        checkpoint bytes and SHA-256
        held-out test evaluation
        signer-independent status

VERIFICATION:
    Evaluator source contains the SHA-256 helper, report field, and printed hash.
    Notebook remains valid JSON with 12 cells; its final cell is executable and
    writes the release-evidence manifest.

DOWNSTREAM RESULTS INVALIDATED:
    No real-data model-quality evidence existed.

RESTART REQUIRED: YES


# 2026-09-25 — V3 corrected-state full audit

ITERATION: 34
RESULT: BLOCKED, SOURCE CLEAN

STEP 1:
    Current main head: 1832f3559369c1e89bb21cd3d11709b3af57b9ec
    Tracked tree inventory: 86 files
    Python files: 35
    TypeScript/TSX files: 9
    Training notebook: present
    GitHub Actions workflows: none
    V3 checkpoint: absent
    V3 critical-path files: present

STEP 2:
    Known V3 release blocker reproduced from repository state:
        backend/app/models/weights/letter_base_model.pt is not present.
    The repository therefore cannot truthfully claim a trained V3 checkpoint,
    held-out test accuracy, or production-ready real-model inference.

STEP 3:
    The missing checkpoint is an execution-state blocker rather than a source
    implementation failure. The training notebook creates it only after a real
    RealSign training run.

STEP 4:
    Static data-flow audit remains:
        RealSign images
        -> MediaPipe Tasks Hand Landmarker 0.10.35
        -> raw Tasks left/right handedness
        -> wrist-relative + scale normalization
        -> 126D vector
        -> VisionBridgeLetterBaseModel
        -> 64D embedding
        -> few-shot signer adapter
        -> A-Z + confidence

STEP 5/6:
    V3 implementation is present, versioned, checkpoint-validated, and browser
    serialized. No additional code defect was discovered in the corrected-state
    audit after the two release-hardening fixes.

STEP 7:
    Source-level checks:
        V3 architecture contract: present in notebook
        A-Z/126D contract: present
        checkpoint strict loading: present
        adapter checksum binding: present
        browser V3 version binding: present
        evaluator checkpoint SHA-256: present
        current GitHub Actions: none

    Runtime checks:
        Real-data training: NOT VERIFIED in this environment
        Held-out V3 accuracy: NOT VERIFIED
        Held-out signer evaluation: BLOCKED pending verified signer metadata
        Browser real-model inference: NOT VERIFIED
        Render real-model verification: NOT VERIFIED

    The local environment could not clone the public repository because direct
    outbound GitHub DNS/network access is unavailable. Repository source was
    therefore audited through the connected GitHub interface rather than a local
    runtime checkout.

STEP 8:
    No cleanup beyond the targeted V3 release-hardening changes was performed.

STEP 9:
    This entry records the corrected source state and the remaining external
    gates. No accuracy number is claimed.

FINAL STATUS:
    V3 SOURCE IMPLEMENTATION: COMPLETE
    V3 RELEASE ARTIFACT: BLOCKED — real trained checkpoint does not yet exist
    V3 HELD-OUT ACCURACY: NOT VERIFIED
    SIGNER-INDEPENDENT RELEASE GATE: BLOCKED
    FULL PROJECT COMPLETION: NOT COMPLETE

RESTART REQUIRED:
    YES, after the real-data training/checkpoint step produces a new evidence
    state. The full protocol must then be rerun from Step 1.
