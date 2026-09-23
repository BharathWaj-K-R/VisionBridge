# VisionBridge Model Debug Log

This file records model-specific failures, fixes, experiments, validation evidence, and unresolved ML blockers for the active letter-recognition architecture.

ACTIVE PIPELINE

    MediaPipe Hands
     -> normalized 126D two-hand landmarks
     -> frozen VisionBridgeLetterBaseModel
     -> 64D embedding
     -> few-shot signer adapter
     -> A-Z letter + confidence

This is the only active ML architecture described here.

---

# 1. Active model contract

    Input:          126 normalized landmark values
    Embedding:      64 dimensions
    Output classes: 26 (A-Z)
    Loss:           CrossEntropyLoss
    Base model:     VisionBridgeLetterBaseModel
    Adapter:        frozen-base-embedding-prototype

Expected checkpoint:

    backend/app/models/weights/letter_base_model.pt

The base model is trained once and then frozen.

The few-shot adapter is fitted from signer calibration examples. It does not require a separate offline training job.

---

# 2. Landmark and normalization contract

Each sample contains:

    left hand:  63 values
    right hand: 63 values
    total:     126 values

Preprocessing:

    MediaPipe Hands
     -> handedness-aware left/right placement
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

    LayerNorm(126)
     -> Linear(126 -> 128)
     -> GELU
     -> Dropout(0.10)
     -> Linear(128 -> 64)
     -> LayerNorm(64)
     -> GELU
     -> Linear(64 -> 26)

The model must expose a 64D embedding and 26-class logits.

The base model must remain frozen during signer adaptation.

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
    embedding size is 64
    logits have shape [batch, 26]
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
     -> MediaPipe Hands
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

# 10. Current training state

Known source status:

    Base model code:                 CI VERIFIED
    Dataset preparation code:        CI VERIFIED
    Training CLI:                    CI VERIFIED
    Colab notebook:                  STATIC VERIFIED
    Few-shot adapter code:           CI VERIFIED
    Checkpoint compatibility:        CI VERIFIED
    Real base checkpoint:            NOT VERIFIED
    Base held-out accuracy:          NOT VERIFIED
    Signer held-out accuracy:        NOT VERIFIED
    Browser real-model inference:    NOT VERIFIED

Known successful code verification:

    GitHub Actions run #155
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
      --epochs 30
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
    input_dim = 126
    embedding_dim = 64
    num_classes = 26
    labels = A-Z
    state_dict loads strictly
    test result is recorded

The runtime must reject missing, malformed, incompatible, or stale checkpoints.

The adapter must remain bound to the exact base checkpoint used to create its prototypes.

No silent fallback to an incompatible model is permitted in real mode.

---

# 14. Current loose ends

    BASE MODEL TRAINING              NOT VERIFIED
    BASE HELD-OUT TEST               NOT VERIFIED
    FEW-SHOT HELD-OUT SIGNER TEST    NOT VERIFIED
    LIVE CAMERA REAL MODE            NOT VERIFIED
    RENDER REAL MODE                 NOT VERIFIED

There is no second hidden offline ML training task.

Intended lifecycle:

    train base once
     -> freeze base
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

The active model source of truth remains:

    126D hands
     -> frozen base model
     -> 64D embedding
     -> few-shot signer adapter
     -> letter
