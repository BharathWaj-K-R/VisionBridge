# VisionBridge Accuracy Improvement Architecture Plan

**Status:** Proposed; no model or application code changes are included in this plan.  
**Last reviewed:** 2026-10-27  
**Current production baseline:** V3, preserved until a separately versioned candidate meets the acceptance gates.

## 1. Executive recommendation

Do not start by making the MLP larger or retraining against the current test set. The most important unresolved accuracy risk is generalization: the recorded prepared-image test report reaches 98.2733% overall and 98.2913% macro class accuracy, but W reaches only 78.7710%. The recorded validation W score is much higher (about 98.47%), so the validation/test gap deserves investigation; it does not, by itself, identify the cause.

The active source metadata does not contain verified signer identifiers. Therefore the current report cannot establish signer-independent generalization. Improving the score on the same image split is useful for engineering regressions, but it is not enough to validate real users.

Recommended order:

1. Freeze the V3 baseline and build trustworthy signer/session-level evaluation.
2. Improve dataset coverage and audit landmark extraction/feature conventions.
3. Run controlled feature and lightweight model ablations.
4. Add a temporal path only for letters that demonstrably require movement.
5. Make confidence, abstention, smoothing, and production monitoring part of the recognition system.
6. Promote only a versioned candidate that beats V3 on unseen signers and passes regression gates.

## 2. Current architecture and evidence

### Current inference path

```text
Camera frame
  -> MediaPipe Tasks Hand Landmarker 0.10.35
  -> left-hand landmarks (21 x XYZ) + right-hand landmarks (21 x XYZ)
  -> independently wrist-center and scale each hand
  -> concatenate 126 values
  -> V3 MLP: LayerNorm -> Linear(128) -> GELU -> Linear(64) -> LayerNorm -> GELU
  -> 26 A-Z logits and a 64D embedding
  -> optional normalized prototype adapter using cosine similarity
  -> confidence/stability gate
  -> letter buffer and optional history
```

The base classifier has 26,582 parameters. Browser inference avoids a backend round trip on each camera frame; the backend handles model/adapter delivery, accounts, calibration persistence, and history.

### Accuracy symptoms worth addressing

- Current recorded test: 4,633 prepared samples, 98.2733% overall accuracy, 98.2913% macro class accuracy.
- W is weakest at 78.7710%. The saved confusion matrix has W commonly predicted as M, N, and O.
- Prepared sample counts vary by letter (137–200) because examples that do not yield usable landmarks are skipped. Classification accuracy is therefore conditional on successful preparation; it does not count every source-image/hand-detection failure.
- Existing training augmentations include coordinate jitter and in-plane rotation, and the optimization workflow focuses on W/M. More loss-function tuning alone is unlikely to address unknown signer overlap, camera domain shift, missed detections, or lost inter-hand geometry.
- Current parity CI tests the numerical forward pass on generated vectors. It does not fully validate raw-frame -> handedness assignment -> normalization -> prediction parity across the Python and browser pipelines.
- The current input is frame-based. A stable-frame gate can reduce flicker, but it does not itself learn motion-dependent letters or sentence-level signing.

These are reasons to prioritize evaluation and feature consistency before increasing model complexity.

## 3. Target architecture: V4, versioned and side-by-side

Keep V3 deployable as the fallback. Build V4 behind a new model/preprocessing contract; do not mutate V3's 126D meaning or silently reuse existing calibration adapters.

### 3.1 Versioned preprocessing and quality gate

Input contract proposal:

- Run the same pinned MediaPipe Tasks version for dataset preparation and browser inference.
- Make the handedness convention explicit, including whether the camera image is mirrored. Test both mirrored and unmirrored frames; do not add a one-sided left/right swap to solve a local visual issue.
- Record whether left and right hands are present, the handedness score, and any usable landmark-quality signal before discarding that information.
- Preserve the present 63D normalized local coordinates per hand as a baseline feature.
- Add a separately versioned set of **relative two-hand features** when both hands are visible, including the wrist-to-wrist displacement/distance in a consistently defined coordinate system. Independent wrist-centering currently removes absolute wrist positions and therefore removes some between-hand spatial information.
- Add derived geometry, such as bone directions and joint-angle features, only through controlled ablations. Do not replace the existing coordinate representation until experiments prove the new representation is better.
- Represent a missing hand explicitly with a presence mask in V4. Keep zero-filled slots for compatibility if needed, but distinguish “not detected” from an actual low-quality/partial observation.

The purpose is not to make landmarks invariant to everything. It is to preserve useful sign geometry while reducing irrelevant camera scale/position variation. Rotation, mirroring, and hand swapping can change the label or hand role, so each transformation must be justified and tested against ISL semantics.

### 3.2 Lightweight bilateral encoder

First V4 candidate:

```text
Normalized left hand (63D)  -> shared hand encoder -> left embedding  ┐
Normalized right hand (63D) -> shared hand encoder -> right embedding ├
Relative hand features + left/right presence/quality masks             ┘
                                  |
                            fusion MLP
                                  |
                   calibrated A-Z classification head
                                  |
                letter / confidence / uncertainty / abstain
```

Use a shared per-hand encoder as an **experiment**, not as an assumed win. A side/hand-role signal may be needed because the two hands are not always interchangeable. Compare it against the existing concatenated V3 MLP using identical grouped splits. Prefer a compact MLP/fusion model while the labelled data remains limited; a transformer-sized model would add overfitting risk before the dataset supports it.

### 3.3 Temporal branch for movement-dependent letters

Do not replace the entire classifier with an LSTM or transformer on day one. First verify from ISL references and recordings which labels require movement in the intended product definition.

If frame-only recognition is inadequate for J/Z or other movement-dependent letters, add a distinct sequence path:

```text
T consecutive validated landmark frames
  -> per-frame features and quality mask
  -> compact temporal convolution (TCN) or GRU
  -> sequence-level letter scores
  -> static/temporal route chosen by validation evidence
```

Train and evaluate the temporal branch on **labelled clips**, including start/end or transition boundaries where applicable, not randomly sampled adjacent frames. Split by recording and signer before extracting overlapping windows; otherwise highly similar windows can leak across splits. Keep the static path for letters where it is adequate. The route/gating rule and supported letter set must be explicit.

An ISL continuous-fingerspelling resource can be evaluated as a supplemental research source, subject to licensing and domain checks; it must not be assumed to represent ordinary webcam users or solve the lack of signer diversity by itself.

## 4. Data architecture: highest-priority work

### 4.1 Create a new signer-aware evaluation set

The present dataset metadata cannot support verified signer holdout. Collect or source a consented evaluation set with, at minimum:

- stable pseudonymous signer ID;
- session/recording ID and capture date;
- label and annotation/review status;
- device, frame size, camera angle, distance, lighting/background conditions;
- raw-image/video hash and derived-frame identifier;
- whether each hand was detected and why a sample was rejected;
- the exact MediaPipe runtime and preprocessing version.

A practical pilot is 12 or more signers, multiple sessions and capture conditions, with repeated examples of every label. Expand sample counts after measuring variability. Store identities separately from training artifacts; only pseudonymous IDs should appear in manifests.

Create group-level partitions by signer (and keep each recording together). Run group-based cross-validation on development signers if the dataset is large enough, then retain a separate, untouched set of signers as the final gold test. Report mean and variation across signer folds, not just one pooled accuracy.

### 4.2 Detect leakage and report coverage

- Keep the official source test split out of training and validation, as the current pipeline intends.
- Retain exact SHA-256 checks, then add perceptual-hash/near-duplicate review and recording/source grouping. Exact hashes do not catch resized, compressed, or adjacent video frames.
- Never place frames from the same clip/recording into different partitions.
- Do not repeatedly tune or promote from repeated inspection of the same held-out set. The current experiment workflow uses test results as a promotion gate; keep the existing report as a historical/regression reference and use a **fresh locked signer-held-out set** for new architecture decisions.
- Report both (a) landmark extraction/detection coverage across all source examples and (b) classifier performance conditional on usable landmarks. Keep per-label failure rates visible.
- Review W/M/N/O examples, especially W-to-M/N/O confusions, with human inspection of source images and landmarks. Confirm labels, image orientation, handedness and extraction quality before changing loss weights.

### 4.3 Controlled data augmentation

Use an ablation matrix rather than stacking every augmentation at once. Candidate augmentations:

1. small landmark noise calibrated to observed tracker jitter;
2. mild rotation and anisotropic scaling;
3. bone-length/segment perturbations constrained by plausible hand anatomy;
4. modest depth/tilt perturbations if supported by the observed camera geometry;
5. partial landmark dropout/low-confidence simulation if it represents real detector failures.

The aim is signer/camera robustness, not synthetic corruption. Verify that every transformed sample preserves its label and hand role. Never horizontally flip or swap hands without explicit tests proving that the label and handedness metadata remain correct. Compare each augmentation against a no-augmentation baseline and record per-class effects.

## 5. Training and model selection

Use a reproducible experiment matrix with a fixed seed list and identical grouped splits:

| Candidate | Change | Question answered |
|---|---|---|
| V3 baseline | No architecture change | What is the comparable baseline? |
| V4 geometry | Add hand-pair relation/presence features | Does lost spatial context explain errors? |
| V4 shared encoder | Shared per-hand encoder + fusion | Does structured encoding help without overfitting? |
| V4 geometry + augmentation | Add one realistic augmentation family at a time | Does robustness improve across signers/devices? |
| V4 calibrated head | Temperature scaling/class-wise thresholds | Are confidence scores safe enough to drive commits? |
| Temporal pilot | TCN/GRU for verified movement-dependent classes | Does temporal evidence improve the relevant letters? |

For every candidate, record checkpoint hash, preprocessing version, dataset manifest hash, signer split IDs, seed, hyperparameters, class counts, all metrics and failure examples.

Keep early stopping/model selection on development validation only. Use a validation objective that prioritizes macro-F1/balanced accuracy and worst-class recall, with overall accuracy as a guardrail. Also report per-class precision/recall/F1 and confusion matrices; class accuracy alone can hide asymmetric errors. Do not increase hidden width or train more epochs just because aggregate validation accuracy moves in the third decimal place.

## 6. Inference decision architecture

Model quality and decision quality are different problems. Keep these stages separate:

1. **Input quality:** require a valid frame, plausible landmark geometry and adequate hand presence. If the tracker is unsure, show “adjust hand position” instead of forcing an A–Z guess.
2. **Per-frame score:** return logits and calibrated probabilities. Fit temperature scaling only on validation data.
3. **Abstention:** define a validation-selected confidence/margin rule that can emit “uncertain.” Measure selective risk versus coverage; do not assume softmax confidence is calibrated.
4. **Temporal stability:** aggregate a short rolling window using a majority/weighted vote or EMA and a hysteresis rule. Commit only after a stable candidate duration; require a neutral/release or clear transition rule where needed to avoid repeated-letter spam.
5. **User feedback:** retain the prediction, confidence, quality state, and model version in diagnostic events without retaining raw video by default. Collect explicit, consented examples for error review rather than silently recording camera streams.

Thresholds must be measured on signer-held-out development data. A confidence gate may reduce false commits while lowering coverage; report both numbers.

## 7. Evaluation and release gates

Every report should include:

- overall accuracy, macro accuracy, macro-F1 and balanced accuracy;
- precision/recall/F1 for each A–Z label;
- the complete confusion matrix, with special attention to W/M/N/O;
- landmark detection success/coverage on the complete raw set;
- accuracy conditional on usable landmarks and end-to-end correct-letter rate;
- calibration error/Brier score plus selective-risk/coverage curves if probabilities guide user-visible decisions;
- grouped-signer results with confidence intervals over signers/folds;
- performance by condition (device, lighting, distance, hand visibility) when the sample counts support it;
- browser end-to-end latency and memory on representative hardware, not only MLP-only batch latency.

Proposed V4 acceptance criteria, to be fixed before looking at the final gold test:

1. A fresh signer-held-out evaluation is available and no signer/recording overlaps the training partitions.
2. Macro-F1 and macro accuracy improve over V3 on development signer folds, and W recall improves without a material regression in any other class.
3. Overall performance and detection coverage are reported together; the model cannot “improve” by silently rejecting difficult examples.
4. The locked gold test is evaluated once for the release decision, with all per-class results disclosed.
5. Browser/Python parity covers both model math and **golden raw-landmark fixtures through the full normalization/feature pipeline**.
6. Existing V3 adapter/checkpoint contracts remain valid. V4 uses new version/checksum metadata, and adapter recalibration is explicit.
7. Product CI passes, followed by manual camera/microphone and real-device validation. CI success alone is not release sign-off.

Set numeric improvement thresholds before the gold evaluation and do not relax them after seeing the result. A practical initial proposal is to seek at least a 5 percentage-point improvement in W recall on signer-held-out development folds while keeping macro metrics non-inferior; treat this as a target to ratify before the experiment, not as a promised outcome.

## 8. File-level implementation sequence

### Phase 0 — Baseline and guardrails
- Freeze V3 checkpoint, evaluation JSON, source archive hashes and data manifests.
- Create a comparison script that produces the same class/coverage metrics for every candidate.
- Add a new untouched signer-held-out evaluation set before architecture tuning.
- Preserve all existing production behavior and keep the accuracy-optimization workflow manual.

### Phase 1 — Data audit
- Extend preparation manifests with source/recording group IDs, extraction status, hand presence and confidence where available.
- Add perceptual near-duplicate detection and class-by-condition summaries.
- Build a human-reviewed W/M/N/O error gallery.
- Record the rate and class distribution of failures before landmark extraction.

### Phase 2 — V4 preprocessing and model experiments
- Define a new `preprocessing_version` and contract for relational/presence features.
- Add Python feature unit tests plus saved golden inputs/outputs.
- Implement a small shared-hand encoder + bilateral fusion candidate beside V3.
- Run grouped ablations and report results; do not overwrite the V3 checkpoint.

### Phase 3 — Robustness and temporal pilot
- Add plausible geometric augmentation one family at a time.
- Capture labelled video clips for movement-dependent letters.
- Train a compact TCN/GRU prototype only if the evaluation demonstrates that static frames are insufficient.
- Compare static V3, static V4 and the temporal branch on the same signer-held-out folds.

### Phase 4 — Decision calibration and browser parity
- Calibrate confidence and choose abstention thresholds on validation signers.
- Add stable-window/re-arm behavior with explicit tests for repeated letters.
- Add parity fixtures that compare offline and browser feature extraction as well as model output.
- Run end-to-end tests for one hand, two hands, a missing hand, poor-quality frames, mirrored preview, and rapid letter changes.

### Phase 5 — Shadow evaluation and promotion
- Run V4 in a non-user-facing/shadow path against V3 on consented test clips.
- Review class-level wins and regressions; inspect the errors before changing hyperparameters.
- Promote only after the predeclared signer-held-out gate passes and CI/manual device checks pass.
- Store V4 as a separate checkpoint. Keep V3 rollback-ready and require adapter recalibration for the V4 embedding space.

## 9. Research starting points

These sources support the evaluation and experimentation approach; they do **not** prove that a proposed architecture will improve VisionBridge:

- Microsoft Research, [On the Importance of Signer Overlap for Sign Language Detection](https://www.microsoft.com/en-us/research/publication/on-the-importance-of-signer-overlap-for-sign-language-detection/): discusses over-optimistic results when signers overlap between splits.
- ACL Anthology, [Signer Diversity-driven Data Augmentation for Signer-Independent Sign Language Translation](https://aclanthology.org/2024.findings-naacl.140/): research direction for signer diversity and robustness; its task and modality differ from VisionBridge's static A–Z task.
- [Continuous Fingerspelling Dataset for Indian Sign Language](https://kirandevraj.github.io/ISL-Fingerspelling/): potentially relevant to a future temporal branch; inspect license, signer diversity, annotation protocol and domain fit before integrating.
- [Amrita SLR — Real Time ISL Dataset](https://data.mendeley.com/datasets/dtg9s785v8/1): a possible supplementary video resource for realistic acquisition conditions; confirm label coverage and usage terms before use.
- [Anatomy-aware 3D keypoint augmentation for BISINDO alphabet recognition](https://journal.unnes.ac.id/journals/sji/article/view/42221): a recent feature-space augmentation idea to test carefully. BISINDO is a different sign language, so transferability must be demonstrated rather than assumed.

## 10. Non-goals for the first iteration

- Do not increase the model size without a grouped-split ablation.
- Do not tune repeatedly against the current test report.
- Do not claim signer-independent accuracy without verified signer IDs and a group-level split.
- Do not treat the 98.27% prepared-image result as live camera accuracy.
- Do not replace the current model with a temporal transformer before collecting and validating sequence data.
- Do not market the speech/sentence prototype as semantic sentence-level ISL translation.
- Do not overwrite the current checkpoint, change V3's feature meaning, or silently invalidate stored user adapters.

**Recommended first implementation:** Phase 0 + Phase 1. Better data evidence is the prerequisite for knowing whether the next change should be the feature contract, the classifier, the tracker, or the decision gate.
