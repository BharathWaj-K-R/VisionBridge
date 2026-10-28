# Augmentation experiment protocol

**Status: NOT RUN on real ISL data.** The data and Phase 2 winner checkpoint are not available in this execution environment. Do not interpret structural tests or results from other sign languages as ISL improvement evidence.

## Implemented independent augmentations

Each augmentation operates on raw 126D MediaPipe coordinates before wrist centering and scale normalization. One named augmentation is allowed per standalone run. Mirroring is excluded. Random single-hand dropout is skipped because the available source does not establish a verified ISL per-letter hand-configuration table, and dropout could remove label-defining information.

<!-- AUTO_RESULTS_START -->
**Real-data status: NOT RUN.** Phase 2 has no recorded validation-selected winner and no aligned raw landmarks are available in this environment. Run separate single-augmentation experiments before treating this table as evidence.

| Augmentation | Validation macro mean ± std | Macro delta vs baseline | Required seed-variance threshold | Keep? | Joint rejection rate | W→M count Δ | W→N count Δ | W→O count Δ | Python model latency |
|---|---|---|---|---|---|---|---|---|---|
| gaussian_jitter | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| in_plane_rotation | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| depth_scale | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| finger_joint_angles | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

### Per-class validation accuracy change (candidate minus unaugmented baseline)

All values below will be measured on validation across paired seeds. Positive values are accuracy gains; negative values are regressions. Test data is not used to tune the augmentation.

| Letter | gaussian_jitter | in_plane_rotation | depth_scale | finger_joint_angles |
|---|---|---|---|---|
<!-- AUTO_RESULTS_END -->

## Anatomical plausibility bounds

The joint-angle check is a conservative geometry screen, not clinical diagnosis and not an ISL hand-shape rule. It checks the 3D internal angle proxy and applies maximum flexion caps:

| Landmark joint group | Maximum flexion cap |
|---|---:|
| Non-thumb PIP | 120° |
| Non-thumb DIP | 80° |
| Thumb IP | 90° |

Bone lengths after joint-angle augmentation must differ from the original corresponding MediaPipe bone lengths by no more than **5%**. The perturbation uses rigid subtree rotation, which should preserve those lengths apart from floating-point error; the 5% bound is a reject gate, not an intended amount of bone deformation. For each augmentation batch, attempted/accepted/rejected counts and rejection rate are emitted. Rejected joint-angle examples use the unchanged raw sample, so labels and training sample counts remain aligned.

ROM values vary by finger, participant, and measurement protocol. These caps are broad plausibility guardrails derived from clinical references, not ISL-specific validation:
- NCBI Bookshelf, *The Musculoskeletal Examination*: DIP digits 2–5 0–80°, PIP digits 2–5 9–120°, thumb IP up to 90° flexion: https://www.ncbi.nlm.nih.gov/books/NBK272/?report=printable
- U.S. Department of Veterans Affairs, *Evaluation of Limitation of Motion of Single or Multiple Digits of the Hand*: finger MCP 0–90°, PIP 0–110°, DIP 0–70/80°: https://www.benefits.va.gov/PREDISCHARGE/DOCS/disexm27.pdf
- Indian population normative study of finger active ROM (390 hands): https://pubmed.ncbi.nlm.nih.gov/39345665/

## Required experiment design

Baseline is the Phase 2 winner with augmentation set to none. Phase 2's documented result is currently NOT RUN, so there is no winner to name and no valid augmentation ranking yet. Do not quietly substitute a test-selected V3 checkpoint or guess a V4 winner.

Once Phase 2 winner selection and raw split-aligned data exist:
1. Run the no-augmentation baseline and each augmentation independently with at least three fixed seeds, the same manifests/splits, model configuration, optimizer/loss, batch size and exact epoch budget.
2. Augment raw coordinates, then normalize using the selected model's original feature contract. Validation remains unaugmented.
3. Select only from validation macro accuracy mean±seed standard deviation. Keep an augmentation only when its gain exceeds the baseline seed standard deviation and the candidate's own uncertainty does not erase that gain. Record per-class validation deltas and the validation W→M/N/O confusion counts.
4. No combined augmentation run until standalone experiments establish the keep set; then allow at most one combined run.
5. Freeze the selected configuration and checkpoint hash before a single final test evaluation. Do not use test confusion to tune augmentation strength or selection.
6. Record measured Python model-only latency; browser latency remains NOT MEASURED unless measured on a named browser/device.

The existing checked-in V3 test confusion matrix is historical baseline context only. Its W row has 11 W→M, 11 W→N and 9 W→O errors. That test split must not be reused to tune or rank these augmentations.


## Reproduction sequence

The current repository has no completed Phase 2 comparison JSON and no aligned raw dataset. The experiment runner therefore stops with an explicit error until a real Phase 2 validation winner and local dataset are supplied. The placeholders below are paths on the training machine; generated arrays, checkpoints, and result JSON must stay outside Git.

### 1. Diagnose temporal data first

    PYTHONPATH=backend python backend/scripts/diagnose_temporal_sequences.py --manifest-root /path/to/dataset-manifests --output-json /path/outside/repo/temporal_diagnostic.json

If the status is NOT TESTABLE, do not train a temporal head. The present schema has recording_id but no frame_index; see docs/experiments/temporal_decision.md.

### 2. Extract raw coordinates for train and validation only

    PYTHONPATH=backend python backend/scripts/prepare_raw_landmark_dataset.py --input-root /path/to/raw_dataset --data-dir-v1 /path/to/visionbridge_letter_data --output-dir /path/to/phase3_raw_train_val

The default splits are train and val. This command does not read test manifest rows or images. It verifies source image hashes and row/label order and saves raw xyz arrays plus explicit left/right presence flags.

### 3. Run each standalone augmentation separately

Run the command once per augmentation ID: gaussian_jitter, in_plane_rotation, depth_scale, and finger_joint_angles. Use a different output directory for each, but retain the same Phase 2 report, source split directories, raw feature manifest, seed set, and epoch budget.

    PYTHONPATH=backend python backend/scripts/run_augmentation_experiment.py --phase2-report /path/to/v4_comparison.json --data-dir-v1 /path/to/visionbridge_letter_data --data-dir-v4 /path/to/visionbridge_v4_letter_data --raw-dir /path/to/phase3_raw_train_val --output-dir /path/to/phase3_results/gaussian_jitter --augmentation gaussian_jitter --seeds 42 123 314 --epochs 1000 --batch-size 128

The runner retrains the Phase 2 winner architecture without augmentation and with exactly one named augmentation for each of at least three seeds. It opens train and validation only. It records per-class validation accuracy deltas, validation W→M/N/O confusion-count deltas, Python model-only latency, and anatomical rejection rate where applicable. It will not invent a Phase 2 winner if the comparison JSON is absent or incomplete.

### 4. Aggregate only completed standalone validation reports

    PYTHONPATH=backend python backend/scripts/aggregate_augmentation_results.py --result /path/to/phase3_results/gaussian_jitter/augmentation_result.json --result /path/to/phase3_results/in_plane_rotation/augmentation_result.json --result /path/to/phase3_results/depth_scale/augmentation_result.json --result /path/to/phase3_results/finger_joint_angles/augmentation_result.json

The aggregator rejects reports that opened test, use a different Phase 2 winner, use different seeds, or reference different train/validation manifests.

### 5. Optional one combined run

Only if all four standalone experiments have completed and at least two are marked KEEP by the validation rule, run at most one combined candidate. The runner reads the keep decisions from all four result JSONs and combines exactly that kept set; it records a shared one-attempt marker.

    PYTHONPATH=backend python backend/scripts/run_augmentation_experiment.py --phase2-report /path/to/v4_comparison.json --data-dir-v1 /path/to/visionbridge_letter_data --data-dir-v4 /path/to/visionbridge_v4_letter_data --raw-dir /path/to/phase3_raw_train_val --output-dir /path/to/phase3_results/combined --experiment-root /path/to/phase3_results --augmentation combined --standalone-result /path/to/phase3_results/gaussian_jitter/augmentation_result.json --standalone-result /path/to/phase3_results/in_plane_rotation/augmentation_result.json --standalone-result /path/to/phase3_results/depth_scale/augmentation_result.json --standalone-result /path/to/phase3_results/finger_joint_angles/augmentation_result.json --seeds 42 123 314 --epochs 1000 --batch-size 128

The shared experiment root refuses a second combined attempt.

### 6. Freeze validation decisions, then open test once

After all validation decisions have been made, run the finalizer once. Include the combined result only if a combined run was performed.

    PYTHONPATH=backend python backend/scripts/finalize_augmentation_test.py --phase2-report /path/to/v4_comparison.json --standalone-result /path/to/phase3_results/gaussian_jitter/augmentation_result.json --standalone-result /path/to/phase3_results/in_plane_rotation/augmentation_result.json --standalone-result /path/to/phase3_results/depth_scale/augmentation_result.json --standalone-result /path/to/phase3_results/finger_joint_angles/augmentation_result.json --combined-result /path/to/phase3_results/combined/augmentation_result.json --data-dir-v1 /path/to/visionbridge_letter_data --data-dir-v4 /path/to/visionbridge_v4_letter_data --output-dir /path/to/phase3_results/final --batch-size 128

The finalizer chooses the candidate from validation evidence only, verifies the checkpoint hash, writes the frozen selection manifest and one-shot marker before calling the evaluator. The marker blocks re-opening test in the same output directory, even if the evaluator fails. If no standalone augmentation meets the keep rule, the unaugmented Phase 2 architecture is evaluated as the final baseline.