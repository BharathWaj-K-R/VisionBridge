# Augmentation experiment protocol

**Status: NOT RUN on real ISL data.** The data and Phase 2 winner checkpoint are not available in this execution environment. Do not interpret structural tests or results from other sign languages as ISL improvement evidence.

## Implemented independent augmentations

Each augmentation operates on raw 126D MediaPipe coordinates before wrist centering and scale normalization. One named augmentation is allowed per training run. Mirroring is excluded. Random single-hand dropout is skipped because the available source does not establish a verified ISL per-letter hand-configuration table, and dropout could remove label-defining information.

| ID | Operation | Default magnitude | Plausibility/rejection policy | Real-data result |
|---|---|---|---|---|
| gaussian_jitter | independent xyz Gaussian noise on detected hands | std 0.002 in raw MediaPipe coordinate units | finite input/output; missing hand stays zero | NOT RUN |
| in_plane_rotation | jointly rotate detected hands around their mean wrist center in x/y | uniform angle ±5° | finite input/output; relative hand arrangement rotates together | NOT RUN |
| depth_scale | global image-plane and z scale about mean wrist center | independent factors 0.95–1.05 | finite input/output; both hands use same factors | NOT RUN |
| finger_joint_angles | perturb one PIP/DIP angle per non-thumb finger and thumb IP; rigidly rotate its distal subtree | uniform angular delta ±5° | reject bad base geometry, joint flexion above caps, non-finite values, or bone-length drift above 5%; rejected augmentations fall back to the original raw sample and are counted | NOT RUN |
| single_hand_dropout | intentionally not implemented | n/a | no verified per-letter ISL hand-configuration table | SKIPPED |
| mirroring | intentionally not implemented | n/a | can alter handedness/letter semantics without ISL validation | SKIPPED |

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
