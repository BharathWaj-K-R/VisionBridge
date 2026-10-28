# V4 feature and checkpoint contract

**Status: proposed and frozen for implementation in Phase 2.** This document defines a new selectable feature/model contract. It does not change the V3 default or promote a checkpoint.

## Version identifiers

| Contract | Identifier |
|---|---|
| V3 model | `visionbridge-letter-base-v3` |
| V3 preprocessing | `two-hand-wrist-scale-v1` |
| V4 model | `visionbridge-letter-base-v4` |
| V4 preprocessing | `two-hand-geometry-v2` |
| Landmark runtime | `mediapipe-hand-landmarker-0.10.35` |
| Vocabulary | `ABCDEFGHIJKLMNOPQRSTUVWXYZ` |
| Embedding dimension | 64 |
| Classifier outputs | 26 logits, A through Z |
| Numeric dtype | IEEE-754 float32 in Python and JavaScript |
| Input handedness convention | Raw MediaPipe Tasks labels: label "left" fills the left slot; label "right" fills the right slot. Do not mirror-swap labels because the preview is mirrored. |

V3 identifiers, V3 checkpoint bytes, and V3 serialization remain unchanged. V3 and V4 checkpoints are not interchangeable. V4 loading requires V4 model and preprocessing metadata matching the architecture variant; the V3 loader must reject every V4 checkpoint and the V4 loader must reject every V3 checkpoint.

## Input to preprocessing

The V4 preprocessing function consumes raw MediaPipe x/y/z landmarks in source handedness order, laid out as left hand's 21 xyz triples followed by right hand's 21 xyz triples (126 float32 values). Detection presence is supplied separately as two booleans. A missing hand must not be inferred from a participant ID, image name, or other metadata.

Each hand has 21 landmarks with x, y, z in MediaPipe Tasks order. Landmark indices are zero-based. Wrist index is 0; fingertips are thumb 4, index 8, middle 12, ring 16, and pinky 20.

## Exact output layout: 141 float32 values

The output is one flat vector. Offsets below are inclusive.

| Offsets | Width | Meaning |
|---|---:|---|
| 0–62 | 63 | Left hand, wrist-centered and scale-normalized xyz for landmarks 0–20; landmark-major xyz order |
| 63 | 1 | Left-hand presence flag, exactly 1.0 if detected, else 0.0 |
| 64–126 | 63 | Right hand, same 21 x (x,y,z) layout |
| 127 | 1 | Right-hand presence flag, exactly 1.0 if detected, else 0.0 |
| 128–130 | 3 | Inter-wrist vector: right wrist minus left wrist, x/y/z, divided by the shared geometry scale |
| 131–135 | 5 | Euclidean distance from each left fingertip (4, 8, 12, 16, 20) to the right wrist, divided by the shared geometry scale |
| 136–140 | 5 | Euclidean distance from each right fingertip (4, 8, 12, 16, 20) to the left wrist, divided by the shared geometry scale |

The shape is exactly `[141]` per sample or `[batch, 141]` per batch. There is no implicit padding or optional trailing geometry.

## Normalization equations

For each present hand independently, let \(p_0\) be its raw wrist and let \(s_h = \max_i ||p_i-p_0||_2\) over its 21 raw landmarks. Each normalized hand landmark is \((p_i-p_0)/s_h\), with the same `1e-6` degeneracy guard used by v1. If a hand is absent, all 63 coordinate values in its slot are zeros, regardless of the coordinate buffer contents. The explicit presence flag still reflects the supplied detection boolean; a detected degenerate hand may therefore have a 1.0 presence flag and zero normalized coordinates.

When both hands are present, the shared geometry scale is \(s_g = \max(s_{left}, s_{right}, 1e-6)\). The inter-wrist vector is \((w_{right}-w_{left})/s_g\). Each cross-wrist fingertip distance is \(||tip-wrist_{opposite}||_2/s_g\). Geometry values are float32.

When either hand is absent, all 13 relative-geometry values are exactly zero. The absent hand's coordinate slot is 63 zeros and its presence flag is 0.0. When both hands are present but \(s_g < 1e-6\), the shared geometry values are all zero rather than divided by an unstable scale.

The entire output must be finite. Inputs of the wrong length, non-finite raw coordinates, or invalid presence flags are rejected; the function must not silently convert NaN/Inf into plausible coordinates.

## V4 model architecture

The full V4 model consumes the 141D vector. Its shared hand encoder receives each 64D hand slot (63 coordinates plus explicit presence flag) with the exact same weights for left and right. The encoder output is 32D per hand. Left/right roles remain explicit: encoded hand vectors are combined in left-then-right order and receive separate learned role embeddings before fusion. The fusion block concatenates the 32D left embedding, 32D right embedding, and 13D relative-geometry block (77D total), applies a 128D hidden fusion layer, then produces a 64D embedding and 26 A-Z logits.

The model must expose `embed(inputs) -> [batch, 64]` and `forward(inputs) -> [batch, 26]`. Every checkpoint records model version, architecture variant, preprocessing version, runtime, input/hidden/embedding/class dimensions, and the exact A-Z labels.

## Named ablations

Ablations are different metadata-bound V4 variants; none is a production default.

| Variant ID | Preprocessing | Input width | Model |
|---|---|---:|---|
| `v4_features_old_mlp` | `two-hand-geometry-v2` | 141 | Original V3-style MLP shape with input width adjusted to 141; no shared hand encoder |
| `v1_features_shared_encoder` | `two-hand-wrist-scale-v1` | 126 | Shared hand encoder and fusion head; relative geometry unavailable and set to zero; hand-presence flags are derived from whether each legacy 63D hand slot is nonzero |
| `full_v4` | `two-hand-geometry-v2` | 141 | Shared hand encoder, explicit presence, and 13D relative-hand geometry |

Use the same sample identities and split assignments for all candidate variants. A V4 141D array cannot be reconstructed truthfully from legacy v1-normalized 126D arrays because per-hand normalization removes absolute wrist placement. The full V4 data path therefore needs raw landmark frames or precomputed V4 features generated from those frames, aligned with the same split manifests and sample IDs.

## Adapter binding

Adapters remain prototype-based and bound to the exact checkpoint SHA-256, model version, preprocessing identifier, runtime, input feature contract, and 64D embedding. A V4 checkpoint requires fresh recalibration. V3 adapters must be rejected by V4 and V4 adapters must be rejected by V3 even if the prototype payload happens to have 64D vectors.

## Experiment protocol

- Use identical train/validation/test sample identities and splits across variants, at least three fixed seeds, and identical epoch budgets, optimizer/loss settings, and selection rules.
- Candidate training and selection may inspect train and validation only. Use validation mean and standard deviation across seeds for selection; do not choose a seed based on test performance.
- Freeze the final candidate configuration and checkpoint SHA-256 before the one final test evaluation. Do not report more than one final test score as a promotion comparison.
- Report parameter count and Python model-only latency using the same batch size, warm-up count, and timed runs. A browser latency value may be reported only when measured on an identified browser/device; otherwise record it as not measured.
- Record all actual run commands, seeds, dataset/split manifest hashes, code commit, selected candidate, checkpoint hash, and the test-evaluation run identity.

The active RealSign dataset currently lacks verified signer IDs. This blocks signer-independent claims. The V4 tooling must never infer signer IDs; its held-out-signer evaluation gate stays blocked until verified, consented new data is available.
