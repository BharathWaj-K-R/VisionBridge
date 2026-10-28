# VisionBridge V3 frozen baseline

**Status:** frozen snapshot  
**Captured:** 2026-10-28  
**Checkpoint SHA-256 verified against the committed binary before any edits.**

## Frozen checkpoint

| Property | Value |
|---|---|
| Path | `backend/app/models/weights/letter_base_model.pt` |
| SHA-256 | `f903872e96f269c5fbee6ea897f0027b91e162f2fc45b4c9a115e8ba8e38c917` |
| Size | 111,429 bytes |
| Parameter count | 26,582 |
| Model version | `visionbridge-letter-base-v3` |
| Input | 126 float values: left hand 63D followed by right hand 63D |
| Hidden width | 128 |
| Embedding | 64D |
| Output | 26 logits, labels A-Z |
| Preprocessing | `two-hand-wrist-scale-v1` |
| Landmark runtime | `mediapipe-hand-landmarker-0.10.35` |

Absent hands are represented by an all-zero 63D slot. Do not alter the V3 checkpoint, architecture, feature contract, or stored evaluation report as part of baseline work.

## Pinned dataset source

| Property | Value |
|---|---|
| Source repository | `https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset` |
| Source commit | `17c51dcc158b7b6359b7c6667edc51c7f148f3f4` |
| Archive size | 656,689,688 bytes |
| Archive SHA-256 | `008cae248e346b8c31fbbea057fcc3f69c6909d29a88e6bb1fb0369f528de2b5` |

These are the pinned values recorded with V3 release evidence. This phase did not download or re-verify the 656 MB source archive.

## Canonical evaluator command

Run from the repository root after preparing the pinned dataset at the stated path:

```bash
PYTHONPATH=backend python -m app.training.evaluate_letter_base --checkpoint backend/app/models/weights/letter_base_model.pt --data-dir /content/visionbridge_letter_data --split test --output-json /content/visionbridge_letter_evaluation.json
```

The evaluator defaults to batch size 128. The command above reproduces the documented test-split evaluation path; it does not tune or select a model.

## Stored report values, not a new evaluation

The frozen source report records 4,633 test samples, overall accuracy 0.9827325706885387, macro accuracy 0.9829129429244154, and W accuracy 0.7877094972067039. The recorded CPU latency is a historical measurement from that report. It is not reproduced latency, live-camera accuracy, or signer-independent evaluation.

The source report's class counts, per-class accuracy and confusion matrix are preserved verbatim in `visionbridge_letter_evaluation.json`.

## Historical-report immutability

The files in this directory are byte-for-byte copies of the two root-level V3 JSON reports as captured for this baseline. Treat both copies and their root-level source reports as **read-only historical evidence**. Do not overwrite them with newly generated results. Write future evaluations to a new output path and create a new versioned baseline snapshot instead.

Signer-independent evaluation remains **BLOCKED** until verified signer IDs are available. The stored aggregate test score does not establish signer-independent generalization.
