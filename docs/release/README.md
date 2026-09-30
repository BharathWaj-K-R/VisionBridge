# VisionBridge Release Hardening

## V3 release gate

The active release is `visionbridge-letter-base-v3`.

The release is **blocked** until all of the following are backed by real artifacts:

1. `backend/app/models/weights/letter_base_model.pt` exists and passes the strict loader.
2. SHA-256 is recorded.
3. The canonical held-out test split is evaluated once and the report is retained.
4. Overall and macro accuracy are recorded.
5. All A-Z per-letter accuracies are recorded.
6. The full 26x26 confusion matrix is recorded.
7. The model contract remains 126 -> 128 -> 64 -> 26 with the active preprocessing/runtime versions.
8. Browser/PyTorch parity passes with max absolute embedding/logit error <= 1e-5.
9. Signer-independent evaluation remains a separate release gate and is currently blocked because the active RealSign metadata does not expose verified signer IDs.

## Commands

After the verified checkpoint and prepared dataset are available:

```bash
PYTHONPATH=backend python -m app.training.evaluate_letter_base \
  --checkpoint backend/app/models/weights/letter_base_model.pt \
  --data-dir /path/to/visionbridge_letter_data \
  --split test \
  --output-json docs/release/visionbridge-v3-evaluation.json

PYTHONPATH=backend python backend/scripts/verify_browser_parity.py \
  --checkpoint backend/app/models/weights/letter_base_model.pt \
  --samples 32 \
  --tolerance 1e-5 \
  --output-json docs/release/visionbridge-v3-browser-parity.json
```

The current gate manifest is:

`docs/release/visionbridge-v3-release-gate.json`

It intentionally contains null metrics while the checkpoint/evaluation evidence is unavailable. Null is preferable to a very impressive fictional number, which is how technical debt gets promoted to management.
