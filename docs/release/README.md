# VisionBridge Release Hardening

## V3 release gate

The active release is `visionbridge-letter-base-v3`.

The release remains **blocked** until production database binding and custom-auth RLS are backed by real runtime evidence. The V3 checkpoint, evaluation evidence, browser/PyTorch parity, and release-hardening tests are now verified.

## Verified evidence

- Checkpoint: `backend/app/models/weights/letter_base_model.pt`
- Checkpoint SHA-256: `03bcb9f08d44025a025f78a08d68b2303b231270395117815ffcb322ca090cdf`
- Held-out test samples: 4633
- Overall accuracy: 98.143751%
- Macro accuracy: 98.167981%
- Weakest recorded letter: W at 72.625698%
- Model contract: `126 -> 128 -> 64 -> 26`
- Browser/PyTorch parity: 32 samples, embedding max error `1.9669532775878906e-06`, logits max error `3.814697265625e-06`, tolerance `1e-5`, passed
- Release-hardening tests: 5/5 passed

Parity evidence is retained at `docs/release/visionbridge-v3-browser-parity.json`.

## Remaining blockers

1. Verify the sanitized live Render startup log identifies the production database as the VisionBridge Supabase PostgreSQL instance.
2. Replace the current `postgres` production application connection with a dedicated non-bypass RLS role and implement transaction-local custom-auth identity context.
3. Enable and test RLS only after the dedicated role and policy suite are proven.
4. Signer-independent evaluation remains blocked because the active RealSign metadata does not expose verified signer IDs.

## Commands

```bash
PYTHONPATH=backend python backend/scripts/verify_browser_parity.py \
  --checkpoint backend/app/models/weights/letter_base_model.pt \
  --samples 32 \
  --tolerance 1e-5 \
  --output-json docs/release/visionbridge-v3-browser-parity.json

PYTHONPATH=backend pytest -q backend/tests/test_release_hardening.py
```

The release gate manifest is `docs/release/visionbridge-v3-release-gate.json`.

Do not claim signer-independent evaluation, and do not hide the W-class weakness behind aggregate accuracy.
