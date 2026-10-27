# VisionBridge V3 Public Model Baseline

## Status reviewed: 2026-10-27

**Model artifact status: recorded and test-evaluated. Release gate: BLOCKED.**

VisionBridge V3 is a browser-based Indian Sign Language **A–Z isolated-letter recognition** system. It uses MediaPipe hand landmarks and a compact letter model, with optional few-shot signer adaptation. The scope is single-letter recognition; the speech-to-sentence page is a prototype and is not a validated semantic ISL sentence translator.

This page distinguishes evidence for the current checkpoint from full release readiness. A same-source random/stratified image test is not signer-independent evaluation.

## Current checkpoint

- Path: `backend/app/models/weights/letter_base_model.pt`
- SHA-256: `f903872e96f269c5fbee6ea897f0027b91e162f2fc45b4c9a115e8ba8e38c917`
- Model contract: **126 → 128 → 64 → 26**.
- Preprocessing: `two-hand-wrist-scale-v1`
- Landmark runtime: `mediapipe-hand-landmarker-0.10.35`
- Parameters: **26,582**
- File size: **111,429 bytes**

## Current recorded test evaluation

The committed `visionbridge_letter_evaluation.json` report records the following results for the prepared test split:

| Metric | Result |
|---|---:|
| Prepared test samples | 4,633 |
| Overall accuracy | 98.273257% |
| Macro class accuracy | 98.291294% |
| Weakest class | W |
| W accuracy | 78.770950% |

The report includes per-letter counts and a confusion matrix. W is commonly confused with M, N, and O. The per-letter prepared test counts range from 137 to 200; samples where landmark extraction fails are skipped during preparation. These metrics are not a live-webcam accuracy guarantee and do not characterize every source image.

Previously published metrics from an earlier checkpoint remain historical records in the dated experiment/debug evidence. Do not present archived values as the current checkpoint's results.

## What has recorded verification evidence

- Current checkpoint is present in the repository and its expected hash is recorded.
- Prepared held-out test evaluation is stored in `visionbridge_letter_evaluation.json`.
- Browser/PyTorch numerical parity passed for 32 generated vectors at `1e-5` tolerance in the recorded workflow run.
- Release-hardening test file recorded 5/5 passing tests in its cited run.
- A production PostgreSQL/RLS verification artifact records checks against the non-bypass `visionbridge_app` role.

These artifacts are dated evidence, not a claim that every corresponding external service or physical device was re-tested on 2026-10-27. The latest observed product-validation run before this documentation update is [GitHub Actions run 40485483197](https://github.com/BharathWaj-K-R/VisionBridge/actions/runs/40485483197); the documentation commit requires its own new CI result.

## Release blocker: signer-independent evaluation

Signer-independent evaluation is **BLOCKED** because the active RealSign metadata does not expose verified signer IDs. The engineering contract treats signer-independent evaluation as a mandatory release gate. Until a dataset with verified signer identities can support a proper signer-held-out test, this release must not be described as fully validated or ready for release.

A strong aggregate accuracy on the available prepared image split does not establish generalization to unseen signers.

## Other limitations

- The W class remains materially weaker than most other classes.
- The model predicts isolated A–Z letters from a frame; it does not decode continuous ISL grammar or sentence-level sign language.
- The browser speech/sentence path is a prototype; the word and phrase sign-asset maps are currently empty.
- Automated CI does not exercise actual camera/microphone permissions or performance on physical devices.
- A dedicated pgTAP RLS suite has not yet been added and run in CI.
- Real-browser and complete live production end-to-end behavior must be manually verified.

## Evidence of record

- `docs/release/visionbridge-v3-release-gate.json`
- `docs/release/visionbridge-v3-browser-parity.json`
- `docs/release/visionbridge-custom-auth-rls-verification.json`
- `docs/release/visionbridge-custom-auth-rls-plan.md`
- `visionbridge_v3_release_evidence.json`
- `visionbridge_letter_evaluation.json`
- `supabase/migrations/20260930103000_custom_auth_rls.sql`

## Human verification checklist

Run this on a real device using the production deployment. Check these items only after actually performing each one.

### Browser and permissions
- [ ] Open the deployed VisionBridge site in a supported desktop/mobile browser.
- [ ] Confirm the page loads without a blank screen or fatal application error.
- [ ] Grant camera permission and confirm the preview starts.
- [ ] Grant microphone permission when testing the Speak prototype.
- [ ] Confirm denying and re-granting permissions does not leave the app permanently stuck.

### Authentication and persistence
- [ ] Register or log in with a test account.
- [ ] Refresh and confirm the authenticated session remains usable.
- [ ] Confirm history belongs to the logged-in account.
- [ ] Log out and confirm protected account data is no longer accessible.

### Letter recognition and adaptation
- [ ] Hold several clearly formed A–Z signs in front of the camera.
- [ ] Hold one sign steady and confirm the same letter is not repeatedly committed every frame.
- [ ] Change letters and confirm the next stable letter is committed.
- [ ] Move the hand out of frame and test reasonable lighting/distance changes.
- [ ] Test adapter calibration, persistence, and mismatch rejection if calibration is used.

### Speak prototype
- [ ] Speak a short test phrase and confirm the page responds or reports its current failure state clearly.
- [ ] Confirm microphone denial produces a usable recovery state.
- [ ] Do not treat a successful speech transcript as proof of semantic ISL translation.

### Resilience
- [ ] Refresh during normal use.
- [ ] Test a temporary network interruption and recovery.
- [ ] Verify that one account's data is not shown after switching accounts.
- [ ] Confirm the deployment remains usable after a normal browser restart.

## Baseline decision

The model artifact and its recorded same-source test evaluation are documented. **Release sign-off remains blocked** by the required signer-independent evaluation gate. Passing the manual browser checklist alone does not resolve the missing signer-identity evidence.
