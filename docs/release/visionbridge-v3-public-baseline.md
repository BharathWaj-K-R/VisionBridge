# VisionBridge V3 Public Release Baseline

## Current status

**READY WITH LIMITATIONS**

VisionBridge V3 is a browser-based Indian Sign Language **A–Z letter recognition** system. It uses MediaPipe hand landmarks with the V3 letter model and supports optional few-shot signer adaptation.

The current release baseline has verified model, runtime, persistence, authentication, and browser/PyTorch parity evidence. The release is suitable to be treated as the current production baseline within the documented scope below.

## What is verified

- V3 checkpoint is present and SHA-256 verified.
- Held-out test evaluation contains 4,633 samples.
- Overall held-out accuracy: **98.143751%**.
- Macro held-out accuracy: **98.167981%**.
- Model contract: **126 → 128 → 64 → 26**.
- Browser/PyTorch parity passed on 32 samples with a `1e-5` tolerance.
- Release-hardening test suite: **5/5 passed**.
- Production Render database binding is verified against the VisionBridge Supabase PostgreSQL instance.
- Production database execution uses the non-bypass `visionbridge_app` role.
- Custom FastAPI JWT/cookie authentication is integrated with PostgreSQL RLS.
- RLS is enabled on all seven application tables.

## Known limitations

### W-class weakness

W is the weakest recorded class in the held-out evaluation: **72.625698% accuracy**.

The aggregate 98.14% result must therefore not be interpreted as uniform performance across all letters.

### No signer-independent proof

Signer-independent evaluation remains **BLOCKED**.

The active RealSign source metadata does not expose verified signer IDs, so this release does not claim signer-independent generalization.

### Letter-level scope

VisionBridge currently recognizes **A–Z letters** with optional few-shot signer adaptation.

It does **not** claim full sentence-level Indian Sign Language translation, continuous linguistic translation, or broad signer-independent ISL understanding.

## Evidence of record

- `docs/release/visionbridge-v3-release-gate.json`
- `docs/release/visionbridge-v3-browser-parity.json`
- `docs/release/visionbridge-custom-auth-rls-verification.json`
- `docs/release/visionbridge-custom-auth-rls-plan.md`
- `visionbridge_v3_release_evidence.json`
- `visionbridge_letter_evaluation.json`
- `supabase/migrations/20260930103000_custom_auth_rls.sql`

The checkpoint referenced by the evidence is `backend/app/models/weights/letter_base_model.pt`.

SHA-256: `03bcb9f08d44025a025f78a08d68b2303b231270395117815ffcb322ca090cdf`

## Remaining maintenance

### pgTAP RLS test suite

A dedicated pgTAP suite is intentionally **not part of this release change**.

Future maintenance should add tests covering:

1. RLS enabled on every application table.
2. Application role has `BYPASSRLS=false`.
3. User A cannot read User B's history, adapters, profiles, words, quick-access rows, or usage rows.
4. User A cannot update or delete User B's rows.
5. Missing or invalid `app.user_id` cannot access user-owned rows.
6. Login identifier lookup remains limited to the requested account.
7. Registration INSERT remains limited to the registration operation context.
8. Transaction-local identity does not leak between pooled connections.

The pgTAP suite should be run through the Supabase CLI/CI against a controlled test database before being considered release evidence.

## Human production-baseline checklist

Run this on a real device using the production deployment.

### Browser and permissions
- [ ] Open the production VisionBridge site in a supported desktop/mobile browser.
- [ ] Confirm the page loads without a blank screen or fatal application error.
- [ ] Grant camera permission.
- [ ] Confirm the camera preview starts.
- [ ] Grant microphone permission when using the voice-to-sign flow.
- [ ] Confirm denying/regranting permissions does not leave the application permanently stuck.

### Authentication and persistence
- [ ] Register or log in with a test account.
- [ ] Refresh the page and confirm the authenticated session remains usable.
- [ ] Open history and confirm previous test translations are visible only to the logged-in account.
- [ ] Log out and confirm protected account data is no longer accessible.

### Letter recognition
- [ ] Open the recognition flow.
- [ ] Hold several clearly formed A–Z signs in front of the camera.
- [ ] Confirm predictions update without obvious UI freezing.
- [ ] Hold one sign steady and confirm the same letter is not repeatedly committed every frame.
- [ ] Change from one letter to another and confirm the new stable letter is eventually committed.
- [ ] Move the hand out of frame and confirm the UI does not crash.
- [ ] Test reasonable changes in distance and lighting.
- [ ] Confirm both-hand input behaves as expected where applicable.

### Personalization / adapter path
- [ ] If using the optional few-shot adapter, create a small test calibration set.
- [ ] Confirm the adapter can be saved.
- [ ] Refresh the application.
- [ ] Confirm the saved adapter remains available.
- [ ] Confirm an incompatible model/checkpoint is rejected rather than silently used.

### Voice / microphone path
- [ ] Open the existing voice-to-sign flow.
- [ ] Grant microphone permission.
- [ ] Speak a short test phrase.
- [ ] Confirm the flow responds or reports its existing failure state cleanly.
- [ ] Confirm microphone denial produces a usable error state rather than a crash.

### Basic resilience
- [ ] Refresh during normal use.
- [ ] Temporarily disable/re-enable the network and confirm the application fails visibly rather than silently corrupting state.
- [ ] Verify no test data from one account appears after switching to another account.
- [ ] Confirm the production site remains usable after a normal browser restart.

## Baseline decision

If the human checklist passes, treat **VisionBridge V3 as the current production baseline** within the stated A–Z letter-recognition scope.

Do not remove or weaken the documented W-class limitation or signer-independent-evaluation limitation when publishing the project.
