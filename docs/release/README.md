# VisionBridge Release Hardening

**Status reviewed: 2026-10-27**

## Release decision

**Release gate: BLOCKED.** The current checkpoint has recorded same-source held-out test metrics and the code-validation workflow passed on the preceding README commit, but the engineering contract requires signer-independent evaluation. The active RealSign metadata does not contain verified signer IDs, so that mandatory gate cannot currently be completed. Do not promote aggregate test accuracy into a signer-independent claim.

The model artifact and its same-source test evaluation can be described as recorded/verified within their stated scope. That is not equivalent to complete release sign-off.

## Current model artifact and recorded evaluation

- Checkpoint: `backend/app/models/weights/letter_base_model.pt`
- SHA-256: `f903872e96f269c5fbee6ea897f0027b91e162f2fc45b4c9a115e8ba8e38c917`
- Model contract: `126 → 128 → 64 → 26`
- Recorded prepared test set: 4,633 samples.
- Overall accuracy: 98.273257%.
- Macro class accuracy: 98.291294%.
- Weakest class: W at 78.770950%.
- Browser/PyTorch numerical parity: 32 generated vectors, maximum absolute errors below the recorded `1e-5` tolerance.
- Release-hardening tests: 5/5 passed in the recorded run.
- Production PostgreSQL/RLS checks: runtime evidence is recorded in the release verification artifact; it was not re-probed as part of this documentation pass.

See `visionbridge_letter_evaluation.json`, `visionbridge_v3_release_evidence.json`, and `docs/release/visionbridge-v3-release-gate.json` for artifact-level evidence. Historic comparison metrics remain in the experiment record and are not the current checkpoint's metrics.

## Known limitations and blockers

- **Mandatory release blocker:** signer-independent test evaluation is blocked because the source metadata does not expose verified signer IDs.
- **W-class weakness:** W has 78.770950% accuracy in the current recorded test report and is often confused with M, N, and O.
- **Physical device verification:** GitHub Actions does not exercise real camera/microphone permissions, lighting, camera angles, or varied hardware.
- **Live production end-to-end verification:** do not infer this from the production build or the historical database verification artifact.
- **RLS test automation:** a dedicated pgTAP suite still needs to be added and run through Supabase CLI/CI.

## Evidence artifacts

- `docs/release/visionbridge-accuracy-experiments.json` — retained candidate-comparison history.
- `docs/release/visionbridge-v3-release-gate.json` — release status and recorded evidence references.
- `docs/release/visionbridge-v3-browser-parity.json` — browser/PyTorch parity record for the pinned checkpoint.
- `docs/release/visionbridge-custom-auth-rls-verification.json` — recorded production SQL/RLS verification.
- `docs/release/visionbridge-custom-auth-rls-plan.md` — implementation and remaining RLS test work.
- `supabase/migrations/20260930103000_custom_auth_rls.sql` — versioned migration.

## Required next release work

1. Obtain or construct a dataset with verified signer identities and run signer-independent evaluation. Do not substitute a random image split for signer holdout.
2. Add and run a dedicated pgTAP RLS suite through the Supabase CLI/CI.
3. Complete real-browser camera/microphone and production end-to-end checks.
4. Re-run the full release workflow against the exact candidate checkpoint, then update the status only from the resulting evidence.

Until the signer-independent gate is satisfied, the release remains **BLOCKED** even though the current recorded test report has high aggregate accuracy.
