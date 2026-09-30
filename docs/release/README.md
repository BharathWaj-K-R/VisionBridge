# VisionBridge Release Hardening

## V3 release gate

The active release is `visionbridge-letter-base-v3`.

**Release status: READY WITH LIMITATIONS.**

Verified gates:
- V3 checkpoint present and SHA-256 verified.
- Held-out test evidence: 98.143751% overall accuracy, 98.167981% macro accuracy.
- Model contract: `126 -> 128 -> 64 -> 26`.
- Browser/PyTorch parity passed on 32 samples at `1e-5` tolerance.
- Release-hardening suite: 5/5 passed.
- Render production database binding verified as Supabase pooler PostgreSQL.
- Render effective execution role verified as `visionbridge_app` with `BYPASSRLS=false`.
- RLS enabled on all seven application tables with custom FastAPI JWT/cookie context.

## Known limitations

- W is the weakest recorded class at 72.625698% accuracy.
- Signer-independent evaluation remains blocked because the active RealSign metadata does not expose verified signer IDs.
- Product scope remains browser-based A-Z letter recognition with optional few-shot signer adaptation. It is not a claim of sentence-level/full ISL translation.

## Evidence artifacts

- `docs/release/visionbridge-v3-release-gate.json`
- `docs/release/visionbridge-v3-browser-parity.json`
- `docs/release/visionbridge-custom-auth-rls-verification.json`
- `docs/release/visionbridge-custom-auth-rls-plan.md`
- `supabase/migrations/20260930103000_custom_auth_rls.sql`

## Release maintenance

Add and run a dedicated pgTAP RLS suite through the Supabase CLI/CI. This is release-maintenance hardening, not a product feature. Until that suite is run, the RLS gate is runtime-verified but not pgTAP-verified.

Do not claim signer-independent evaluation, and do not hide the W-class weakness behind aggregate accuracy.
