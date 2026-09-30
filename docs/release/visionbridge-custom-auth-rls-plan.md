# VisionBridge custom-auth RLS implementation

## Implemented state

Production Supabase PostgreSQL now has RLS enabled on all seven VisionBridge application tables.

The FastAPI service does not use Supabase Auth. It keeps its existing JWT/HttpOnly-cookie authentication and binds the authenticated integer user id to a transaction-local PostgreSQL setting with `set_config(..., true)`.

The Render connection authenticates with the existing database role and immediately executes `SET ROLE visionbridge_app`. The effective execution role is non-bypass: `rolbypassrls=false`.

## Implemented controls

- Dedicated execution role: `visionbridge_app`
- Effective production role: `visionbridge_app`
- `BYPASSRLS=false`
- Transaction-local `app.user_id`
- Transaction-local auth-operation context for login/registration
- Owner policies for adapters, history, custom words, quick access, profiles, and usage
- User-row policy permitting only the authenticated user's own row, the specific login identifier being checked, or the specific username/email being checked during registration
- `anon`, `authenticated`, and `PUBLIC` table privileges revoked from the seven application tables
- Sequence privileges granted only to the application role as required

## Verification

Production verification is recorded in `docs/release/visionbridge-custom-auth-rls-verification.json`.

Observed runtime evidence:
- Render database host: `aws-0-ap-southeast-1.pooler.supabase.com:5432`
- Database: `postgres`
- Render startup effective role: `current_user=visionbridge_app`, `session_user=postgres`
- All seven tables report RLS enabled.
- The application role reports `bypassrls=false`.
- Login lookup succeeded under the custom operation context.
- Registration INSERT policy accepted a transactional probe, which was rolled back.
- Owner-scoped reads were verified under two user contexts.

## Important security property

The identity settings use transaction-local scope. This matters because SQLAlchemy pools connections. A request's user identity therefore cannot persist into the next transaction on the same pooled connection.

The application never uses `auth.uid()`. That would describe Supabase Auth identity, which VisionBridge does not use.

## Versioned migration

The reproducible migration is `supabase/migrations/20260930103000_custom_auth_rls.sql`.

## Remaining test work

The production RLS implementation is active and runtime-verified. A dedicated pgTAP suite should be added to the repository and run through the Supabase CLI in CI as a follow-up release-maintenance task. The current release evidence is based on direct production SQL verification plus application runtime startup evidence, not a claimed pgTAP run.

## Explicit limitations

- Signer-independent evaluation remains blocked because verified signer IDs are not exposed by the active RealSign metadata.
- W remains the weakest recorded letter at 72.625698% accuracy.
