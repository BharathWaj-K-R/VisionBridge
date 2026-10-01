-- Email verification and strong-account lifecycle fields.
-- Existing accounts are backfilled as verified so the new security control does
-- not unexpectedly lock users out. New accounts use the false default.
BEGIN;

ALTER TABLE public.users
  ADD COLUMN IF NOT EXISTS is_verified boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS otp_hash text,
  ADD COLUMN IF NOT EXISTS otp_expires_at timestamptz,
  ADD COLUMN IF NOT EXISTS otp_sent_at timestamptz,
  ADD COLUMN IF NOT EXISTS otp_attempts integer NOT NULL DEFAULT 0;

UPDATE public.users
SET is_verified = true
WHERE is_verified IS DISTINCT FROM true;

ALTER TABLE public.users
  ALTER COLUMN is_verified SET DEFAULT false;

DROP POLICY IF EXISTS visionbridge_users_select ON public.users;
CREATE POLICY visionbridge_users_select ON public.users
FOR SELECT TO visionbridge_app
USING (
  current_setting('app.user_id', true) = id::text
  OR (
    current_setting('app.auth_operation', true) = 'login'
    AND (
      username = current_setting('app.auth_identifier', true)
      OR email = lower(current_setting('app.auth_identifier', true))
    )
  )
  OR (
    current_setting('app.auth_operation', true) = 'register'
    AND (
      username = current_setting('app.auth_username', true)
      OR email = current_setting('app.auth_email', true)
    )
  )
  OR (
    current_setting('app.auth_operation', true) IN ('verify_otp', 'resend_otp')
    AND email = lower(current_setting('app.auth_email', true))
  )
);

DROP POLICY IF EXISTS visionbridge_users_update ON public.users;
CREATE POLICY visionbridge_users_update ON public.users
FOR UPDATE TO visionbridge_app
USING (
  current_setting('app.user_id', true) = id::text
  OR (
    current_setting('app.auth_operation', true) IN ('verify_otp', 'resend_otp')
    AND email = lower(current_setting('app.auth_email', true))
  )
)
WITH CHECK (
  current_setting('app.user_id', true) = id::text
  OR (
    current_setting('app.auth_operation', true) IN ('verify_otp', 'resend_otp')
    AND email = lower(current_setting('app.auth_email', true))
  )
);

COMMIT;
