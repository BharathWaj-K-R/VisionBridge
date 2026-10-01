-- Remove the email OTP account-verification lifecycle.
-- Existing accounts remain usable; normal password registration/login remains the
-- sole account-creation authentication flow.
BEGIN;

ALTER TABLE public.users
  DROP COLUMN IF EXISTS otp_hash,
  DROP COLUMN IF EXISTS otp_expires_at,
  DROP COLUMN IF EXISTS otp_sent_at,
  DROP COLUMN IF EXISTS otp_attempts,
  DROP COLUMN IF EXISTS is_verified;

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
);

DROP POLICY IF EXISTS visionbridge_users_update ON public.users;
CREATE POLICY visionbridge_users_update ON public.users
FOR UPDATE TO visionbridge_app
USING (
  current_setting('app.user_id', true) = id::text
)
WITH CHECK (
  current_setting('app.user_id', true) = id::text
);

COMMIT;
