-- VisionBridge custom FastAPI-auth RLS.
-- The production Render connection authenticates with the existing Supabase
-- database role, then SET ROLE visionbridge_app. The execution role has
-- NOBYPASSRLS. FastAPI binds its JWT identity to transaction-local custom GUCs.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'visionbridge_app') THEN
    CREATE ROLE visionbridge_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
  END IF;
END
$$;

GRANT CONNECT ON DATABASE postgres TO visionbridge_app;
GRANT USAGE ON SCHEMA public TO visionbridge_app;
GRANT SELECT, INSERT, UPDATE, DELETE
ON TABLE public.users,
         public.signer_adapters,
         public.translation_logs,
         public.communication_words,
         public.quick_access_slots,
         public.personalization_profiles,
         public.communication_usage
TO visionbridge_app;
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public TO visionbridge_app;
GRANT visionbridge_app TO postgres WITH INHERIT FALSE, SET TRUE;

REVOKE ALL ON TABLE
  public.users,
  public.signer_adapters,
  public.translation_logs,
  public.communication_words,
  public.quick_access_slots,
  public.personalization_profiles,
  public.communication_usage
FROM PUBLIC, anon, authenticated;

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.signer_adapters ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.translation_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.communication_words ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quick_access_slots ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.personalization_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.communication_usage ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS visionbridge_users_select ON public.users;
CREATE POLICY visionbridge_users_select ON public.users
FOR SELECT TO visionbridge_app
USING (
  current_setting('app.user_id', true) = id::text
  OR (current_setting('app.auth_operation', true) = 'login'
      AND (username = current_setting('app.auth_identifier', true)
           OR email = lower(current_setting('app.auth_identifier', true))))
  OR (current_setting('app.auth_operation', true) = 'register'
      AND (username = current_setting('app.auth_username', true)
           OR email = current_setting('app.auth_email', true)))
);

DROP POLICY IF EXISTS visionbridge_users_insert ON public.users;
CREATE POLICY visionbridge_users_insert ON public.users
FOR INSERT TO visionbridge_app
WITH CHECK (
  current_setting('app.auth_operation', true) = 'register'
  AND username = current_setting('app.auth_username', true)
  AND email = current_setting('app.auth_email', true)
);

DROP POLICY IF EXISTS visionbridge_users_update ON public.users;
CREATE POLICY visionbridge_users_update ON public.users
FOR UPDATE TO visionbridge_app
USING (current_setting('app.user_id', true) = id::text)
WITH CHECK (current_setting('app.user_id', true) = id::text);

DROP POLICY IF EXISTS visionbridge_signer_adapters_owner ON public.signer_adapters;
CREATE POLICY visionbridge_signer_adapters_owner ON public.signer_adapters
TO visionbridge_app
USING (current_setting('app.user_id', true) = owner_id::text)
WITH CHECK (current_setting('app.user_id', true) = owner_id::text);

DROP POLICY IF EXISTS visionbridge_translation_logs_owner ON public.translation_logs;
CREATE POLICY visionbridge_translation_logs_owner ON public.translation_logs
TO visionbridge_app
USING (current_setting('app.user_id', true) = user_id::text)
WITH CHECK (current_setting('app.user_id', true) = user_id::text);

DROP POLICY IF EXISTS visionbridge_communication_words_owner ON public.communication_words;
CREATE POLICY visionbridge_communication_words_owner ON public.communication_words
TO visionbridge_app
USING (current_setting('app.user_id', true) = user_id::text)
WITH CHECK (current_setting('app.user_id', true) = user_id::text);

DROP POLICY IF EXISTS visionbridge_quick_access_owner ON public.quick_access_slots;
CREATE POLICY visionbridge_quick_access_owner ON public.quick_access_slots
TO visionbridge_app
USING (current_setting('app.user_id', true) = user_id::text)
WITH CHECK (current_setting('app.user_id', true) = user_id::text);

DROP POLICY IF EXISTS visionbridge_profiles_owner ON public.personalization_profiles;
CREATE POLICY visionbridge_profiles_owner ON public.personalization_profiles
TO visionbridge_app
USING (current_setting('app.user_id', true) = user_id::text)
WITH CHECK (current_setting('app.user_id', true) = user_id::text);

DROP POLICY IF EXISTS visionbridge_usage_owner ON public.communication_usage;
CREATE POLICY visionbridge_usage_owner ON public.communication_usage
TO visionbridge_app
USING (current_setting('app.user_id', true) = user_id::text)
WITH CHECK (current_setting('app.user_id', true) = user_id::text);
