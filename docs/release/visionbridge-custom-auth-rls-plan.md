# VisionBridge custom-auth RLS plan

## Current state

Supabase PostgreSQL is the durable production database, but the application currently connects as the `postgres` role. Supabase documents that the table owner and roles with `BYPASSRLS` are not constrained by RLS. RLS is therefore intentionally still disabled while the application authentication model is custom FastAPI JWT/HttpOnly-cookie authentication rather than Supabase Auth.

## Required design

1. **Use a dedicated application database role**
   - Create a login role such as `visionbridge_app`.
   - It must have `NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS`.
   - Grant only the schema/table/sequence privileges required by FastAPI.
   - Store its password only in Render's secret `DATABASE_URL`.
   - Use Supabase's IPv4-capable session pooler because Render is IPv4-only.

2. **Bind FastAPI JWT identity to a transaction-local PostgreSQL setting**
   - The JWT `sub` is the VisionBridge integer user id.
   - Before any authenticated ORM query, execute `SELECT set_config('app.user_id', :user_id, true)`.
   - Keep the third argument `true`, making the identity transaction-local and preventing pooled-connection leakage.
   - Anonymous requests do not receive an application user context.
   - `get_current_user` should establish this context before querying `users`.

3. **Policies use custom context, not `auth.uid()`**
   - Use `current_setting('app.user_id', true)` in policies.
   - Enforce ownership on `users.id`, `signer_adapters.owner_id`, `translation_logs.user_id`, `communication_words.user_id`, `quick_access_slots.user_id`, `personalization_profiles.user_id`, and `communication_usage.user_id`.
   - Define explicit SELECT, INSERT, UPDATE, and DELETE policies as appropriate.

4. **Registration is an explicit exception**
   - Registration is unauthenticated, so `users` needs an INSERT policy permitting account creation.
   - No anonymous SELECT, UPDATE, or DELETE access is granted.

5. **Grants and RLS are one migration**
   - Enable RLS on all seven application tables.
   - Revoke unnecessary privileges from `anon` and `authenticated`.
   - Grant only required privileges to `visionbridge_app`.
   - Keep administrative access separate from the application role.

6. **Prove isolation before production cutover**
   - Test two users and verify each can access only its own rows.
   - Test cross-user SELECT/UPDATE/DELETE denial for every user-owned table.
   - Test missing/invalid user context.
   - Verify `rolbypassrls = false` for the application role.
   - Run authenticated smoke tests after enabling RLS.

## Cutover order

1. Create the dedicated non-bypass role.
2. Grant least-privilege table and sequence access.
3. Implement transaction-local `app.user_id` context in FastAPI.
4. Add and run RLS policy tests against the real Supabase database.
5. Enable RLS in one migration.
6. Change Render `DATABASE_URL` to the dedicated role/pooler URL.
7. Deploy and verify authenticated smoke tests.
8. Only then mark the RLS gate passed.

## Explicit non-goals

- Do not use `auth.uid()`: VisionBridge does not use Supabase Auth.
- Do not grant `BYPASSRLS` to the application role.
- Do not enable RLS first and discover afterward that the current `postgres` connection bypasses it.
- Do not expose database credentials or the application role password in logs, source, or API responses.
