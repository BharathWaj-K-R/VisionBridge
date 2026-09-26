# Changelog

## Unreleased

### Product readiness

- Added fail-closed production configuration checks for secret, database, and CORS safety.
- Added explicit Render production API configuration to the deployment contract.
- Added a safe local environment template without production secrets.
- Documented the live Render frontend/backend boundary and release verification gates.
- Restored automated frontend and backend regression verification in GitHub Actions on the release candidate branch.

### Known release gates

- Durable PostgreSQL is still required for persistent production data.
- Production session hardening beyond browser session storage remains required.
- Real browser camera acceptance remains required.
- Signer-independent evaluation requires verified signer metadata.
