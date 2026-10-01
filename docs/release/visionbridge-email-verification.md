# VisionBridge Email Verification

## Runtime configuration

Set these environment variables on the FastAPI service:

- `RESEND_API_KEY`: a Resend API key with permission to send email.
- `RESEND_FROM_EMAIL`: a sender address in the form `VisionBridge <verify@your-domain.example>`. The sender/domain must be configured for sending in Resend.

Security defaults:
- OTP lifetime: 10 minutes
- Resend cooldown: 60 seconds
- Maximum OTP verification attempts: 5
- Resend rate limit: 5 requests per hour, plus an IP/email bucket
- Registration rate limit: 5 requests per minute
- OTP verification rate limit: 10 requests per minute

## Registration flow

1. The API validates the password against all five strong-password rules.
2. The account is created with `is_verified=false`.
3. A cryptographically secure six-digit OTP is generated.
4. Only an HMAC-SHA256 OTP digest is stored.
5. The OTP is sent through Resend.
6. The browser moves to `/verify-email`.
7. Successful verification clears OTP fields, marks the account verified, and sets the existing HttpOnly session cookie.

The registration endpoint deliberately does not create a session before verification.

## Existing accounts

The migration backfills existing users as verified before changing the column default to false. This prevents an existing production account from being unexpectedly locked out by the new verification requirement.

## Local development

Local demo mode does not send email and treats local accounts as already verified. The production/API path uses the real OTP flow.

## Validation

Backend coverage includes strong-password rejection, unverified registration, hashed OTP storage, wrong-code attempt counting, expiry, successful verification and session creation, unverified login blocking, and resend cooldown.
