"""Email delivery for VisionBridge verification codes via Resend."""
from __future__ import annotations

import html

import httpx

from app.core.config import get_settings

settings = get_settings()

RESEND_URL = "https://api.resend.com/emails"


class EmailDeliveryError(RuntimeError):
    """Raised when the configured email provider cannot deliver a message."""


def send_verification_email(email: str, otp: str) -> None:
    if not settings.RESEND_API_KEY or not settings.RESEND_FROM_EMAIL:
        raise EmailDeliveryError(
            "Email verification is not configured yet. Add RESEND_API_KEY and RESEND_FROM_EMAIL."
        )

    safe_otp = html.escape(otp)
    try:
        response = httpx.post(
            RESEND_URL,
        headers={
            "Authorization": "Bearer " + settings.RESEND_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "from": settings.RESEND_FROM_EMAIL,
            "to": [email],
            "subject": "Your VisionBridge verification code",
            "text": (
                "Your VisionBridge verification code is "
                + otp
                + ".\n\n"
                + f"This code expires in {settings.OTP_EXPIRY_MINUTES} minutes. "
                + "If you did not create this account, you can ignore this email."
            ),
            "html": (
                "<div style='font-family:Arial,sans-serif;max-width:520px;color:#16302d'>"
                "<h2 style='margin-bottom:8px'>Verify your VisionBridge account</h2>"
                "<p>Enter this 6-digit code in VisionBridge:</p>"
                f"<p style='font-size:32px;font-weight:700;letter-spacing:8px;margin:20px 0'>{safe_otp}</p>"
                f"<p>This code expires in {settings.OTP_EXPIRY_MINUTES} minutes.</p>"
                "<p style='color:#5f716e;font-size:13px'>If you did not create this account, you can ignore this email.</p>"
                "</div>"
            ),
        },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        raise EmailDeliveryError(
            "We could not reach the email service. Please try again shortly."
        ) from exc

    if response.is_error:
        raise EmailDeliveryError(
            "We could not send the verification email. Please try again shortly."
        )
