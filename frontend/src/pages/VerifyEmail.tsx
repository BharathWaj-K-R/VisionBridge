import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { api, setSessionHint } from "../api";
import Seo from "../components/Seo";
import { AppBootstrap, ErrorState, SuccessMessage } from "../components/SystemStates";

const RESEND_FALLBACK_SECONDS = 60;

export default function VerifyEmail() {
  const location = useLocation();
  const navigate = useNavigate();
  const emailFromQuery = new URLSearchParams(location.search).get("email") || "";
  const [email, setEmail] = useState(emailFromQuery);
  const [otp, setOtp] = useState("");
  const [cooldown, setCooldown] = useState(RESEND_FALLBACK_SECONDS);
  const [busy, setBusy] = useState(false);
  const [resending, setResending] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const otpRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    otpRef.current?.focus();
  }, []);

  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = window.setInterval(() => setCooldown((value) => Math.max(0, value - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [cooldown]);

  function updateOtp(value: string) {
    setOtp(value.replace(/\D/g, "").slice(0, 6));
    setError("");
  }

  async function verify(event: React.FormEvent) {
    event.preventDefault();
    if (!email.trim() || otp.length !== 6 || busy) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await api.verifyOtp(email.trim(), otp);
      setSessionHint();
      setMessage("Email verified. Welcome to VisionBridge.");
      window.setTimeout(() => navigate("/dashboard", { replace: true }), 350);
    } catch (err) {
      setError(err instanceof Error ? err.message : "We could not verify that code.");
    } finally {
      setBusy(false);
    }
  }

  async function resend() {
    if (!email.trim() || cooldown > 0 || resending) return;
    setResending(true);
    setError("");
    setMessage("");
    try {
      const result = await api.resendOtp(email.trim());
      setCooldown(result.resend_after_seconds || RESEND_FALLBACK_SECONDS);
      setMessage(result.message);
    } catch (err) {
      const apiError = err as Error & { status?: number };
      if (apiError.status === 429) {
        setCooldown(30);
      }
      setError(apiError.message || "We could not send a new code.");
    } finally {
      setResending(false);
    }
  }

  if (!email.trim()) {
    return (
      <main className="auth-page">
        <Seo />
        <section className="auth-card" aria-labelledby="verify-title">
          <div className="eyebrow">EMAIL VERIFICATION</div>
          <h1 id="verify-title">Verify your email</h1>
          <p className="muted">Enter the email you used when creating your VisionBridge account.</p>
          <div className="stack">
            <label>Email<input value={email} onChange={(event) => setEmail(event.target.value)} type="email" autoComplete="email" /></label>
            <button className="primary-btn" type="button" onClick={() => { if (email.trim()) otpRef.current?.focus(); }}>Continue</button>
            <Link className="text-btn" to="/login">Back to sign in</Link>
          </div>
          <input ref={otpRef} aria-hidden="true" tabIndex={-1} className="visually-hidden" />
        </section>
      </main>
    );
  }

  return (
    <main className="auth-page">
      <Seo />
      <section className="auth-card verification-card" aria-labelledby="verify-title">
        <div className="eyebrow">EMAIL VERIFICATION</div>
        <h1 id="verify-title">Check your email</h1>
        <p className="muted">We sent a 6-digit code to <strong>{email}</strong>.</p>

        <form onSubmit={verify} className="stack">
          <label>Verification code
            <input
              ref={otpRef}
              className="otp-input"
              value={otp}
              onChange={(event) => updateOtp(event.target.value)}
              inputMode="numeric"
              autoComplete="one-time-code"
              pattern="[0-9]*"
              maxLength={6}
              placeholder="000000"
              aria-describedby="otp-help"
            />
          </label>
          <p id="otp-help" className="verification-help">The code expires in 10 minutes. You can request another code after the cooldown.</p>

          {error && (
            <ErrorState
              title="Verification needs attention"
              message={error}
              actionLabel=""
              onRetry={undefined}
            />
          )}
          {message && <SuccessMessage>{message}</SuccessMessage>}

          <button className="primary-btn auth-submit" disabled={busy || otp.length !== 6}>
            {busy ? "Verifying…" : "Verify email"}
          </button>
          <button type="button" className="ghost-btn" onClick={() => void resend()} disabled={resending || cooldown > 0}>
            {resending ? "Sending…" : cooldown > 0 ? "Resend in " + cooldown + "s" : "Resend code"}
          </button>
          <Link className="text-btn" to="/login">Use a different account</Link>
        </form>
      </section>
    </main>
  );
}

void AppBootstrap;
