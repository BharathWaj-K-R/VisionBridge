import { lazy, Suspense, useEffect, useState, type FormEvent, type ReactNode } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, clearLocalAuth, clearSessionHint, hasSessionHint, setSessionHint, isLocalAuthenticated } from "./api";
import NotFound from "./NotFound";
import Seo from "./components/Seo";
import QuickAccessBar from "./components/QuickAccessBar";
import { PersonalizationProvider } from "./components/PersonalizationContext";
import ProfileSwitcher from "./components/ProfileSwitcher";
import { QuickAccessProvider } from "./components/QuickAccessContext";
import { OfflineBanner } from "./components/SystemStates";
import { PASSWORD_REQUIREMENTS, isStrongPassword, passwordChecks, passwordStrengthLabel } from "./auth/password";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Recognize = lazy(() => import("./pages/Recognize"));
const Calibration = lazy(() => import("./pages/Calibration"));
const History = lazy(() => import("./pages/History"));
const Settings = lazy(() => import("./pages/Settings"));
const VoiceToSign = lazy(() => import("./pages/VoiceToSign"));
const WordBank = lazy(() => import("./pages/WordBank"));
const Personalization = lazy(() => import("./pages/Personalization"));

const navItems = [
  ["/translate", "Translate"],
  ["/voice-to-sign", "Speak"],
  ["/word-bank", "Words"],
  ["/history", "History"],
  ["/settings", "Settings"],
] as const;

function Shell({ children, username, onLogout }: { children: ReactNode; username?: string; onLogout: () => void }) {
  const location = useLocation();
  return <PersonalizationProvider><QuickAccessProvider><div className="app-shell"><Seo />
    <header className="topbar">
      <Link to="/dashboard" className="brand-lockup" aria-label="VisionBridge home"><span className="brand-mark">V</span><span className="brand-copy"><strong>VisionBridge</strong><small>COMMUNICATION TOOL</small></span></Link>
      <nav className="topnav" aria-label="Primary navigation">{navItems.map(([path, label]) => <Link key={path} to={path} className={location.pathname.startsWith(path) ? "topnav-link active" : "topnav-link"}>{label}</Link>)}</nav>
      <div className="top-actions"><ProfileSwitcher /><Link className="icon-btn" to="/settings" aria-label="Settings" title="Settings"><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 3.5a2.2 2.2 0 0 1 4.2 1.2l1.4.8a2.2 2.2 0 0 1 2.9 2.9l-.8 1.4a2.2 2.2 0 0 1-1.2 4.2l-.8 1.4a2.2 2.2 0 0 1-2.9 2.9l-1.4-.8a2.2 2.2 0 0 1-4.2 1.2l-1.4-.8a2.2 2.2 0 0 1-2.9-2.9l.8-1.4a2.2 2.2 0 0 1 1.2-4.2l.8-1.4A2.2 2.2 0 0 1 8.3 4.7l1.4.8A2.2 2.2 0 0 1 12 3.5Z" /><circle cx="12" cy="12" r="3.2" /></svg></Link><span className="user-avatar" title={username || "Signed in"} aria-label={username || "Signed in"}>{(username || "U").slice(0, 1).toUpperCase()}</span><button className="icon-btn" onClick={onLogout} aria-label="Sign out" title="Sign out"><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M10 6H5.5A1.5 1.5 0 0 0 4 7.5v9A1.5 1.5 0 0 0 5.5 18H10" /><path d="M13 8l4 4-4 4M17 12H8" /></svg></button></div>
    </header>
    <QuickAccessBar />
    <OfflineBanner />
    <nav className="mobile-primary-nav" aria-label="Primary navigation">{navItems.map(([path, label]) => <Link key={path} to={path} className={location.pathname.startsWith(path) ? "active" : ""}>{label}</Link>)}</nav>
    <main id="main-content" className="main-pane">{children}</main>
  </div></QuickAccessProvider></PersonalizationProvider>;
}

function Auth({ onAuthed }: { onAuthed: () => void }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [identifierMode, setIdentifierMode] = useState<"username" | "email">("username");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const checks = passwordChecks(password);
  const strongPassword = isStrongPassword(password);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const authError = params.get("auth_error");
    if (authError) setError(authError.replaceAll("_", " "));
  }, [location.search]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError("");

    try {
      if (mode === "register") {
        const cleanUsername = username.trim();
        const cleanEmail = email.trim().toLowerCase();

        if (!/^[A-Za-z0-9_.-]{1,128}$/.test(cleanUsername)) {
          throw new Error("Use 1–128 characters for your username: letters, numbers, dot, underscore, or hyphen.");
        }
        if (!cleanEmail || cleanEmail.length > 320 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail)) {
          throw new Error("Enter a valid email address.");
        }
        if (password.length < 8 || password.length > 72 || !strongPassword) {
          throw new Error("Please meet all password requirements before creating your account.");
        }
        if (password !== confirmPassword) {
          throw new Error("Passwords do not match.");
        }

        await api.register(cleanUsername, cleanEmail, password);
      } else {
        const identifier = (identifierMode === "email" ? email : username).trim();
        if (!identifier) throw new Error("Enter your username or email.");
        if (password.length < 8 || password.length > 72) {
          throw new Error("Your password must be between 8 and 72 characters.");
        }
        await api.login(identifier, password);
      }

      setSessionHint();
      onAuthed();
      navigate("/dashboard", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth-page">
      <Seo />
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="auth-intro">
          <div className="auth-intro-top">
            <div className="auth-brand-row">
              <span className="brand-mark" aria-hidden="true">V</span>
              <div>
                <div className="eyebrow">VISIONBRIDGE</div>
                <strong>COMMUNICATION TOOL</strong>
              </div>
            </div>
            <span className="auth-status"><i aria-hidden="true" /> SYSTEM READY</span>
          </div>

          <div className="auth-intro-copy">
            <div className="eyebrow">INDIAN SIGN LANGUAGE</div>
            <h1 id="auth-title">Your signs.<br />Your workspace.</h1>
            <p>Recognize A–Z letters, calibrate a signer profile, and keep your everyday communication tools in one place.</p>
          </div>

          <div className="auth-contract" aria-label="VisionBridge capabilities">
            <div><span>01</span><strong>A–Z LETTERS</strong><small>Live hand-sign recognition</small></div>
            <div><span>02</span><strong>SIGNER ADAPTATION</strong><small>Personal calibration profiles</small></div>
            <div><span>03</span><strong>COMMUNICATION</strong><small>Words, history, and quick access</small></div>
          </div>

          <div className="auth-footer-note">
            <span>126D LANDMARK INPUT</span>
            <span>·</span>
            <span>LOCAL CAMERA INFERENCE</span>
          </div>
        </div>

        <div className="auth-form-panel">
          <div className="auth-form-header">
            <div>
              <div className="eyebrow">{mode === "login" ? "RETURNING USER" : "NEW WORKSPACE"}</div>
              <h2>{mode === "login" ? "Sign in" : "Create account"}</h2>
            </div>
            <span className="auth-step">{mode === "login" ? "01 / 02" : "02 / 02"}</span>
          </div>

          <div className="auth-tabs" role="tablist" aria-label="Authentication mode">
            <button
              type="button"
              role="tab"
              aria-selected={mode === "login"}
              className={mode === "login" ? "auth-tab active" : "auth-tab"}
              onClick={() => { setMode("login"); setError(""); }}
            >
              Sign in
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "register"}
              className={mode === "register" ? "auth-tab active" : "auth-tab"}
              onClick={() => { setMode("register"); setError(""); }}
            >
              Create account
            </button>
          </div>

          <form onSubmit={submit} className="stack auth-form" noValidate>
            {mode === "register" ? (
              <>
                <label>
                  Username
                  <input
                    value={username}
                    onChange={e => setUsername(e.target.value)}
                    autoComplete="username"
                    minLength={1}
                    maxLength={128}
                    pattern="^[A-Za-z0-9_.-]+$"
                    aria-describedby="username-help"
                    required
                  />
                  <small id="username-help" className="field-hint">Letters, numbers, dot, underscore, and hyphen.</small>
                </label>

                <label>
                  Email
                  <input
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    type="email"
                    autoComplete="email"
                    maxLength={320}
                    required
                  />
                </label>
              </>
            ) : (
              <>
                <div className="identifier-toggle" role="group" aria-label="Login identifier">
                  <button
                    type="button"
                    className={identifierMode === "username" ? "active" : ""}
                    onClick={() => setIdentifierMode("username")}
                  >
                    Username
                  </button>
                  <button
                    type="button"
                    className={identifierMode === "email" ? "active" : ""}
                    onClick={() => setIdentifierMode("email")}
                  >
                    Email
                  </button>
                </div>
                <label>
                  {identifierMode === "email" ? "Email" : "Username"}
                  <input
                    value={identifierMode === "email" ? email : username}
                    onChange={e => identifierMode === "email" ? setEmail(e.target.value) : setUsername(e.target.value)}
                    type={identifierMode === "email" ? "email" : "text"}
                    autoComplete={identifierMode === "email" ? "email" : "username"}
                    maxLength={identifierMode === "email" ? 320 : 128}
                    required
                  />
                </label>
              </>
            )}

            <label>
              Password
              <span className="password-field">
                <input
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  type={showPassword ? "text" : "password"}
                  minLength={8}
                  maxLength={72}
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  required
                />
                <button type="button" onClick={() => setShowPassword(v => !v)} aria-label={showPassword ? "Hide password" : "Show password"}>
                  {showPassword ? "Hide" : "Show"}
                </button>
              </span>
            </label>

            {mode === "register" ? (
              <div className="password-requirements" aria-live="polite">
                <div className={"password-strength " + passwordStrengthLabel(password).toLowerCase().replace(" ", "-")}>
                  Password strength: <strong>{password ? passwordStrengthLabel(password) : "Not set"}</strong>
                </div>
                <div className="password-rule-list">
                  {PASSWORD_REQUIREMENTS.map(rule => (
                    <div key={rule.id} className={checks[rule.id] ? "password-rule met" : "password-rule"}>
                      <span aria-hidden="true">{checks[rule.id] ? "✓" : "○"}</span>
                      {rule.label}
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {mode === "register" ? (
              <label>
                Confirm password
                <span className="password-field">
                  <input
                    value={confirmPassword}
                    onChange={e => setConfirmPassword(e.target.value)}
                    type={showConfirmPassword ? "text" : "password"}
                    minLength={8}
                    maxLength={72}
                    autoComplete="new-password"
                    required
                  />
                  <button type="button" onClick={() => setShowConfirmPassword(v => !v)} aria-label={showConfirmPassword ? "Hide confirmation password" : "Show confirmation password"}>
                    {showConfirmPassword ? "Hide" : "Show"}
                  </button>
                </span>
              </label>
            ) : null}

            {error ? <div className="alert error auth-error" role="alert">{error}</div> : null}

            <button
              className="primary-btn auth-submit"
              disabled={busy || (mode === "register" && (!strongPassword || password !== confirmPassword))}
            >
              <span>{busy ? "Working…" : mode === "login" ? "Sign in to workspace" : "Create workspace"}</span>
              <span aria-hidden="true">↗</span>
            </button>
          </form>

          <div className="auth-form-foot">
            <span>{mode === "login" ? "Use your VisionBridge account." : "Your account unlocks the full workspace."}</span>
            <span className="mono">SECURE SESSION</span>
          </div>
        </div>
      </section>
    </div>
  );
}
export default function App() {
  const localMode = import.meta.env.VITE_LOCAL_MODE !== "false";
  const [authed, setAuthed] = useState(localMode ? isLocalAuthenticated() : hasSessionHint());
  const [authChecking, setAuthChecking] = useState(!localMode && hasSessionHint());
  const [username, setUsername] = useState<string>();
  const navigate = useNavigate();
  useEffect(() => {
    if (localMode) { if (authed) api.me().then(user => setUsername(user.username)).catch(() => { clearLocalAuth(); setAuthed(false); navigate("/login?auth_error=session_expired", { replace: true }); }); return; }
    if (!hasSessionHint()) { setAuthChecking(false); return; }
    api.me().then(user => { setUsername(user.username); setAuthed(true); }).catch(() => { clearSessionHint(); setAuthed(false); navigate("/login?auth_error=session_expired", { replace: true }); }).finally(() => setAuthChecking(false));
  }, [authed, localMode]);
  const logout = () => void api.logout().then(() => {
    clearLocalAuth(); clearSessionHint(); setUsername(undefined); setAuthed(false); navigate("/login");
  }).catch(() => {
    clearSessionHint(); setUsername(undefined); setAuthed(false); navigate("/login");
  });
  if (authChecking) return <><Seo /><LoadingFallback /></>;
  if (!authed) return <Routes>
    <Route path="/" element={<Navigate to="/login" replace />} />
    <Route path="/login" element={<Auth onAuthed={() => setAuthed(true)} />} />
    <Route path="*" element={<Auth onAuthed={() => setAuthed(true)} />} />
  </Routes>;
  return <Shell username={username} onLogout={logout}><Suspense fallback={<LoadingFallback />}><Routes>
    <Route path="/" element={<Navigate to="/dashboard" replace />} />
    <Route path="/login" element={<Navigate to="/dashboard" replace />} />
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/translate" element={<Recognize />} />
    <Route path="/calibration" element={<Calibration />} />
    <Route path="/history" element={<History />} />
    <Route path="/settings" element={<Settings />} />
    <Route path="/voice-to-sign" element={<VoiceToSign />} />
    <Route path="/word-bank" element={<WordBank />} />
    <Route path="/personalization" element={<Personalization />} />
    <Route path="*" element={<NotFound />} />
  </Routes></Suspense></Shell>;
}
function LoadingFallback() { return <main className="loading-page" role="status" aria-live="polite"><div className="loading">Loading workspace…</div></main>; }