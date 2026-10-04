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
  const [username, setUsername] = useState(""); const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false); const [showConfirmPassword, setShowConfirmPassword] = useState(false); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const checks = passwordChecks(password);
  const strongPassword = isStrongPassword(password);
  useEffect(() => { const params = new URLSearchParams(location.search); const authError = params.get("auth_error"); if (authError) setError(authError.replaceAll("_", " ")); }, [location.search]);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError("");
    try {
      if (mode === "register") {
        if (!strongPassword) throw new Error("Please meet all password requirements before creating your account.");
        if (password !== confirmPassword) throw new Error("Passwords do not match.");
        await api.register(username, email, password);
      } else {
        await api.login(identifierMode === "email" ? email : username, password);
      }
      setSessionHint(); onAuthed(); navigate("/dashboard", { replace: true });
    } catch (err) { setError(err instanceof Error ? err.message : "Authentication failed"); }
    finally { setBusy(false); }
  };
  return <div className="auth-page"><Seo /><section className="auth-card">
    <div className="eyebrow">INDIAN SIGN LANGUAGE</div><h1>VisionBridge</h1>
    <p className="muted">A simple way to communicate with hand signs, speech, and everyday phrases.</p>
    <div className="auth-tabs"><button type="button" className={mode === "login" ? "auth-tab active" : "auth-tab"} onClick={() => setMode("login")}>Sign in</button><button type="button" className={mode === "register" ? "auth-tab active" : "auth-tab"} onClick={() => setMode("register")}>Create account</button></div>
    <form onSubmit={submit} className="stack">
      {mode === "register" ? <><label>Username<input value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" required /></label><label>Email<input value={email} onChange={e => setEmail(e.target.value)} type="email" autoComplete="email" required /></label></> : <><div className="identifier-toggle" role="group" aria-label="Login identifier"><button type="button" className={identifierMode === "username" ? "active" : ""} onClick={() => setIdentifierMode("username")}>Username</button><button type="button" className={identifierMode === "email" ? "active" : ""} onClick={() => setIdentifierMode("email")}>Email</button></div><label>{identifierMode === "email" ? "Email" : "Username"}<input value={identifierMode === "email" ? email : username} onChange={e => identifierMode === "email" ? setEmail(e.target.value) : setUsername(e.target.value)} type={identifierMode === "email" ? "email" : "text"} autoComplete={identifierMode === "email" ? "email" : "username"} required /></label></>}
      <label>Password<span className="password-field"><input value={password} onChange={e => setPassword(e.target.value)} type={showPassword ? "text" : "password"} minLength={8} autoComplete={mode === "login" ? "current-password" : "new-password"} required /><button type="button" onClick={() => setShowPassword(v => !v)}>{showPassword ? "Hide" : "Show"}</button></span></label>
      {mode === "register" && <div className="password-requirements" aria-live="polite">
        <div className={"password-strength " + passwordStrengthLabel(password).toLowerCase().replace(" ", "-")}>Password strength: <strong>{password ? passwordStrengthLabel(password) : "Not set"}</strong></div>
        <div className="password-rule-list">
          {PASSWORD_REQUIREMENTS.map((rule) => <div key={rule.id} className={checks[rule.id] ? "password-rule met" : "password-rule"}><span aria-hidden="true">{checks[rule.id] ? "✓" : "○"}</span>{rule.label}</div>)}
        </div>
      </div>}
      {mode === "register" && <label>Confirm password<span className="password-field"><input value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} type={showConfirmPassword ? "text" : "password"} minLength={8} autoComplete="new-password" required /><button type="button" onClick={() => setShowConfirmPassword(v => !v)}>{showConfirmPassword ? "Hide" : "Show"}</button></span></label>}
      {error && <div className="alert error" role="alert">{error}</div>}<button className="primary-btn auth-submit" disabled={busy || (mode === "register" && (!strongPassword || password !== confirmPassword))}>{busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}</button>
    </form>
  </section></div>;
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