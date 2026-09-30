import { lazy, Suspense, useEffect, useState, type FormEvent, type ReactNode } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, clearLocalAuth, clearSessionHint, hasSessionHint, setSessionHint, isLocalAuthenticated } from "./api";
import NotFound from "./NotFound";
import Seo from "./components/Seo";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Recognize = lazy(() => import("./pages/Recognize"));
const Calibration = lazy(() => import("./pages/Calibration"));
const History = lazy(() => import("./pages/History"));
const Settings = lazy(() => import("./pages/Settings"));

const navItems = [
  ["/dashboard", "Dashboard"],
  ["/translate", "Live Translate"],
  ["/calibration", "Few-Shot Calibration"],
  ["/history", "Letter History"],
  ["/settings", "Signer Profiles"],
] as const;

function Shell({ children, username, onLogout }: { children: ReactNode; username?: string; onLogout: () => void }) {
  const location = useLocation();
  return <div className="app-shell"><Seo />
    <header className="topbar">
      <Link to="/dashboard" className="brand-lockup" aria-label="VisionBridge dashboard"><span className="brand-mark">V</span><span className="brand-copy"><strong>VisionBridge</strong><small>ISL RECOGNIZER</small></span></Link>
      <div className="header-badges"><span className="status-chip"><i /> Base Model · 26 A–Z</span><span className="status-chip"><i /> Few-Shot Adapter · Optional</span></div>
      <nav className="topnav" aria-label="Primary navigation">{navItems.map(([path, label]) => <Link key={path} to={path} className={location.pathname.startsWith(path) ? "topnav-link active" : "topnav-link"}>{label}</Link>)}</nav>
      <div className="top-actions"><Link className="icon-btn" to="/dashboard" aria-label="Dashboard" title="Dashboard">⌂</Link><span className="user-avatar" title={username || "Signed in"}>{(username || "U").slice(0, 1).toUpperCase()}</span><button className="icon-btn" onClick={onLogout} aria-label="Log out" title="Log out">↗</button></div>
    </header>
    <div className="runtime-strip"><div><span className="engine-status"><i /> ENGINE {import.meta.env.VITE_LOCAL_MODE !== "false" ? "DEMO" : "ONLINE"}</span><span>DEVICE: <b>Browser Camera</b></span><span>•</span><span>PIPELINE: <b>MediaPipe Tasks · 126D</b></span><span>•</span><span>TRACKING: <b>2 HANDS</b></span></div><div><span>USER: <b>{username || "Signed in"}</b></span><span className="mode-tag">{import.meta.env.VITE_LOCAL_MODE !== "false" ? "LOCAL RUNTIME" : "BROWSER INFERENCE"}</span></div></div>
    <main id="main-content" className="main-pane">{children}</main>
  </div>;
}

function Auth({ onAuthed }: { onAuthed: () => void }) {
  const location = useLocation();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [identifierMode, setIdentifierMode] = useState<"username" | "email">("username");
  const [username, setUsername] = useState(""); const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false); const [showConfirmPassword, setShowConfirmPassword] = useState(false); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  useEffect(() => { const params = new URLSearchParams(location.search); const authError = params.get("auth_error"); if (authError) setError(authError.replaceAll("_", " ")); }, [location.search]);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError("");
    try {
      if (mode === "register") { if (password !== confirmPassword) throw new Error("Passwords do not match."); await api.register(username, email, password); await api.login(username, password); }
      else await api.login(identifierMode === "email" ? email : username, password);
      setSessionHint(); onAuthed(); navigate("/dashboard", { replace: true });
    } catch (err) { setError(err instanceof Error ? err.message : "Authentication failed"); }
    finally { setBusy(false); }
  };
  return <div className="auth-page"><Seo /><section className="auth-card">
    <div className="eyebrow">INDIAN SIGN LANGUAGE · LETTERS</div><h1>VisionBridge</h1>
    <p className="muted">Signer-adaptive fingerspelling recognition from a few hand examples.</p>
    <div className="auth-tabs"><button type="button" className={mode === "login" ? "auth-tab active" : "auth-tab"} onClick={() => setMode("login")}>Sign in</button><button type="button" className={mode === "register" ? "auth-tab active" : "auth-tab"} onClick={() => setMode("register")}>Create account</button></div>
    <form onSubmit={submit} className="stack">
      {mode === "register" ? <><label>Username<input value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" required /></label><label>Email<input value={email} onChange={e => setEmail(e.target.value)} type="email" autoComplete="email" required /></label></> : <><div className="identifier-toggle" role="group" aria-label="Login identifier"><button type="button" className={identifierMode === "username" ? "active" : ""} onClick={() => setIdentifierMode("username")}>Username</button><button type="button" className={identifierMode === "email" ? "active" : ""} onClick={() => setIdentifierMode("email")}>Email</button></div><label>{identifierMode === "email" ? "Email" : "Username"}<input value={identifierMode === "email" ? email : username} onChange={e => identifierMode === "email" ? setEmail(e.target.value) : setUsername(e.target.value)} type={identifierMode === "email" ? "email" : "text"} autoComplete={identifierMode === "email" ? "email" : "username"} required /></label></>}
      <label>Password<span className="password-field"><input value={password} onChange={e => setPassword(e.target.value)} type={showPassword ? "text" : "password"} minLength={8} autoComplete={mode === "login" ? "current-password" : "new-password"} required /><button type="button" onClick={() => setShowPassword(v => !v)}>{showPassword ? "Hide" : "Show"}</button></span></label>
      {mode === "register" && <label>Confirm password<span className="password-field"><input value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} type={showConfirmPassword ? "text" : "password"} minLength={8} autoComplete="new-password" required /><button type="button" onClick={() => setShowConfirmPassword(v => !v)}>{showConfirmPassword ? "Hide" : "Show"}</button></span></label>}
      {error && <div className="alert error" role="alert">{error}</div>}<button className="primary-btn auth-submit" disabled={busy}>{busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}</button>
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
    if (localMode) { if (authed) api.me().then(user => setUsername(user.username)).catch(() => { clearLocalAuth(); setAuthed(false); }); return; }
    if (!hasSessionHint()) { setAuthChecking(false); return; }
    api.me().then(user => { setUsername(user.username); setAuthed(true); }).catch(() => { clearSessionHint(); setAuthed(false); }).finally(() => setAuthChecking(false));
  }, [authed, localMode]);
  const logout = () => void api.logout().then(() => { clearLocalAuth(); clearSessionHint(); setUsername(undefined); setAuthed(false); navigate("/login"); });
  if (authChecking) return <><Seo /><LoadingFallback /></>;
  if (!authed) return <Routes><Route path="/login" element={<Auth onAuthed={() => setAuthed(true)} />} /><Route path="*" element={<Auth onAuthed={() => setAuthed(true)} />} /></Routes>;
  return <Shell username={username} onLogout={logout}><Suspense fallback={<LoadingFallback />}><Routes>
    <Route path="/" element={<Navigate to="/dashboard" replace />} />
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/translate" element={<Recognize />} />
    <Route path="/calibration" element={<Calibration />} />
    <Route path="/history" element={<History />} />
    <Route path="/settings" element={<Settings />} />
    <Route path="*" element={<NotFound />} />
  </Routes></Suspense></Shell>;
}
function LoadingFallback() { return <main className="loading-page" role="status" aria-live="polite"><div className="loading">Loading workspace…</div></main>; }