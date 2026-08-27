import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, clearToken, getToken, setToken } from "./api";
import { FRAME_WINDOW } from "./landmarks";
import { useLandmarkSession } from "./useLandmarkSession";

const navItems = [
  ["/dashboard", "Dashboard"],
  ["/translate", "Translate"],
  ["/calibration", "Calibration"],
  ["/history", "History"],
  ["/evaluation", "Evaluation"],
  ["/settings", "Settings"],
] as const;

function Shell({ children, username, onLogout }: { children: ReactNode; username?: string; onLogout: () => void }) {
  const location = useLocation();
  return <div className="app-shell"><aside className="sidebar">
    <Link to="/dashboard" className="brand"><span className="brand-mark">VB</span><span>VisionBridge</span></Link>
    <nav className="nav-list" aria-label="Primary navigation">
      {navItems.map(([path, label]) => <Link key={path} to={path} className={location.pathname.startsWith(path) ? "nav-link active" : "nav-link"}>{label}</Link>)}
    </nav>
    <div className="sidebar-foot"><div className="user-chip"><span className="status-dot" />{username || "Signed in"}</div><button className="ghost-btn" onClick={onLogout}>Log out</button></div>
  </aside><main className="main-pane">{children}</main></div>;
}

function Auth({ onAuthed }: { onAuthed: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState(""); const [password, setPassword] = useState(""); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError("");
    try { if (mode === "register") await api.register(username, password); const token = await api.login(username, password); setToken(token.access_token); onAuthed(); }
    catch (err) { setError(err instanceof Error ? err.message : "Authentication failed"); }
    finally { setBusy(false); }
  };
  return <div className="auth-page"><section className="auth-card"><div className="eyebrow">INDIAN SIGN LANGUAGE</div><h1>VisionBridge</h1><p className="muted">Real-time sign recognition, hand skeleton tracking, and signer personalization.</p><form onSubmit={submit} className="stack">
    <label>Username<input value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" required /></label>
    <label>Password<input value={password} onChange={e => setPassword(e.target.value)} type="password" minLength={8} autoComplete={mode === "login" ? "current-password" : "new-password"} required /></label>
    {error && <div className="alert error">{error}</div>}<button className="primary-btn" disabled={busy}>{busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}</button>
  </form><button className="text-btn" onClick={() => setMode(mode === "login" ? "register" : "login")}>{mode === "login" ? "Create an account" : "Back to sign in"}</button></section></div>;
}

function Page({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return <div className="page"><header className="page-header"><div><div className="eyebrow">VISIONBRIDGE</div><h1>{title}</h1><p className="muted">{subtitle}</p></div></header>{children}</div>;
}
function Loading() { return <div className="loading">Loading…</div>; }
function Empty({ text }: { text: string }) { return <div className="empty">{text}</div>; }
function Metric({ label, value, detail }: { label: string; value: string | number; detail: string }) { return <div className="metric"><span className="eyebrow">{label}</span><strong>{value}</strong><span className="muted">{detail}</span></div>; }

function Dashboard() {
  const [data, setData] = useState<any>(null); const [error, setError] = useState("");
  useEffect(() => { api.dashboard().then(setData).catch(e => setError(e.message)); }, []);
  return <Page title="Dashboard" subtitle="A quiet view of model readiness, usage, and recent activity.">
    {error ? <div className="alert error">{error}</div> : !data ? <Loading /> : <>
      <div className="metric-grid"><Metric label="Model" value={data.model?.status || "unknown"} detail={data.model?.modality || "—"} /><Metric label="Translations" value={data.usage?.translation_events ?? 0} detail="stored events" /><Metric label="Confidence" value={data.usage?.average_confidence != null ? `${Math.round(data.usage.average_confidence * 100)}%` : "—"} detail="average" /><Metric label="Latency" value={data.usage?.average_latency_ms != null ? `${Math.round(data.usage.average_latency_ms)} ms` : "—"} detail="average" /></div>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">RECENT</div><h2>Translation activity</h2></div><Link to="/history" className="text-btn">View history</Link></div>{data.recent_activity?.length ? <div className="activity-list">{data.recent_activity.map((item: any) => <div className="activity-row" key={item.id}><div><strong>{item.predicted_text || "(no sign detected)"}</strong><span>{new Date(item.created_at).toLocaleString()}</span></div><span className="activity-meta">{Math.round((item.confidence || 0) * 100)}% · {Math.round(item.latency_ms || 0)} ms</span></div>)}</div> : <Empty text="No translation events yet." />}</section>
    </>}
  </Page>;
}

function Translate() {
  const session = useLandmarkSession(15); const [prediction, setPrediction] = useState("—"); const [confidence, setConfidence] = useState(0); const [latency, setLatency] = useState<number | null>(null); const [error, setError] = useState(""); const [sending, setSending] = useState(false); const [userId, setUserId] = useState<number>(); const [adapters, setAdapters] = useState<any[]>([]); const [adapterId, setAdapterId] = useState<number | undefined>();
  useEffect(() => { api.me().then(u => setUserId(u.id)); api.adapters().then(setAdapters).catch(() => setAdapters([])); }, []);
  useEffect(() => { const timer = window.setInterval(async () => { if (sending) return; const frames = session.snapshot(); if (frames.length < FRAME_WINDOW) return; const batch = frames.slice(-FRAME_WINDOW); session.clear(); setSending(true); const started = performance.now(); try { const result = await api.translate({ user_id: userId, adapter_id: adapterId, pose_keypoints: batch.map(f => f.pose), face_keypoints: batch.map(f => f.face), left_hand_keypoints: batch.map(f => f.leftHand), right_hand_keypoints: batch.map(f => f.rightHand) }); setPrediction(result.predicted_text); setConfidence(result.confidence || 0); setLatency(result.latency_ms); setError(""); } catch (err) { setError(err instanceof Error ? err.message : "Translation failed"); } finally { setSending(false); if (!latency) setLatency(performance.now() - started); } }, 850); return () => window.clearInterval(timer); }, [adapterId, latency, sending, session, userId]);
  return <Page title="Translate" subtitle="Live browser extraction with visible hand skeletons and synchronized server inference."><div className="translate-grid"><section className="panel camera-panel"><div className="camera-shell"><video ref={session.videoRef} muted playsInline /><canvas ref={session.canvasRef} className="skeleton-overlay" /><div className="camera-meta"><span>{session.status}</span><span>{session.fps} fps</span><span>{latency ? `${Math.round(latency)} ms` : "—"}</span></div></div><div className="button-row"><button className="primary-btn" onClick={() => session.start().catch(() => undefined)} disabled={session.running}>{session.running ? "Running" : "Start camera"}</button><button className="ghost-btn" onClick={session.stop} disabled={!session.running}>Stop</button></div><div className="selector-row"><label>Signer adapter<select value={adapterId ?? ""} onChange={e => setAdapterId(e.target.value ? Number(e.target.value) : undefined)}><option value="">Base model</option>{adapters.map(a => <option key={a.id} value={a.id}>Adapter #{a.id}</option>)}</select></label></div>{error && <div className="alert error">{error}</div>}<div className="hand-legend"><span><i className="legend-mark" /> Left hand</span><span><i className="legend-mark second" /> Right hand</span><span className="muted">21 landmarks per hand</span></div></section><section className="panel output-panel"><div className="panel-head"><div><div className="eyebrow">OUTPUT</div><h2>Translation</h2></div><span className="confidence">{Math.round(confidence * 100)}%</span></div><div className="translation-text">{prediction}</div><div className="progress"><span style={{ width: `${Math.round(confidence * 100)}%` }} /></div><div className="output-meta"><span>Pose 132</span><span>Face 1404</span><span>Left hand 63</span><span>Right hand 63</span></div><div className="alert">{sending ? "Inference request in flight…" : "Hands are tracked as 21-point skeletons per hand."}</div></section></div></Page>;
}

function Calibration() {
  const session = useLandmarkSession(3); const [startedAt, setStartedAt] = useState<number | null>(null); const [seconds, setSeconds] = useState(0); const [target, setTarget] = useState("i am hungry"); const [capturedFrames, setCapturedFrames] = useState<any[]>([]); const [busy, setBusy] = useState(false); const [message, setMessage] = useState("Record the target sentence naturally for five minutes. The browser samples synchronized hand, pose, and face landmarks at a bounded rate.");
  useEffect(() => { if (!startedAt) return; const timer = window.setInterval(() => { const value = Math.floor((Date.now() - startedAt) / 1000); setSeconds(value); if (value >= 300) { const frames = session.snapshot(); session.stop(); setCapturedFrames(frames); setStartedAt(null); setMessage(`Capture complete: ${frames.length} synchronized frames ready for adapter fitting.`); } }, 1000); return () => window.clearInterval(timer); }, [session, startedAt]);
  const begin = async () => { setCapturedFrames([]); setSeconds(0); setMessage("Starting camera…"); await session.start(); setStartedAt(Date.now()); setMessage("Recording. Repeat the target sentence with your normal signing style."); };
  const submit = async () => { const user = await api.me(); if (capturedFrames.length === 0 || seconds < 300) return; setBusy(true); setMessage("Submitting calibration…"); try { const result = await api.calibrate({ user_id: user.id, calibration_seconds: Math.max(300, seconds), target_text: target, pose_keypoints: capturedFrames.map(f => f.pose), face_keypoints: capturedFrames.map(f => f.face), left_hand_keypoints: capturedFrames.map(f => f.leftHand), right_hand_keypoints: capturedFrames.map(f => f.rightHand) }); setMessage(`Adapter #${result.adapter_id} created successfully.`); setCapturedFrames([]); } catch (err) { setMessage(err instanceof Error ? err.message : "Calibration failed"); } finally { setBusy(false); } };
  return <Page title="Calibration" subtitle="Create a lightweight signer adapter from real multimodal examples."><section className="panel calibration-panel"><div className="camera-shell small"><video ref={session.videoRef} muted playsInline /><canvas ref={session.canvasRef} className="skeleton-overlay" /></div><div className="calibration-controls"><label>Target sentence<input value={target} onChange={e => setTarget(e.target.value)} disabled={Boolean(startedAt) || busy} /></label><div className="calibration-timer">{String(Math.floor(seconds / 60)).padStart(2, "0")}:{String(seconds % 60).padStart(2, "0")}</div><div className="button-row"><button className="primary-btn" onClick={begin} disabled={Boolean(startedAt) || busy}>{startedAt ? "Recording…" : "Start 5-minute capture"}</button><button className="ghost-btn" onClick={session.stop} disabled={!session.running || busy}>Stop early</button><button className="ghost-btn" onClick={submit} disabled={capturedFrames.length === 0 || seconds < 300 || busy}>Fit adapter</button></div>{message && <div className="alert">{message}</div>}</div></section></Page>;
}

function History() { const [data, setData] = useState<any>(); const [query, setQuery] = useState(""); useEffect(() => { api.history("range=all&sort=newest").then(setData).catch(() => setData({ items: [] })); }, []); const rows = data?.items?.filter((x: any) => !query || String(x.predicted_text).toLowerCase().includes(query.toLowerCase())) || []; return <Page title="History" subtitle="Searchable translation events persisted for your account."><section className="panel"><div className="panel-head"><div><div className="eyebrow">EVENTS</div><h2>History</h2></div><input className="compact-input" placeholder="Search translations" value={query} onChange={e => setQuery(e.target.value)} /></div>{!data ? <Loading /> : rows.length ? <div className="table-wrap"><table><thead><tr><th>Time</th><th>Prediction</th><th>Confidence</th><th>Latency</th></tr></thead><tbody>{rows.map((row: any) => <tr key={row.id}><td>{new Date(row.created_at).toLocaleString()}</td><td>{row.predicted_text || "(no sign detected)"}</td><td>{Math.round((row.confidence || 0) * 100)}%</td><td>{Math.round(row.latency_ms || 0)} ms</td></tr>)}</tbody></table></div> : <Empty text="No matching events." />}</section></Page>; }

function Evaluation() { const [data, setData] = useState<any>(); useEffect(() => { api.evaluation().then(setData); }, []); return <Page title="Evaluation" subtitle="Only measured model evidence is shown here. No decorative fiction masquerading as science."><section className="panel narrow">{!data ? <Loading /> : <><div className="eyebrow">MODEL</div><h2>{data.model?.status || "unknown"}</h2><p className="muted">{data.message || "No benchmark run is persisted."}</p><div className="metric-grid compact"><Metric label="Benchmark" value={data.evaluation_data_available ? "available" : "not measured"} detail="persisted evidence" /><Metric label="Adapters" value={data.measured_adapter_count ?? 0} detail="measured gains" /></div></>}</section></Page>; }

function Settings() { const [user, setUser] = useState<any>(); const [adapters, setAdapters] = useState<any[]>([]); useEffect(() => { Promise.all([api.me(), api.adapters()]).then(([u,a]) => { setUser(u); setAdapters(a); }); }, []); return <Page title="Settings" subtitle="Account details and signer adapter lifecycle."><section className="panel narrow"><div className="eyebrow">ACCOUNT</div><h2>{user?.username || "Loading…"}</h2><p className="muted">Account ID {user?.id ?? "—"}</p></section><section className="panel narrow"><div className="eyebrow">SIGNER ADAPTERS</div><h2>Your adapters</h2>{adapters.length ? <div className="adapter-list">{adapters.map(a => <div className="adapter-row" key={a.id}><div><strong>Adapter #{a.id}</strong><span>{a.calibration_seconds}s · {a.param_count ?? 0} parameters</span></div><button className="ghost-btn" onClick={() => api.deleteAdapter(a.id).then(() => setAdapters(current => current.filter(x => x.id !== a.id)))}>Delete</button></div>)}</div> : <Empty text="No signer adapters yet." />}</section></Page>; }

export default function App() {
  const [authed, setAuthed] = useState(Boolean(getToken())); const [username, setUsername] = useState<string>(); const navigate = useNavigate();
  useEffect(() => { if (authed) api.me().then(user => setUsername(user.username)).catch(() => { clearToken(); setAuthed(false); }); }, [authed]);
  const logout = () => { clearToken(); setAuthed(false); navigate("/login"); };
  if (!authed) return <Routes><Route path="*" element={<Auth onAuthed={() => setAuthed(true)} />} /></Routes>;
  return <Shell username={username} onLogout={logout}><Routes><Route path="/" element={<Navigate to="/dashboard" replace />} /><Route path="/dashboard" element={<Dashboard />} /><Route path="/translate" element={<Translate />} /><Route path="/calibration" element={<Calibration />} /><Route path="/history" element={<History />} /><Route path="/evaluation" element={<Evaluation />} /><Route path="/settings" element={<Settings />} /><Route path="*" element={<Navigate to="/dashboard" replace />} /></Routes></Shell>;
}
