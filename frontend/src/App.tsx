import { useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, clearToken, getToken, setToken, type LetterSample } from "./api";
import { useLandmarkSession } from "./useLandmarkSession";

const navItems = [
  ["/dashboard", "Dashboard"],
  ["/translate", "Recognize"],
  ["/calibration", "Calibrate"],
  ["/history", "History"],
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
  return <div className="auth-page"><section className="auth-card"><div className="eyebrow">INDIAN SIGN LANGUAGE · LETTERS</div><h1>VisionBridge</h1><p className="muted">Signer-adaptive fingerspelling recognition from a few hand examples.</p><form onSubmit={submit} className="stack">
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
  useEffect(() => { api.dashboard().then(setData).catch((e) => setError(e.message)); }, []);
  return <Page title="Letter recognition" subtitle="Hand-only, signer-adaptive recognition. The sentence decoder has finally been evicted.">
    {error ? <div className="alert error">{error}</div> : !data ? <Loading /> : <>
      <div className="metric-grid">
        <Metric label="Mode" value="Letters" detail="hand-only" />
        <Metric label="Events" value={data.usage?.translation_events ?? 0} detail="recognition events" />
        <Metric label="Confidence" value={data.usage?.average_confidence != null ? Math.round(data.usage.average_confidence * 100) + "%" : "—"} detail="recent average" />
        <Metric label="Latency" value={data.usage?.average_latency_ms != null ? Math.round(data.usage.average_latency_ms) + " ms" : "—"} detail="recent average" />
      </div>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">QUICK START</div><h2>Calibrate → Recognize</h2></div><Link to="/calibration" className="text-btn">Start calibration</Link></div><p className="muted">Capture three examples for each letter you want to recognize, fit a signer adapter, then test unseen examples live.</p></section>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">RECENT</div><h2>Recognition events</h2></div><Link to="/history" className="text-btn">View history</Link></div>{data.recent_activity?.length ? <div className="activity-list">{data.recent_activity.map((item: any) => <div className="activity-row" key={item.id}><div><strong>{item.predicted_text}</strong><span>{new Date(item.created_at).toLocaleString()}</span></div><span className="activity-meta">{Math.round((item.confidence || 0) * 100)}%</span></div>)}</div> : <Empty text="No letter predictions yet." />}</section>
    </>}
  </Page>;
}

function Recognize() {
  const session = useLandmarkSession(10); const [prediction, setPrediction] = useState("—"); const [confidence, setConfidence] = useState(0); const [latency, setLatency] = useState<number | null>(null); const [error, setError] = useState(""); const [sending, setSending] = useState(false); const [userId, setUserId] = useState<number>(); const [adapters, setAdapters] = useState<any[]>([]); const [adapterId, setAdapterId] = useState<number | undefined>();
  useEffect(() => { api.me().then((u) => setUserId(u.id)); api.letterAdapters().then((items) => { setAdapters(items); if (items[0]) setAdapterId(items[0].id); }).catch(() => setAdapters([])); }, []);
  useEffect(() => {
    const timer = window.setInterval(async () => {
      if (sending || !userId || !adapterId) return;
      const frame = session.snapshot().at(-1);
      if (!frame || (!frame.leftVisible && !frame.rightVisible)) return;
      setSending(true);
      try {
        const result = await api.letterPredict(userId, adapterId, [...frame.leftHand, ...frame.rightHand]);
        setPrediction(result.predicted_letter);
        setConfidence(result.confidence || 0);
        setLatency(result.latency_ms);
        setError("");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Recognition failed");
      } finally {
        setSending(false);
      }
    }, 700);
    return () => window.clearInterval(timer);
  }, [adapterId, sending, session, userId]);
  return <Page title="Recognize" subtitle="Live fingerspelling recognition using your signer-specific few-shot adapter."><div className="translate-grid"><section className="panel camera-panel"><div className="camera-shell"><video ref={session.videoRef} muted playsInline /><canvas ref={session.canvasRef} className="skeleton-overlay" /><div className="camera-meta"><span>{session.status}</span><span>{session.fps} fps</span><span>{latency ? Math.round(latency) + " ms" : "—"}</span></div></div><div className="button-row"><button className="primary-btn" onClick={() => session.start().catch(() => undefined)} disabled={session.running}>{session.running ? "Running" : "Start camera"}</button><button className="ghost-btn" onClick={session.stop} disabled={!session.running}>Stop</button></div><div className="selector-row"><label>Signer adapter<select value={adapterId ?? ""} onChange={e => setAdapterId(e.target.value ? Number(e.target.value) : undefined)}><option value="">Choose an adapter</option>{adapters.map((a) => <option key={a.id} value={a.id}>Adapter #{a.id}{a.letters ? " · " + a.letters.join("") : ""}</option>)}</select></label></div>{error && <div className="alert error">{error}</div>}<div className="hand-legend"><span><i className="legend-mark" /> Left hand</span><span><i className="legend-mark second" /> Right hand</span><span className="muted">126 normalized XYZ features</span></div></section><section className="panel output-panel"><div className="panel-head"><div><div className="eyebrow">LETTER</div><h2>Prediction</h2></div><span className="confidence">{Math.round(confidence * 100)}%</span></div><div className="translation-text">{prediction}</div><div className="progress"><span style={{ width: Math.round(confidence * 100) + "%" }} /></div><div className="alert">{adapterId ? (sending ? "Recognizing…" : "Show one calibrated letter at a time.") : "Calibrate at least two letters first."}</div></section></div></Page>;
}

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

function Calibration() {
  const session = useLandmarkSession(10); const [selectedLetter, setSelectedLetter] = useState("A"); const [samples, setSamples] = useState<Record<string, number[][]>>({}); const [userId, setUserId] = useState<number>(); const [busy, setBusy] = useState(false); const [message, setMessage] = useState("Capture three examples for each letter you want to recognize."); const [startedAt, setStartedAt] = useState<number | null>(null);
  useEffect(() => { api.me().then((u) => setUserId(u.id)); }, []);
  const selectedSamples = samples[selectedLetter] || [];
  const calibratedLetters = useMemo(() => LETTERS.filter((letter) => (samples[letter] || []).length > 0), [samples]);
  const capture = () => {
    const frame = session.snapshot().at(-1);
    if (!frame || (!frame.leftVisible && !frame.rightVisible)) { setMessage("No hand detected. Keep the hand inside the camera frame and try again."); return; }
    const vector = [...frame.leftHand, ...frame.rightHand];
    setSamples((current) => ({ ...current, [selectedLetter]: [...(current[selectedLetter] || []), vector].slice(-5) }));
    setMessage(selectedLetter + ": " + (selectedSamples.length + 1) + " example captured.");
  };
  const fit = async () => {
    if (!userId) return;
    const chosen = LETTERS.filter((letter) => (samples[letter] || []).length > 0);
    if (chosen.length < 2) { setMessage("Choose at least two letters before fitting the adapter."); return; }
    const weak = chosen.find((letter) => (samples[letter] || []).length < 3);
    if (weak) { setMessage(weak + " needs 3 examples before the adapter can be fitted."); return; }
    setBusy(true); setMessage("Fitting signer adapter…");
    try {
      const payload: LetterSample[] = chosen.flatMap((letter) => samples[letter].map((hand_keypoints) => ({ letter, hand_keypoints })));
      const started = startedAt || Date.now();
      const result = await api.letterCalibrate(userId, payload, Math.max(1, Math.round((Date.now() - started) / 1000)));
      setMessage("Adapter #" + result.adapter_id + " fitted for " + result.letters.join(", ") + ".");
      setSamples({});
      setStartedAt(null);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Calibration failed");
    } finally { setBusy(false); }
  };
  return <Page title="Calibration" subtitle="Teach VisionBridge your hand shapes with a small number of signer-specific examples."><section className="panel"><div className="camera-shell small"><video ref={session.videoRef} muted playsInline /><canvas ref={session.canvasRef} className="skeleton-overlay" /></div><div className="button-row" style={{ marginTop: "1rem" }}><button className="primary-btn" onClick={() => { session.start().then(() => setStartedAt((current) => current || Date.now())).catch(() => undefined); }} disabled={session.running || busy}>Start camera</button><button className="ghost-btn" onClick={session.stop} disabled={!session.running || busy}>Stop</button><button className="ghost-btn" onClick={capture} disabled={!session.running || busy}>Capture {selectedLetter}</button><button className="primary-btn" onClick={fit} disabled={busy || calibratedLetters.length < 2}>Fit adapter</button></div><div style={{ display: "grid", gridTemplateColumns: "repeat(13, minmax(32px, 1fr))", gap: 6, marginTop: "1rem" }}>{LETTERS.map((letter) => { const count = samples[letter]?.length || 0; return <button key={letter} type="button" onClick={() => setSelectedLetter(letter)} className={selectedLetter === letter ? "primary-btn" : "ghost-btn"} style={{ padding: "8px 4px" }}>{letter}<span style={{ display: "block", fontSize: 10, opacity: .7 }}>{count}/3</span></button>; })}</div><div className="alert" style={{ marginTop: "1rem" }}>{message}</div></section></Page>;
}

function History() {
  const [data, setData] = useState<any>(); const [query, setQuery] = useState(""); const [exporting, setExporting] = useState(false);
  useEffect(() => { api.history().then(setData).catch(() => setData({ items: [] })); }, []);
  async function downloadCsv() { setExporting(true); try { const blob = await api.exportHistoryCsv(); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = "visionbridge-letter-history.csv"; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url); } finally { setExporting(false); } }
  const rows = data?.items?.filter((x: any) => !query || String(x.predicted_text).toLowerCase().includes(query.toLowerCase())) || [];
  return <Page title="History" subtitle="Recorded letter predictions from the current signer session."><section className="panel"><div className="panel-head"><div><div className="eyebrow">EVENTS</div><h2>History</h2></div><div style={{ display: "flex", gap: ".5rem" }}><input className="compact-input" placeholder="Search letters" value={query} onChange={e => setQuery(e.target.value)} /><button type="button" className="ghost-btn" onClick={downloadCsv} disabled={exporting}>{exporting ? "Exporting…" : "Export CSV"}</button></div></div>{!data ? <Loading /> : rows.length ? <div className="table-wrap"><table><thead><tr><th>Time</th><th>Letter</th><th>Confidence</th><th>Latency</th></tr></thead><tbody>{rows.map((row: any) => <tr key={row.id}><td>{new Date(row.created_at).toLocaleString()}</td><td><strong>{row.predicted_text}</strong></td><td>{Math.round((row.confidence || 0) * 100)}%</td><td>{Math.round(row.latency_ms || 0)} ms</td></tr>)}</tbody></table></div> : <Empty text="No letter predictions yet." />}</section></Page>;
}

function Settings() {
  const [user, setUser] = useState<any>(); const [adapters, setAdapters] = useState<any[]>([]);
  const refresh = () => Promise.all([api.me(), api.letterAdapters()]).then(([u, a]) => { setUser(u); setAdapters(a); });
  useEffect(() => { refresh(); }, []);
  return <Page title="Settings" subtitle="Signer identity and adapter lifecycle."><section className="panel narrow"><div className="eyebrow">ACCOUNT</div><h2>{user?.username || "Loading…"}</h2><p className="muted">Account ID {user?.id ?? "—"}</p></section><section className="panel narrow"><div className="eyebrow">ADAPTERS</div><h2>Few-shot signer adapters</h2>{adapters.length ? <div className="adapter-list">{adapters.map((a) => <div className="adapter-row" key={a.id}><div><strong>Adapter #{a.id}</strong><span>{a.letters ? a.letters.join(" · ") : ((a.calibration_seconds ?? 0) + "s")}</span></div><button className="ghost-btn" onClick={() => api.deleteAdapter(a.id).then(refresh)}>Delete</button></div>)}</div> : <Empty text="No adapters yet. Calibrate a few letters first." />}</section></Page>;
}

export default function App() {
  const [authed, setAuthed] = useState(Boolean(getToken())); const [username, setUsername] = useState<string>(); const navigate = useNavigate();
  useEffect(() => { if (authed) api.me().then((user) => setUsername(user.username)).catch(() => { clearToken(); setAuthed(false); }); }, [authed]);
  const logout = () => { clearToken(); setAuthed(false); navigate("/login"); };
  if (!authed) return <Routes><Route path="*" element={<Auth onAuthed={() => setAuthed(true)} />} /></Routes>;
  return <Shell username={username} onLogout={logout}><Routes>
    <Route path="/" element={<Navigate to="/dashboard" replace />} />
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/translate" element={<Recognize />} />
    <Route path="/calibration" element={<Calibration />} />
    <Route path="/history" element={<History />} />
    <Route path="/settings" element={<Settings />} />
    <Route path="/evaluation" element={<Navigate to="/dashboard" replace />} />
    <Route path="*" element={<Navigate to="/dashboard" replace />} />
  </Routes></Shell>;
}
