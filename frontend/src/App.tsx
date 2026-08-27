import { FormEvent, useEffect, useRef, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, clearToken, getToken, setToken } from "./api";

const navItems = [
  ["/dashboard", "Dashboard"],
  ["/translate", "Translate"],
  ["/calibration", "Calibration"],
  ["/history", "History"],
  ["/evaluation", "Evaluation"],
  ["/settings", "Settings"],
] as const;

function Shell({ children, username, onLogout }: { children: React.ReactNode; username?: string; onLogout: () => void }) {
  const location = useLocation();
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link to="/dashboard" className="brand"><span className="brand-mark">VB</span><span>VisionBridge</span></Link>
        <nav className="nav-list" aria-label="Primary">
          {navItems.map(([path, label]) => <Link key={path} className={location.pathname.startsWith(path) ? "nav-link active" : "nav-link"} to={path}>{label}</Link>)}
        </nav>
        <div className="sidebar-foot">
          <div className="user-chip"><span className="status-dot" />{username || "Signed in"}</div>
          <button className="ghost-btn" onClick={onLogout}>Log out</button>
        </div>
      </aside>
      <main className="main-pane">{children}</main>
    </div>
  );
}

function Login({ onAuthed }: { onAuthed: (token: string) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setError(""); setBusy(true);
    try {
      if (mode === "register") await api.register(username, password);
      const token = await api.login(username, password); setToken(token.access_token); onAuthed(token.access_token);
    } catch (err) { setError(err instanceof Error ? err.message : "Authentication failed"); }
    finally { setBusy(false); }
  };
  return <div className="auth-page"><div className="auth-card">
    <div className="eyebrow">INDIAN SIGN LANGUAGE</div><h1>VisionBridge</h1><p className="muted">A focused workspace for real-time sign recognition and signer adaptation.</p>
    <form onSubmit={submit} className="stack">
      <label>Username<input value={username} onChange={e => setUsername(e.target.value)} minLength={1} autoComplete="username" required /></label>
      <label>Password<input value={password} onChange={e => setPassword(e.target.value)} minLength={8} type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} required /></label>
      {error && <div className="alert error">{error}</div>}
      <button className="primary-btn" disabled={busy}>{busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}</button>
    </form>
    <button className="text-btn" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>{mode === "login" ? "Need an account? Create one" : "Already have an account? Sign in"}</button>
  </div></div>;
}

function Dashboard() {
  const [data, setData] = useState<any>(null); const [error, setError] = useState("");
  useEffect(() => { api.dashboard().then(setData).catch(e => setError(e.message)); }, []);
  if (error) return <Page title="Dashboard"><div className="alert error">{error}</div></Page>;
  return <Page title="Dashboard" subtitle="System status, usage, and recent translation activity.">
    {!data ? <Loading /> : <>
      <div className="metric-grid">
        <Metric label="Model" value={data.model?.status || "unknown"} detail={data.model?.modality || "—"} />
        <Metric label="Translations" value={data.usage?.translation_events ?? 0} detail="stored events" />
        <Metric label="Confidence" value={data.usage?.average_confidence != null ? `${Math.round(data.usage.average_confidence * 100)}%` : "—"} detail="average" />
        <Metric label="Latency" value={data.usage?.average_latency_ms != null ? `${Math.round(data.usage.average_latency_ms)} ms` : "—"} detail="average" />
      </div>
      <section className="panel"><div className="panel-head"><div><span className="eyebrow">RECENT</span><h2>Translation activity</h2></div><Link to="/history" className="text-btn">Open history</Link></div>
        {data.recent_activity?.length ? <div className="activity-list">{data.recent_activity.map((item: any) => <div className="activity-row" key={item.id}><div><strong>{item.predicted_text || "(no sign detected)"}</strong><span>{new Date(item.created_at).toLocaleString()}</span></div><span className="activity-meta">{Math.round((item.confidence || 0) * 100)}% · {Math.round(item.latency_ms || 0)} ms</span></div>)}</div> : <Empty text="No translation events yet." />}
      </section>
    </>}
  </Page>;
}

function Metric({ label, value, detail }: { label: string; value: string | number; detail: string }) { return <div className="metric"><span className="eyebrow">{label}</span><strong>{value}</strong><span className="muted">{detail}</span></div>; }

function Translate() {
  const videoRef = useRef<HTMLVideoElement>(null); const overlayRef = useRef<HTMLCanvasElement>(null); const holisticRef = useRef<any>(null); const streamRef = useRef<MediaStream | null>(null); const activeRef = useRef(false); const lastSent = useRef(0); const poseBuffer = useRef<number[][]>([]); const faceBuffer = useRef<number[][]>([]); const leftBuffer = useRef<number[][]>([]); const rightBuffer = useRef<number[][]>([]);
  const [running, setRunning] = useState(false); const [status, setStatus] = useState("Ready"); const [prediction, setPrediction] = useState("—"); const [confidence, setConfidence] = useState(0); const [fps, setFps] = useState(0); const [latency, setLatency] = useState<number | null>(null);
  const FRAME_WINDOW = 50;
  const loadScript = (src: string) => new Promise<void>((resolve, reject) => { const existing = document.querySelector(`script[src="${src}"]`); if (existing) return resolve(); const s = document.createElement("script"); s.src = src; s.crossOrigin = "anonymous"; s.onload = () => resolve(); s.onerror = () => reject(new Error(`Failed to load ${src}`)); document.head.appendChild(s); });
  const flatten = (landmarks: any[] | undefined, dim: 63 | 132 | 1404, visibility: boolean) => { const out: number[] = []; (landmarks || []).forEach((lm: any) => { out.push(Number.isFinite(lm.x) ? lm.x : 0, Number.isFinite(lm.y) ? lm.y : 0, Number.isFinite(lm.z) ? lm.z : 0); if (visibility) out.push(Number.isFinite(lm.visibility) ? lm.visibility : 0); }); while (out.length < dim) out.push(0); return out.slice(0, dim); };
  const drawSkeleton = (ctx: CanvasRenderingContext2D, landmarks: any[] | undefined, width: number, height: number, links: number[][], label: string, anchor: string) => { if (!landmarks?.length) return; ctx.beginPath(); ctx.font = "12px Inter, sans-serif"; ctx.fillText(label, anchor === "left" ? 14 : width - 74, 20); links.forEach(([a,b]) => { if (!landmarks[a] || !landmarks[b]) return; ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height); ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height); }); ctx.stroke(); landmarks.forEach((lm: any) => { ctx.beginPath(); ctx.arc(lm.x * width, lm.y * height, 2.2, 0, Math.PI * 2); ctx.fill(); }); };
  useEffect(() => () => { activeRef.current = false; streamRef.current?.getTracks().forEach(t => t.stop()); }, []);
  const sendBatch = async () => { const now = performance.now(); if (now - lastSent.current < 700 || poseBuffer.current.length < FRAME_WINDOW) return; lastSent.current = now; const payload = { pose_keypoints: poseBuffer.current.slice(-FRAME_WINDOW), face_keypoints: faceBuffer.current.slice(-FRAME_WINDOW), left_hand_keypoints: leftBuffer.current.slice(-FRAME_WINDOW), right_hand_keypoints: rightBuffer.current.slice(-FRAME_WINDOW) }; poseBuffer.current = []; faceBuffer.current = []; leftBuffer.current = []; rightBuffer.current = []; const t = performance.now(); try { const result = await api.translate(payload); setPrediction(result.predicted_text); setConfidence(result.confidence || 0); setLatency(result.latency_ms); setStatus("Live"); } catch (e) { setStatus(e instanceof Error ? e.message : "Translation failed"); } finally { setLatency(performance.now() - t); } };
  const start = async () => { try { setStatus("Loading landmark engine…"); await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"); const Holistic = (window as any).Holistic; if (!Holistic) throw new Error("MediaPipe Holistic failed to load"); const holistic = new Holistic({ locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}` }); holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true, refineFaceLandmarks: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 }); holistic.onResults((results: any) => { const width = videoRef.current?.videoWidth || 640; const height = videoRef.current?.videoHeight || 480; const canvas = overlayRef.current; if (canvas) { canvas.width = width; canvas.height = height; const ctx = canvas.getContext("2d"); if (ctx) { ctx.clearRect(0, 0, width, height); const links = [[0,1],[1,2],[2,3],[3,4],[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[0,5],[5,9],[9,13],[13,17]]; drawSkeleton(ctx, results.leftHandLandmarks, width, height, links, "L", "left"); drawSkeleton(ctx, results.rightHandLandmarks, width, height, links, "R", "right"); } } poseBuffer.current.push(flatten(results.poseLandmarks, 132, true)); faceBuffer.current.push(flatten(results.faceLandmarks, 1404, false)); leftBuffer.current.push(flatten(results.leftHandLandmarks, 63, false)); rightBuffer.current.push(flatten(results.rightHandLandmarks, 63, false)); void sendBatch(); }); holisticRef.current = holistic; const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false }); streamRef.current = stream; if (!videoRef.current) throw new Error("Video element unavailable"); videoRef.current.srcObject = stream; await videoRef.current.play(); activeRef.current = true; setRunning(true); setStatus("Live"); let frames = 0; let last = performance.now(); const loop = async () => { if (!activeRef.current || !videoRef.current) return; await holistic.send({ image: videoRef.current }); frames += 1; const now = performance.now(); if (now - last > 1000) { setFps(frames); frames = 0; last = now; } requestAnimationFrame(loop); }; requestAnimationFrame(loop); } catch (e) { setStatus(e instanceof Error ? e.message : "Camera start failed"); setRunning(false); } };
  const stop = () => { activeRef.current = false; setRunning(false); streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null; holisticRef.current?.close?.(); holisticRef.current = null; poseBuffer.current = []; faceBuffer.current = []; leftBuffer.current = []; rightBuffer.current = []; setStatus("Stopped"); };
  return <Page title="Translate" subtitle="Real-time browser landmark extraction with synchronized hand skeleton tracking.">
    <div className="translate-grid"><section className="panel camera-panel"><div className="camera-shell"><video ref={videoRef} muted playsInline /><canvas ref={overlayRef} className="skeleton-overlay" /><div className="camera-meta"><span className="pill">{status}</span><span>{fps || 0} fps</span><span>{latency ? `${Math.round(latency)} ms` : "—"}</span></div></div><div className="button-row"><button className="primary-btn" onClick={start} disabled={running}>{running ? "Running" : "Start camera"}</button><button className="ghost-btn" onClick={stop} disabled={!running}>Stop</button></div><div className="hand-legend"><span><i className="legend-mark" /> Left hand</span><span><i className="legend-mark second" /> Right hand</span><span className="muted">21 landmarks per hand</span></div></section>
      <section className="panel output-panel"><div className="panel-head"><div><span className="eyebrow">OUTPUT</span><h2>Translation</h2></div><span className="confidence">{Math.round(confidence * 100)}%</span></div><div className="translation-text">{prediction}</div><div className="progress"><span style={{ width: `${Math.round(confidence * 100)}%` }} /></div><div className="output-meta"><span>Pose 132</span><span>Face 1404</span><span>Hands 63 + 63</span></div></section></div>
  </Page>;
}

function Calibration() { const [seconds, setSeconds] = useState(0); const [target, setTarget] = useState("i am hungry"); const [message, setMessage] = useState("Calibration requires authenticated real-video capture."); const [busy, setBusy] = useState(false); const [adapter, setAdapter] = useState<any>(null); const start = () => { setSeconds(0); const began = Date.now(); const timer = window.setInterval(() => { const value = Math.floor((Date.now() - began) / 1000); setSeconds(value); if (value >= 300) window.clearInterval(timer); }, 1000); }; return <Page title="Calibration" subtitle="Capture signer-specific examples for the lightweight BridgeAdapter."><section className="panel narrow"><div className="eyebrow">CALIBRATION</div><h2>Five-minute signer profile</h2><p className="muted">The adapter is trained on top of the frozen base model. Record real examples, provide the target sentence, then submit synchronized pose, face, and hand landmarks.</p><label>Target sentence<input value={target} onChange={e => setTarget(e.target.value)} /></label><div className="calibration-timer">{String(Math.floor(seconds / 60)).padStart(2, "0")}:{String(seconds % 60).padStart(2, "0")}</div><div className="button-row"><button className="primary-btn" onClick={start}>Start capture timer</button><button className="ghost-btn" disabled={seconds < 300 || busy} onClick={() => { setBusy(true); setMessage("Capture integration is ready for the real hand-aware payload from the live recorder."); setBusy(false); }}>Submit calibration</button></div>{adapter && <div className="alert success">Adapter {adapter.id} created.</div>}<div className="alert">{message}</div></section></Page>; }

function History() { const [data, setData] = useState<any>(null); const [query, setQuery] = useState(""); useEffect(() => { api.history("range=all&sort=newest").then(setData).catch(() => setData({ items: [] })); }, []); const filtered = data?.items?.filter((x: any) => !query || String(x.predicted_text).toLowerCase().includes(query.toLowerCase())) || []; return <Page title="History" subtitle="Persisted translation events from your account."><section className="panel"><div className="panel-head"><div><span className="eyebrow">EVENTS</span><h2>History</h2></div><input className="compact-input" placeholder="Search" value={query} onChange={e => setQuery(e.target.value)} /></div>{!data ? <Loading /> : filtered.length ? <div className="table-wrap"><table><thead><tr><th>Time</th><th>Prediction</th><th>Confidence</th><th>Latency</th></tr></thead><tbody>{filtered.map((x: any) => <tr key={x.id}><td>{new Date(x.created_at).toLocaleString()}</td><td>{x.predicted_text || "(no sign detected)"}</td><td>{Math.round((x.confidence || 0) * 100)}%</td><td>{Math.round(x.latency_ms || 0)} ms</td></tr>)}</tbody></table></div> : <Empty text="No stored events match this view." />}</section></Page>; }

function Evaluation() { const [data, setData] = useState<any>(null); useEffect(() => { api.evaluation().then(setData).catch(() => setData({ evaluation_data_available: false })); }, []); return <Page title="Evaluation" subtitle="Evidence-based model evaluation, without invented benchmark numbers."><section className="panel narrow">{!data ? <Loading /> : <><div className="eyebrow">MODEL</div><h2>{data.model?.status || "unknown"}</h2><p className="muted">{data.message || "No benchmark evidence is currently persisted."}</p><div className="metric-grid compact"><Metric label="Benchmark" value={data.evaluation_data_available ? "available" : "not measured"} detail="persisted evidence" /><Metric label="Measured adapters" value={data.measured_adapter_count ?? 0} detail="with recorded gain" /></div></>}</section></Page>; }

function Settings() { const [user, setUser] = useState<any>(null); const [adapters, setAdapters] = useState<any[]>([]); useEffect(() => { Promise.all([api.me(), api.adapters()]).then(([u, a]) => { setUser(u); setAdapters(a); }); }, []); return <Page title="Settings" subtitle="Account and signer-adapter management."><section className="panel narrow"><div className="eyebrow">ACCOUNT</div><h2>{user?.username || "Loading…"}</h2><p className="muted">Account ID {user?.id ?? "—"}</p></section><section className="panel narrow"><div className="eyebrow">ADAPTERS</div><h2>Signer profiles</h2>{adapters.length ? <div className="adapter-list">{adapters.map(a => <div className="adapter-row" key={a.id}><div><strong>Adapter #{a.id}</strong><span>{a.calibration_seconds}s calibration</span></div><button className="ghost-btn" onClick={() => api.deleteAdapter(a.id).then(() => setAdapters(current => current.filter(x => x.id !== a.id)))}>Delete</button></div>)}</div> : <Empty text="No signer adapters yet." />}</section></Page>; }

function Page({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) { return <div className="page"><header className="page-header"><div><span className="eyebrow">VISIONBRIDGE</span><h1>{title}</h1><p className="muted">{subtitle}</p></div></header>{children}</div>; }
function Loading() { return <div className="loading">Loading…</div>; }
function Empty({ text }: { text: string }) { return <div className="empty">{text}</div>; }

export default function App() {
  const [authed, setAuthed] = useState(Boolean(getToken())); const [username, setUsername] = useState<string>(); const navigate = useNavigate();
  useEffect(() => { if (authed) api.me().then(user => setUsername(user.username)).catch(() => { clearToken(); setAuthed(false); }); }, [authed]);
  const logout = () => { clearToken(); setAuthed(false); navigate("/login"); };
  if (!authed) return <Routes><Route path="*" element={<Login onAuthed={() => setAuthed(true)} />} /></Routes>;
  return <Shell username={username} onLogout={logout}><Routes><Route path="/" element={<Navigate to="/dashboard" replace />} /><Route path="/dashboard" element={<Dashboard />} /><Route path="/translate" element={<Translate />} /><Route path="/calibration" element={<Calibration />} /><Route path="/history" element={<History />} /><Route path="/evaluation" element={<Evaluation />} /><Route path="/settings" element={<Settings />} /><Route path="*" element={<Navigate to="/dashboard" replace />} /></Routes></Shell>;
}
