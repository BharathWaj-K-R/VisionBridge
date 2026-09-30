import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, clearLocalAuth, clearSessionHint, hasSessionHint, setSessionHint, isLocalAuthenticated, type LetterSample } from "./api";
import { BrowserLetterAdapter, type BrowserLetterModel } from "./browserModel";
import { useLandmarkSession } from "./useLandmarkSession";


const navItems = [
  ["/translate", "Live Translate"],
  ["/calibration", "Few-Shot Calibration"],
  ["/history", "Letter History"],
  ["/settings", "Signer Profiles"],
] as const;

function Shell({ children, username, onLogout }: { children: ReactNode; username?: string; onLogout: () => void }) {
  const location = useLocation();
  return <div className="app-shell">
    <header className="topbar">
      <Link to="/dashboard" className="brand-lockup" aria-label="VisionBridge dashboard"><span className="brand-mark">V</span><span className="brand-copy"><strong>VisionBridge</strong><small>ISL RECOGNIZER</small></span></Link>
      <div className="header-badges"><span className="status-chip"><i /> Base Model · 26 A–Z</span><span className="status-chip"><i /> Few-Shot Adapter · Optional</span></div>
      <nav className="topnav" aria-label="Primary navigation">
        {navItems.map(([path, label]) => <Link key={path} to={path} className={location.pathname.startsWith(path) ? "topnav-link active" : "topnav-link"}>{label}</Link>)}
      </nav>
      <div className="top-actions"><Link className="icon-btn" to="/dashboard" aria-label="Dashboard" title="Dashboard">⌂</Link><span className="user-avatar" title={username || "Signed in"}>{(username || "U").slice(0, 1).toUpperCase()}</span><button className="icon-btn" onClick={onLogout} aria-label="Log out" title="Log out">↗</button></div>
    </header>
    <div className="runtime-strip">
      <div><span className="engine-status"><i /> ENGINE {import.meta.env.VITE_LOCAL_MODE !== "false" ? "DEMO" : "ONLINE"}</span><span>DEVICE: <b>Browser Camera</b></span><span>•</span><span>PIPELINE: <b>MediaPipe Tasks · 126D</b></span><span>•</span><span>TRACKING: <b>2 HANDS</b></span></div>
      <div><span>USER: <b>{username || "Signed in"}</b></span><span className="mode-tag">{import.meta.env.VITE_LOCAL_MODE !== "false" ? "LOCAL RUNTIME" : "BROWSER INFERENCE"}</span></div>
    </div>
    <main className="main-pane">{children}</main>
  </div>;
}

function Auth({ onAuthed }: { onAuthed: () => void }) {
  const location = useLocation();
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
  const [notice, setNotice] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const oauthError = params.get("oauth_error");
    if (oauthError) setError("Google sign-in failed: " + oauthError.replaceAll("_", " "));
    if (params.get("oauth_success") === "1") {
      setSessionHint();
      setNotice("Google sign-in successful.");
      onAuthed();
    }
  }, [location.search, onAuthed]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true); setError(""); setNotice("");
    try {
      if (mode === "register") {
        if (password !== confirmPassword) throw new Error("Passwords do not match.");
        await api.register(username, email, password);
        await api.login(username, password);
      } else {
        await api.login(identifierMode === "email" ? email : username, password);
      }
      setSessionHint();
      onAuthed();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  };

  return <div className="auth-page"><section className="auth-card">
    <div className="eyebrow">INDIAN SIGN LANGUAGE · LETTERS</div><h1>VisionBridge</h1>
    <p className="muted">Signer-adaptive fingerspelling recognition from a few hand examples.</p>
    <div className="auth-tabs">
      <button type="button" className={mode === "login" ? "auth-tab active" : "auth-tab"} onClick={() => setMode("login")}>Sign in</button>
      <button type="button" className={mode === "register" ? "auth-tab active" : "auth-tab"} onClick={() => setMode("register")}>Create account</button>
    </div>
    <form onSubmit={submit} className="stack">
      {mode === "register" ? <>
        <label>Username<input value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" required /></label>
        <label>Email<input value={email} onChange={e => setEmail(e.target.value)} type="email" autoComplete="email" required /></label>
      </> : <>
        <div className="identifier-toggle" role="group" aria-label="Login identifier">
          <button type="button" className={identifierMode === "username" ? "active" : ""} onClick={() => setIdentifierMode("username")}>Username</button>
          <button type="button" className={identifierMode === "email" ? "active" : ""} onClick={() => setIdentifierMode("email")}>Email</button>
        </div>
        <label>{identifierMode === "email" ? "Email" : "Username"}<input value={identifierMode === "email" ? email : username} onChange={e => identifierMode === "email" ? setEmail(e.target.value) : setUsername(e.target.value)} type={identifierMode === "email" ? "email" : "text"} autoComplete={identifierMode === "email" ? "email" : "username"} required /></label>
      </>}
      <label>Password<span className="password-field"><input value={password} onChange={e => setPassword(e.target.value)} type={showPassword ? "text" : "password"} minLength={8} autoComplete={mode === "login" ? "current-password" : "new-password"} required /><button type="button" onClick={() => setShowPassword(v => !v)}>{showPassword ? "Hide" : "Show"}</button></span></label>
      {mode === "register" && <label>Confirm password<span className="password-field"><input value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} type={showConfirmPassword ? "text" : "password"} minLength={8} autoComplete="new-password" required /><button type="button" onClick={() => setShowConfirmPassword(v => !v)}>{showConfirmPassword ? "Hide" : "Show"}</button></span></label>}
      {notice && <div className="alert">{notice}</div>}{error && <div className="alert error">{error}</div>}
      <button className="primary-btn auth-submit" disabled={busy}>{busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}</button>
    </form>
    <div className="auth-divider"><span>OR</span></div>
    <button type="button" className="google-btn" onClick={() => api.googleLogin()}><span className="google-mark">G</span> Continue with Google</button>
    <p className="auth-footnote">Google sign-in requires production OAuth credentials on the backend.</p>
  </section></div>;
}
function Page({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return <div className="page"><header className="page-header"><div><div className="eyebrow">VISIONBRIDGE / WORKSTATION</div><h1>{title}</h1><p className="muted">{subtitle}</p></div></header>{children}</div>;
}
function Loading() { return <div className="loading">Loading workspace…</div>; }
function Empty({ text }: { text: string }) { return <div className="empty">{text}</div>; }
function Metric({ label, value, detail }: { label: string; value: string | number; detail: string }) { return <div className="metric"><span className="eyebrow">{label}</span><strong>{value}</strong><span className="muted">{detail}</span></div>; }

function Dashboard() {
  const [data, setData] = useState<any>(null); const [error, setError] = useState("");
  useEffect(() => { api.dashboard().then(setData).catch((e) => setError(e.message)); }, []);
  return <Page title="Letter recognition" subtitle="Hand-only, signer-adaptive recognition for one ISL letter at a time.">
    {error ? <div className="alert error">{error}</div> : !data ? <Loading /> : <>
      <div className="metric-grid">
        <Metric label="Mode" value="Letters" detail="hand-only" />
        <Metric label="Events" value={data.usage?.translation_events ?? 0} detail="recognition events" />
        <Metric label="Confidence" value={data.usage?.average_confidence != null ? Math.round(data.usage.average_confidence * 100) + "%" : "—"} detail="recent average" />
        <Metric label="Latency" value={data.usage?.average_latency_ms != null ? Math.round(data.usage.average_latency_ms) + " ms" : "—"} detail="recent average" />
      </div>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">QUICK START</div><h2>Start with the base model</h2></div><Link to="/translate" className="text-btn">Open live translate</Link></div><p className="muted">The V3 base model is ready immediately after login. Optional calibration adds signer-specific prototype matching when you need personalization.</p></section>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">RECENT</div><h2>Recognition events</h2></div><Link to="/history" className="text-btn">View history</Link></div>{data.recent_activity?.length ? <div className="activity-list">{data.recent_activity.map((item: any) => <div className="activity-row" key={item.id}><div><strong>{item.predicted_text}</strong><span>{new Date(item.created_at).toLocaleString()}</span></div><span className="activity-meta">{Math.round((item.confidence || 0) * 100)}%</span></div>)}</div> : <Empty text="No letter predictions yet." />}</section>
    </>}
  </Page>;
}


function Recognize() {
  const session = useLandmarkSession(15);
  const [prediction, setPrediction] = useState("—");
  const [confidence, setConfidence] = useState(0);
  const [latency, setLatency] = useState<number | null>(null);
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [fastReady, setFastReady] = useState(false);
  const [userId, setUserId] = useState<number>();
  const [adapters, setAdapters] = useState<any[]>([]);
  const [adapterId, setAdapterId] = useState<number | undefined>();
  const [buffer, setBuffer] = useState<string[]>([]);
  const [temporal, setTemporal] = useState<Array<{ letter: string; confidence: number; at: string }>>([]);
  const baseModelRef = useRef<BrowserLetterModel | null>(null);
  const adapterRef = useRef<BrowserLetterAdapter | null>(null);
  const lastEventRef = useRef({ letter: "", time: 0 });

  useEffect(() => {
    api.me().then((u) => setUserId(u.id));
    api.letterAdapters().then((items) => setAdapters(items)).catch(() => setAdapters([]));
  }, []);

  useEffect(() => {
    let cancelled = false;
    baseModelRef.current = null;
    adapterRef.current = null;
    setFastReady(false);

    if (import.meta.env.VITE_LOCAL_MODE !== "false") {
      return () => { cancelled = true; };
    }

    void (async () => {
      try {
        const model = await api.letterBrowserModel();
        if (cancelled) return;
        baseModelRef.current = model;

        if (adapterId != null) {
          const payload = await api.letterAdapterPayload(adapterId);
          if (cancelled) return;
          adapterRef.current = new BrowserLetterAdapter(model, payload);
        }

        setFastReady(true);
        setError("");
      } catch (err) {
        if (cancelled) return;
        baseModelRef.current = null;
        adapterRef.current = null;
        setFastReady(false);
        setError(err instanceof Error ? err.message : "Browser model could not be loaded");
      }
    })();

    return () => {
      cancelled = true;
      baseModelRef.current = null;
      adapterRef.current = null;
    };
  }, [adapterId]);

  useEffect(() => {
    const intervalMs = import.meta.env.VITE_LOCAL_MODE !== "false" ? 220 : 33;
    const timer = window.setInterval(() => {
      const frame = session.latestFrame();
      if (!userId || !frame || (!frame.leftVisible && !frame.rightVisible)) return;
      const raw = [...frame.leftHand, ...frame.rightHand];

      const record = (letter: string, score: number, ms: number, mode: "base" | "adapter", sim?: number) => {
        setPrediction(letter);
        setConfidence(score);
        setLatency(ms);
        setSimilarity(mode === "adapter" && sim != null ? sim : null);
        setTemporal((items) => [{ letter, confidence: score, at: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) }, ...items].slice(0, 6));
        const now = performance.now();
        const previous = lastEventRef.current;
        if (letter !== previous.letter || now - previous.time >= 1000) {
          lastEventRef.current = { letter, time: now };
          if (letter !== "?") setBuffer((items) => [...items.slice(-5), letter]);
          void api.logLetterEvent({
            user_id: userId,
            adapter_id: mode === "adapter" ? adapterId ?? null : null,
            predicted_letter: letter,
            confidence: score,
            latency_ms: ms,
          });
        }
      };

      if (import.meta.env.VITE_LOCAL_MODE !== "false") {
        if (!adapterId) return;
        api.letterPredict(userId, adapterId, raw)
          .then((result) => record(result.predicted_letter, result.confidence || 0, result.latency_ms, "adapter"))
          .catch((err) => setError(err instanceof Error ? err.message : "Recognition failed"));
        return;
      }

      if (adapterId) {
        const adapter = adapterRef.current;
        if (!adapter) return;
        try {
          const result = adapter.predict(raw);
          record(result.predicted_letter, result.confidence, result.inference_ms, "adapter", result.similarity);
        } catch (err) {
          setError(err instanceof Error ? err.message : "Recognition failed");
        }
        return;
      }

      const model = baseModelRef.current;
      if (!model) return;
      try {
        const started = performance.now();
        const result = model.predictBase(raw);
        record(result.label, result.confidence, performance.now() - started, "base");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Recognition failed");
      }
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [adapterId, session, userId]);

  const activeAdapter = adapters.find((item) => item.id === adapterId);

  return <Page title="Live Translate" subtitle="Real-time signer-adaptive A–Z recognition with live hand tracing.">
    <div className="translate-grid">
      <section className="left-stack">
        <section className="panel camera-panel">
          <div className="camera-topline"><span><i /> CAMERA FEED · {session.status}</span><span>{session.fps.toFixed(1)} FPS · {latency != null ? latency.toFixed(1) + " ms" : "—"}</span></div>
          <div className="camera-shell">
            <video ref={session.videoRef} muted playsInline />
            <canvas ref={session.canvasRef} className="skeleton-overlay" />
            <div className="camera-corner top-left">126D VECTOR · {session.running ? (adapterId ? "ADAPTER ACTIVE" : "BASE MODEL") : "STANDBY"}</div>
            <div className="camera-corner top-right">MEDIA PIPE · 0.10.35</div>
            <div className="camera-corner bottom-left">ISL TWO-HANDED GESTURE</div>
            <div className="camera-corner bottom-right">{session.running ? "LIVE" : "IDLE"}</div>
          </div>
          <div className="button-row">
            <button className="primary-btn" onClick={() => session.start().catch(() => undefined)} disabled={session.running}>{session.running ? "TRACKING" : "START CAMERA"}</button>
            <button className="ghost-btn" onClick={session.stop} disabled={!session.running}>STOP</button>
            <label className="adapter-inline">RECOGNITION MODE<select value={adapterId ?? ""} onChange={(e) => setAdapterId(e.target.value ? Number(e.target.value) : undefined)}><option value="">Base Model · no adapter</option>{adapters.map((a) => <option key={a.id} value={a.id}>Few-Shot Adapter #{a.id}{a.letters ? " · " + a.letters.join("") : ""}</option>)}</select></label>
          </div>
          {error && <div className="alert error">{error}</div>}
          <div className="hand-telemetry">
            <div><span>LEFT HAND</span><strong>{session.latestFrame()?.leftVisible ? "21 / 21" : "0 / 21"}</strong><small>{session.latestFrame()?.leftVisible ? "active" : "not detected"}</small></div>
            <div><span>RIGHT HAND</span><strong>{session.latestFrame()?.rightVisible ? "21 / 21" : "0 / 21"}</strong><small>{session.latestFrame()?.rightVisible ? "active" : "not detected"}</small></div>
            <div><span>MODEL</span><strong>{import.meta.env.VITE_LOCAL_MODE !== "false" ? "DEMO" : fastReady ? (adapterId ? "ADAPTER" : "BASE") : "LOAD"}</strong><small>{adapterId ? "64D prototype match" : "26-class V3 softmax"}</small></div>
            <div><span>LATENCY</span><strong>{latency != null ? latency.toFixed(1) + " ms" : "—"}</strong><small>latest inference</small></div>
          </div>
        </section>
      </section>

      <section className="right-stack">
        <section className="panel classification-card">
          <div className="panel-head"><div><div className="eyebrow">PRIMARY CLASSIFICATION</div><h2>ISL Bilateral Alphabet</h2></div><span className="confidence-badge">{Math.round(confidence * 100)}% CONFIDENCE</span></div>
          <div className="prediction-line"><strong>{prediction}</strong><div><span>ONE LETTER · {adapterId ? "FEW-SHOT" : "BASE MODEL"}</span><small>{adapterId ? "Signer-adaptive prototype classification" : "V3 26-class softmax classification"}</small></div></div>
          <div className="progress"><span style={{ width: Math.round(confidence * 100) + "%" }} /></div>
          <div className="prediction-meta"><span>Similarity {adapterId && similarity != null ? similarity.toFixed(3) : "—"}</span><span>{adapterId ? "Threshold 0.350" : "26-class softmax"}</span><span>Embedding 64D</span></div>
        </section>

        <section className="panel">
          <div className="panel-head"><div><div className="eyebrow">WORD RECONSTRUCTION BUFFER</div><h2>Current sequence</h2></div><span className="mono">TOKEN_COUNT: {buffer.length}</span></div>
          <div className="token-buffer">{buffer.length ? buffer.map((letter, index) => <span className={index === buffer.length - 1 ? "token current" : "token"} key={letter + "-" + index}>{letter === " " ? "·" : letter}</span>) : <span className="buffer-empty">Predict a letter to begin the buffer.</span>}</div>
          <div className="button-row"><button className="primary-btn" onClick={() => setBuffer((items) => [...items, " "])} disabled={!buffer.length}>ADD SPACE</button><button className="ghost-btn" onClick={() => setBuffer((items) => items.slice(0, -1))} disabled={!buffer.length}>BACKSPACE</button><button className="ghost-btn" onClick={() => setBuffer([])} disabled={!buffer.length}>CLEAR</button></div>
          <div className="buffer-output">{buffer.join("") || "—"}</div>
        </section>

        <section className="panel">
          <div className="panel-head"><div><div className="eyebrow">TEMPORAL INFERENCE LOG</div><h2>Last six predictions</h2></div><span className="mono">LIVE WINDOW</span></div>
          <div className="temporal-grid">{temporal.length ? temporal.map((item, index) => <div className="temporal-cell" key={item.at + "-" + index}><strong>{item.letter}</strong><span>{Math.round(item.confidence * 100)}%</span><small>{item.at}</small></div>) : <Empty text="No live predictions yet." />}</div>
        </section>

        <section className="panel signer-card">
          <div className="panel-head"><div><div className="eyebrow">CURRENT RECOGNITION MODE</div><h2>{activeAdapter ? "Few-Shot Adapter" : "Base Model"}</h2></div><span className={activeAdapter ? "status-chip dark" : "status-chip"}>{activeAdapter ? "ACTIVE" : "DEFAULT"}</span></div>
          {activeAdapter ? <div className="signer-grid"><div><span>ADAPTER</span><strong>#{activeAdapter.id}</strong></div><div><span>LETTERS</span><strong>{activeAdapter.letters?.join(" · ") || "—"}</strong></div><div><span>SHOTS</span><strong>{activeAdapter.shots ? Object.values(activeAdapter.shots).reduce((sum: number, value: any) => sum + Number(value || 0), 0) : "—"}</strong></div></div> : <Empty text="Using the V3 base model. Calibration is optional and adds signer-specific prototype matching." />}
          <Link to="/calibration" className="text-btn">CALIBRATE A SIGNER →</Link>
        </section>
      </section>
    </div>
  </Page>;
}

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");


function Calibration() {
  const session = useLandmarkSession(15);
  const [selectedLetter, setSelectedLetter] = useState("A");
  const [samples, setSamples] = useState<Record<string, number[][]>>({});
  const [userId, setUserId] = useState<number>();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("Capture three examples for each letter you want to recognize.");
  const [startedAt, setStartedAt] = useState<number | null>(null);

  useEffect(() => { api.me().then((u) => setUserId(u.id)); }, []);

  const selectedSamples = samples[selectedLetter] || [];
  const completed = useMemo(() => LETTERS.filter((letter) => (samples[letter] || []).length >= 3), [samples]);

  const capture = () => {
    const frame = session.snapshot().at(-1);
    if (!frame || (!frame.leftVisible && !frame.rightVisible)) { setMessage("No hand detected. Keep the gesture inside the frame and try again."); return; }
    const vector = [...frame.leftHand, ...frame.rightHand];
    setSamples((current) => ({ ...current, [selectedLetter]: [...(current[selectedLetter] || []), vector].slice(-5) }));
    setMessage(selectedLetter + ": sample " + Math.min(selectedSamples.length + 1, 5) + " captured.");
  };

  const fit = async () => {
    if (!userId) return;
    const chosen = LETTERS.filter((letter) => (samples[letter] || []).length > 0);
    if (chosen.length < 2) { setMessage("Select at least two letters before fitting the adapter."); return; }
    const weak = chosen.find((letter) => (samples[letter] || []).length < 3);
    if (weak) { setMessage(weak + " needs 3 examples before the adapter can be fitted."); return; }
    setBusy(true); setMessage("Fitting signer prototype adapter…");
    try {
      const payload: LetterSample[] = chosen.flatMap((letter) => samples[letter].map((hand_keypoints) => ({ letter, hand_keypoints })));
      const started = startedAt || Date.now();
      const result = await api.letterCalibrate(userId, payload, Math.max(1, Math.round((Date.now() - started) / 1000)));
      setMessage("Adapter #" + result.adapter_id + " fitted for " + result.letters.join(", ") + ".");
      setSamples({}); setStartedAt(null);
    } catch (err) { setMessage(err instanceof Error ? err.message : "Calibration failed"); }
    finally { setBusy(false); }
  };

  const percent = Math.round((completed.length / 26) * 100);

  return <Page title="Few-Shot Calibration" subtitle="Capture signer-specific prototypes with the same 126D landmark pipeline used for recognition.">
    <div className="calibration-overline"><span><b>SIGNER</b> Primary Profile</span><span><b>ALPHABET</b> 26 Letters · ISL A–Z</span><span className="calibration-progress"><b>PROGRESS</b> {completed.length}/26 <i><span style={{ width: percent + "%" }} /></i></span><button className="ghost-btn" onClick={() => { setSamples({}); setMessage("Calibration samples cleared."); }}>RESET</button></div>
    <div className="calibration-layout">
      <section className="panel">
        <div className="camera-topline"><span><i /> CAMERA FEED · {session.status}</span><span>TARGET: {selectedLetter} · {selectedSamples.length}/3</span></div>
        <div className="camera-shell calibration-camera"><video ref={session.videoRef} muted playsInline /><canvas ref={session.canvasRef} className="skeleton-overlay" /><div className="camera-corner top-left">TARGET GESTURE · {selectedLetter}</div><div className="camera-corner top-right">30 FPS · 720P TARGET</div><div className="camera-corner bottom-left">POSE: CALIBRATING</div></div>
        <div className="target-instruction"><div><span className="eyebrow">TARGET GESTURE</span><h2>Letter ‘{selectedLetter}’</h2><p>Hold the selected gesture steady inside the frame, then capture three clean samples.</p></div><strong>{selectedLetter}</strong></div>
        <div className="button-row"><button className="primary-btn" onClick={() => session.start().then(() => setStartedAt((current) => current || Date.now())).catch(() => undefined)} disabled={session.running || busy}>{session.running ? "TRACKING" : "START CAMERA"}</button><button className="ghost-btn" onClick={session.stop} disabled={!session.running || busy}>STOP</button><button className="primary-btn" onClick={capture} disabled={!session.running || busy}>CAPTURE SAMPLE {Math.min(selectedSamples.length + 1, 3)}/3</button></div>
        <div className="alert">{message}</div>
      </section>

      <section className="calibration-right">
        <div className="panel"><div className="panel-head"><div><div className="eyebrow">A–Z SIGN CALIBRATION MATRIX</div><h2>Capture coverage</h2></div><span className="mono">{completed.length} DONE · {26 - completed.length} LEFT</span></div>
          <div className="letter-grid">{LETTERS.map((letter) => { const count = samples[letter]?.length || 0; const done = count >= 3; const active = letter === selectedLetter; return <button type="button" key={letter} onClick={() => setSelectedLetter(letter)} className={"letter-tile" + (active ? " active" : "") + (done ? " done" : "")}><strong>{letter}</strong><span>{done ? "3/3" : count + "/3"}</span><small>{done ? "READY" : count ? "CAPTURING" : "PENDING"}</small></button>; })}</div>
        </div>
        <div className="panel"><div className="panel-head"><div><div className="eyebrow">PROTOTYPE VECTOR INSPECTOR</div><h2>Letter ‘{selectedLetter}’</h2></div><span className="mono">126D VECTOR</span></div>
          <div className="vector-preview">{selectedSamples.length ? selectedSamples[0].slice(0, 18).map((value, index) => <span key={index}>{Number(value).toFixed(3)}</span>) : <span>No captured vector yet.</span>}</div>
          <div className="inspector-meta"><span>Normalization: wrist-origin + max-distance scale</span><span>Stored server-side only as derived prototype after fit</span></div>
          <div className="button-row"><button className="primary-btn" onClick={fit} disabled={busy || completed.length < 2}>{busy ? "FITTING…" : "FIT SIGNER ADAPTER"}</button><Link to="/translate" className="ghost-btn">TEST LIVE →</Link></div>
        </div>
      </section>
    </div>
  </Page>;
}

function History() {
  const [data, setData] = useState<any>(); const [query, setQuery] = useState(""); const [exporting, setExporting] = useState(false);
  useEffect(() => { api.history().then(setData).catch(() => setData({ items: [] })); }, []);
  async function downloadCsv() { setExporting(true); try { const blob = await api.exportHistoryCsv(); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = "visionbridge-letter-history.csv"; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url); } finally { setExporting(false); } }
  const rows = data?.items?.filter((x: any) => !query || String(x.predicted_text).toLowerCase().includes(query.toLowerCase())) || [];
  return <Page title="History" subtitle="Recorded letter predictions from the current signer session."><section className="panel"><div className="panel-head"><div><div className="eyebrow">EVENTS</div><h2>History</h2></div><div style={{ display: "flex", gap: ".5rem" }}><input className="compact-input" placeholder="Search letters" value={query} onChange={e => setQuery(e.target.value)} /><button type="button" className="ghost-btn" onClick={downloadCsv} disabled={exporting}>{exporting ? "Exporting…" : "Export CSV"}</button></div></div>{!data ? <Loading /> : rows.length ? <div className="table-wrap"><table><thead><tr><th>Time</th><th>Letter</th><th>Confidence</th><th>Latency</th></tr></thead><tbody>{rows.map((row: any) => <tr key={row.id}><td>{new Date(row.created_at).toLocaleString()}</td><td><strong>{row.predicted_text}</strong></td><td>{Math.round((row.confidence || 0) * 100)}%</td><td>{Math.round(row.latency_ms || 0)} ms</td></tr>)}</tbody></table></div> : <Empty text="No letter predictions yet." />}</section></Page>;
}

function Settings() {
  const [user, setUser] = useState<any>();
  const [adapters, setAdapters] = useState<any[]>([]);
  const [theme, setTheme] = useState(localStorage.getItem("visionbridge_theme") || "system");
  const [cameraFps, setCameraFps] = useState(localStorage.getItem("visionbridge_camera_fps") || "30");
  const [autoStart, setAutoStart] = useState(localStorage.getItem("visionbridge_auto_camera") === "1");
  const [saved, setSaved] = useState(false);
  const refresh = () => Promise.all([api.me(), api.letterAdapters()]).then(([u,a]) => { setUser(u); setAdapters(a); });
  useEffect(() => { refresh(); }, []);
  const savePreferences = () => {
    localStorage.setItem("visionbridge_theme", theme);
    localStorage.setItem("visionbridge_camera_fps", cameraFps);
    localStorage.setItem("visionbridge_auto_camera", autoStart ? "1" : "0");
    document.documentElement.dataset.theme = theme;
    setSaved(true); window.setTimeout(() => setSaved(false), 1800);
  };
  return <Page title="Settings" subtitle="Account, appearance, camera behavior, and signer profile controls.">
    <div className="settings-grid">
      <section className="panel"><div className="eyebrow">ACCOUNT</div><h2>{user?.username || "Loading…"}</h2><p className="muted">{user?.email || "No email on this account yet."}</p><p className="mono">Account ID {user?.id ?? "—"}</p><div className="button-row"><button className="ghost-btn" onClick={() => void api.logout().then(() => { clearLocalAuth(); clearSessionHint(); window.location.assign("/login"); })}>Sign out</button></div></section>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">APPEARANCE</div><h2>Interface</h2></div></div><div className="settings-form">
        <label>Theme<select value={theme} onChange={e => setTheme(e.target.value)}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
        <label>Recognition FPS<select value={cameraFps} onChange={e => setCameraFps(e.target.value)}><option value="15">15 FPS · battery friendly</option><option value="30">30 FPS · balanced</option><option value="60">60 FPS · high performance</option></select></label>
        <label className="setting-toggle"><input type="checkbox" checked={autoStart} onChange={e => setAutoStart(e.target.checked)} /> Auto-start camera on Live Translate</label>
        <button className="primary-btn" onClick={savePreferences}>{saved ? "Saved" : "Save settings"}</button>
      </div></section>
      <section className="panel"><div className="eyebrow">SIGNER PROFILES</div><h2>Few-shot adapters</h2>{adapters.length ? <div className="adapter-list">{adapters.map(a => <div className="adapter-row" key={a.id}><div><strong>Adapter #{a.id}</strong><span>{a.letters ? a.letters.join(" · ") : ((a.calibration_seconds ?? 0)+"s")}</span></div><button className="ghost-btn" onClick={() => api.deleteAdapter(a.id).then(refresh)}>Delete</button></div>)}</div> : <Empty text="No adapters yet. Calibrate a few letters first." />}<Link to="/calibration" className="text-btn">CALIBRATE A SIGNER →</Link></section>
      <section className="panel"><div className="eyebrow">PRIVACY & SESSION</div><h2>Browser controls</h2><p className="muted">Authentication uses an HttpOnly session cookie. Recognition history and signer adapters are stored against your account on the production backend.</p><div className="settings-checks"><span>Session cookie · HttpOnly</span><span>CSRF protection · enabled</span><span>Camera · browser permission required</span></div></section>
    </div>
  </Page>;
}
export default function App() {
  const localMode = import.meta.env.VITE_LOCAL_MODE !== "false";
  const [authed, setAuthed] = useState(localMode ? isLocalAuthenticated() : hasSessionHint());
  const [authChecking, setAuthChecking] = useState(!localMode && hasSessionHint());
  const [username, setUsername] = useState<string>();
  const navigate = useNavigate();

  useEffect(() => {
    if (localMode) {
      if (authed) api.me().then(user => setUsername(user.username)).catch(() => { clearLocalAuth(); setAuthed(false); });
      return;
    }
    if (!hasSessionHint()) { setAuthChecking(false); return; }
    api.me().then(user => { setUsername(user.username); setAuthed(true); })
      .catch(() => { clearSessionHint(); setAuthed(false); })
      .finally(() => setAuthChecking(false));
  }, [authed, localMode]);

  const logout = () => void api.logout().then(() => { clearLocalAuth(); clearSessionHint(); setUsername(undefined); setAuthed(false); navigate("/login"); });

  if (authChecking) return <Loading />;
  if (!authed) return <Routes><Route path="*" element={<Auth onAuthed={() => setAuthed(true)} />} /></Routes>;
  return <Shell username={username} onLogout={logout}><Routes>
    <Route path="/" element={<Navigate to="/dashboard" replace />} />
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/translate" element={<Recognize />} />
    <Route path="/calibration" element={<Calibration />} />
    <Route path="/history" element={<History />} />
    <Route path="/settings" element={<Settings />} />
    <Route path="*" element={<Navigate to="/dashboard" replace />} />
  </Routes></Shell>;
}