import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { BrowserLetterAdapter, type BrowserLetterModel } from "../browserModel";
import { normalizeHandPair, type TrackerMode } from "../landmarks";
import { useLandmarkSession } from "../useLandmarkSession";
import { Empty, Page } from "../components/Page";

export default function Recognize() {
  const configuredFps = Number(localStorage.getItem("visionbridge_camera_fps") || "30");
  const [trackerMode, setTrackerMode] = useState<TrackerMode>(() => {
    const saved = localStorage.getItem("visionbridge_tracker_mode");
    return saved === "fixed" || saved === "off" ? saved : "adaptive";
  });
  const [trackerFixedColor, setTrackerFixedColor] = useState(() => localStorage.getItem("visionbridge_tracker_fixed_color") || "#00E5FF");
  const [prediction, setPrediction] = useState("—");
  const [confidence, setConfidence] = useState(0);
  const session = useLandmarkSession(
    configuredFps,
    { mode: trackerMode, fixedColor: trackerFixedColor },
    { prediction, confidence },
  );

  useEffect(() => {
    localStorage.setItem("visionbridge_tracker_mode", trackerMode);
  }, [trackerMode]);

  useEffect(() => {
    localStorage.setItem("visionbridge_tracker_fixed_color", trackerFixedColor);
  }, [trackerFixedColor]);
  const latestFrame = session.latestFrame;
  useEffect(() => {
    if (localStorage.getItem("visionbridge_auto_camera") === "1" && !session.running) {
      void session.start().catch(() => undefined);
    }
  }, []);
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
  const candidateRef = useRef({ letter: "", since: 0, confidence: 0 });
  const readyToCommitRef = useRef(true);
  const lastCommittedLetterRef = useRef("");
  const lastLoggedAtRef = useRef(0);
  const STABLE_COMMIT_MS = 700;
  const COMMIT_CONFIDENCE = 0.70;

  useEffect(() => {
    void api.me().then((u) => setUserId(u.id)).catch(() => setError("Authentication session could not be verified. Please sign in again."));
    void api.letterAdapters().then((items) => setAdapters(items)).catch((error) => setError(error instanceof Error ? error.message : "Signer adapters could not be loaded."));
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
      const frame = latestFrame();
      if (!frame) return;
      const visible = frame.leftVisible || frame.rightVisible;
      if (!visible) {
        candidateRef.current = { letter: "", since: 0, confidence: 0 };
        readyToCommitRef.current = true;
        return;
      }
      if (!userId) return;
      const raw = [...frame.leftHand, ...frame.rightHand];

      const record = (letter: string, score: number, ms: number, mode: "base" | "adapter", sim?: number) => {
        setPrediction(letter);
        setConfidence(score);
        setLatency(ms);
        setSimilarity(mode === "adapter" && sim != null ? sim : null);
        setTemporal((items) => [{ letter, confidence: score, at: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) }, ...items].slice(0, 6));

        const now = performance.now();
        if (letter === "?" || score < COMMIT_CONFIDENCE) {
          candidateRef.current = { letter: "", since: 0, confidence: 0 };
          return;
        }

        const candidate = candidateRef.current;
        if (candidate.letter !== letter) {
          candidateRef.current = { letter, since: now, confidence: score };
          return;
        }

        candidateRef.current.confidence = Math.max(candidate.confidence, score);
        if (
          now - candidate.since < STABLE_COMMIT_MS ||
          (!readyToCommitRef.current && letter === lastCommittedLetterRef.current)
        ) return;

        readyToCommitRef.current = false;
        lastCommittedLetterRef.current = letter;
        const committedConfidence = candidateRef.current.confidence;
        setBuffer((items) => [...items.slice(-5), letter]);

        if (now - lastLoggedAtRef.current >= 1200) {
          lastLoggedAtRef.current = now;
          void api.logLetterEvent({
            user_id: userId,
            adapter_id: mode === "adapter" ? adapterId ?? null : null,
            predicted_letter: letter,
            confidence: committedConfidence,
            latency_ms: ms,
          }).catch((err) => setError(err instanceof Error ? err.message : "Prediction history could not be saved."));
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
        const result = model.predictBase(normalizeHandPair(raw));
        record(result.label, result.confidence, performance.now() - started, "base");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Recognition failed");
      }
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [adapterId, latestFrame, userId]);

  const activeAdapter = adapters.find((item) => item.id === adapterId);

  return <Page title="Translate with camera" subtitle="Show a hand sign inside the frame and VisionBridge will recognize the letter.">
    <div className="translate-grid">
      <section className="left-stack">
        <section className="panel camera-panel">
          <div className="camera-topline"><span><i /> {session.running ? "Camera is on" : "Camera is off"}</span><span>{session.running ? "Keep your hands inside the frame" : "Ready when you are"}</span></div>
          <div className="camera-shell">
            <video ref={session.videoRef} muted playsInline aria-label="Live camera preview" />
            <canvas ref={session.canvasRef} className="skeleton-overlay" aria-hidden="true" />
            <div className="camera-corner top-left">Place your hands here</div>
            <div className="camera-corner top-right">{session.running ? "Live" : "Ready"}</div>
            <div className="camera-corner bottom-left">Good light helps</div>
            {!session.running && (
              <div className="camera-empty-state">
                <div className="camera-empty-state-card">
                  <div className="eyebrow">CAMERA</div>
                  <h3>Ready to start</h3>
                  <p>{session.status !== "Ready" && session.status !== "Stopped" ? session.status : "Allow camera access, then keep your hand(s) fully visible inside the frame."}</p>
                  <button className="primary-btn" type="button" onClick={() => session.start().catch(() => undefined)}>Start camera</button>
                </div>
              </div>
            )}
          </div>
          <details className="advanced-details"><summary>Adjust hand outline visibility</summary><div className="tracker-controls">
            <div className="tracker-controls-copy">
              <span className="eyebrow">HAND TRACKER</span>
              <strong>
                {trackerMode === "adaptive" ? "ADAPTIVE VISIBILITY" : trackerMode === "fixed" ? "FIXED COLOUR" : "TRACKER VISUAL OFF"}
              </strong>
              <small>
                {trackerMode === "adaptive"
                  ? "Automatically selects a high-contrast colour against the live camera background."
                  : trackerMode === "fixed"
                    ? "Uses your selected colour consistently."
                    : "Landmark graphics are hidden while recognition continues in the background."}
              </small>
            </div>
            <div className="tracker-options">
              <button type="button" className={trackerMode === "adaptive" ? "active" : ""} onClick={() => setTrackerMode("adaptive")} aria-pressed={trackerMode === "adaptive"}>ADAPTIVE</button>
              <button type="button" className={trackerMode === "fixed" ? "active" : ""} onClick={() => setTrackerMode("fixed")} aria-pressed={trackerMode === "fixed"}>FIXED</button>
              <button type="button" className={trackerMode === "off" ? "active" : ""} onClick={() => setTrackerMode("off")} aria-pressed={trackerMode === "off"}>OFF</button>
              {trackerMode === "fixed" && (
                <label className="tracker-color-control">
                  <span>COLOUR</span>
                  <input type="color" value={trackerFixedColor} onChange={(event) => setTrackerFixedColor(event.target.value)} aria-label="Fixed tracker colour" />
                  <code>{trackerFixedColor.toUpperCase()}</code>
                </label>
              )}
            </div>
          </div></details>
          <div className="button-row">
            <button className="primary-btn" onClick={() => session.start().catch(() => undefined)} disabled={session.running}>{session.running ? "TRACKING" : "START CAMERA"}</button>
            <button className="ghost-btn" onClick={session.stop} disabled={!session.running}>STOP</button>
            <label className="adapter-inline">Recognition personalization (optional)
              <select value={adapterId ?? ""} onChange={(e) => setAdapterId(e.target.value ? Number(e.target.value) : undefined)}>
                <option value="">Standard recognition</option>
                {adapters.map((a) => <option key={a.id} value={a.id}>Personalized profile #{a.id}{a.letters ? " · " + a.letters.join("") : ""}</option>)}
              </select>
            </label>
          </div>
          {error && <div className="alert error">{error}</div>}
          <details className="advanced-details">
            <summary>Show technical details</summary>
            <div className="hand-telemetry">
              <div><span>LEFT HAND</span><strong>{session.latestFrame()?.leftVisible ? "21 / 21" : "0 / 21"}</strong><small>{session.latestFrame()?.leftVisible ? "active" : "not detected"}</small></div>
              <div><span>RIGHT HAND</span><strong>{session.latestFrame()?.rightVisible ? "21 / 21" : "0 / 21"}</strong><small>{session.latestFrame()?.rightVisible ? "active" : "not detected"}</small></div>
              <div><span>MODEL</span><strong>{import.meta.env.VITE_LOCAL_MODE !== "false" ? "DEMO" : fastReady ? (adapterId ? "ADAPTER" : "BASE") : "LOAD"}</strong><small>{adapterId ? "Personalized matching" : "Standard A–Z recognition"}</small></div>
              <div><span>RESPONSE</span><strong>{latency != null ? latency.toFixed(1) + " ms" : "—"}</strong><small>latest recognition</small></div>
            </div>
          </details>
        </section>
      </section>

      <section className="right-stack">
        <section className="panel classification-card">
          <div className="panel-head"><div><div className="eyebrow">RESULT</div><h2>Detected letter</h2></div><span className="confidence-badge">{Math.round(confidence * 100)}% sure</span></div>
          <div className="prediction-line"><strong>{prediction}</strong><div><span>{adapterId ? "Personalized recognition" : "Standard recognition"}</span><small>{confidence >= COMMIT_CONFIDENCE ? "The result is stable enough to save." : "Hold the gesture steady for a clearer result."}</small></div></div>
          <div className="progress"><span style={{ width: Math.round(confidence * 100) + "%" }} /></div>
          <div className="prediction-meta"><span>{session.running ? "Camera active" : "Camera paused"}</span><span>{adapterId ? "Personalized profile" : "Standard recognition"}</span></div>
        </section>

        <section className="panel">
          <div className="panel-head"><div><div className="eyebrow">YOUR LETTERS</div><h2>Current message</h2></div><span className="status-chip">{buffer.length} letter{buffer.length === 1 ? "" : "s"}</span></div>
          <div className="token-buffer">{buffer.length ? buffer.map((letter, index) => <span className={index === buffer.length - 1 ? "token current" : "token"} key={letter + "-" + index}>{letter === " " ? "·" : letter}</span>) : <span className="buffer-empty">Predict a letter to begin the buffer.</span>}</div>
          <div className="button-row"><button className="primary-btn" onClick={() => setBuffer((items) => [...items, " "])} disabled={!buffer.length}>ADD SPACE</button><button className="ghost-btn" onClick={() => setBuffer((items) => items.slice(0, -1))} disabled={!buffer.length}>BACKSPACE</button><button className="ghost-btn" onClick={() => setBuffer([])} disabled={!buffer.length}>CLEAR</button></div>
          <div className="buffer-output">{buffer.join("") || "—"}</div>
        </section>

        <section className="panel">
          <div className="panel-head"><div><div className="eyebrow">RECENT RESULTS</div><h2>Last few letters</h2></div></div>
          <div className="temporal-grid">{temporal.length ? temporal.map((item, index) => <div className="temporal-cell" key={item.at + "-" + index}><strong>{item.letter}</strong><span>{Math.round(item.confidence * 100)}%</span><small>{item.at}</small></div>) : <Empty text="No live predictions yet." />}</div>
        </section>

        <section className="panel signer-card">
          <div className="panel-head"><div><div className="eyebrow">PERSONALIZATION</div><h2>{activeAdapter ? "Personalized profile" : "Standard recognition"}</h2></div><span className={activeAdapter ? "status-chip dark" : "status-chip"}>{activeAdapter ? "ON" : "OFF"}</span></div>
          {activeAdapter ? <div className="signer-grid"><div><span>PROFILE</span><strong>#{activeAdapter.id}</strong></div><div><span>LETTERS</span><strong>{activeAdapter.letters?.join(" · ") || "—"}</strong></div><div><span>EXAMPLES</span><strong>{activeAdapter.shots ? Object.values(activeAdapter.shots).reduce((sum: number, value: any) => sum + Number(value || 0), 0) : "—"}</strong></div></div> : <Empty text="Using the V3 base model. Calibration is optional and adds signer-specific prototype matching." />}
          <Link to="/calibration" className="text-btn">Personalize recognition →</Link>
        </section>
      </section>
    </div>
  </Page>;
}
