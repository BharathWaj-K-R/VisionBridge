import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { BrowserLetterAdapter, type BrowserLetterModel } from "../browserModel";
import { normalizeHandPair, type TrackerVisual } from "../landmarks";
import { useLandmarkSession } from "../useLandmarkSession";
import { Empty, Page } from "../components/Page";

export default function Recognize() {
  const configuredFps = Number(localStorage.getItem("visionbridge_camera_fps") || "30");
  const [trackerVisual, setTrackerVisual] = useState<TrackerVisual>(() => {
    const saved = localStorage.getItem("visionbridge_tracker_visual");
    return saved === "anatomy" || saved === "holographic" ? saved : "neon";
  });
  const session = useLandmarkSession(configuredFps, trackerVisual, { prediction, confidence });

  useEffect(() => {
    localStorage.setItem("visionbridge_tracker_visual", trackerVisual);
  }, [trackerVisual]);
  const latestFrame = session.latestFrame;
  useEffect(() => {
    if (localStorage.getItem("visionbridge_auto_camera") === "1" && !session.running) {
      void session.start().catch(() => undefined);
    }
  }, []);
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
          <div className="tracker-visual-controls">
            <div>
              <span className="eyebrow">TRACKER VISUAL</span>
              <strong>{trackerVisual === "anatomy" ? "DIGITAL ANATOMY" : trackerVisual === "holographic" ? "HOLOGRAPHIC ENVELOPE" : "NEON NODE"}</strong>
            </div>
            <div className="tracker-visual-options">
              {([
                ["anatomy", "01 · ANATOMY", "Joint rings + cyan structural rig"],
                ["neon", "02 · NEON NODE", "Fast green nodes + motion trails"],
                ["holographic", "03 · HOLOGRAPHIC", "Purple geometric volume"],
              ] as Array<[TrackerVisual, string, string]>).map(([value, label, description]) => (
                <button
                  key={value}
                  type="button"
                  className={trackerVisual === value ? "active" : ""}
                  onClick={() => setTrackerVisual(value)}
                  title={description}
                  aria-pressed={trackerVisual === value}
                >
                  {label}
                </button>
              ))}
            </div>
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
