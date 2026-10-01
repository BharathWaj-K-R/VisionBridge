const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type LetterSample } from "../api";
import { useLandmarkSession } from "../useLandmarkSession";
import { Page } from "../components/Page";

export default function Calibration() {
  const session = useLandmarkSession(15);
  const [selectedLetter, setSelectedLetter] = useState("A");
  const [samples, setSamples] = useState<Record<string, number[][]>>({});
  const [userId, setUserId] = useState<number>();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("Capture three examples for each letter you want to recognize.");
  const [startedAt, setStartedAt] = useState<number | null>(null);

  useEffect(() => { void api.me().then((u) => setUserId(u.id)).catch(() => setMessage("Authentication session could not be verified. Please sign in again.")); }, []);

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
        <div className="camera-topline"><span><i /> {session.running ? "Camera is on" : "Camera is off"}</span><span>Letter {selectedLetter} · {selectedSamples.length}/3 examples</span></div>
        <div className="camera-shell calibration-camera"><video ref={session.videoRef} muted playsInline aria-label="Calibration camera preview" /><canvas ref={session.canvasRef} className="skeleton-overlay" aria-hidden="true" /><div className="camera-corner top-left">Show letter {selectedLetter}</div><div className="camera-corner top-right">{session.running ? "Live" : "Ready"}</div><div className="camera-corner bottom-left">Keep the gesture steady</div></div>
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