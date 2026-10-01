import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, clearLocalAuth, clearSessionHint } from "../api";
import { Empty, Page } from "../components/Page";

export default function Settings() {
  const [user, setUser] = useState<any>();
  const [adapters, setAdapters] = useState<any[]>([]);
  const [theme, setTheme] = useState(localStorage.getItem("visionbridge_theme") || "system");
  const [cameraFps, setCameraFps] = useState(localStorage.getItem("visionbridge_camera_fps") || "30");
  const [autoStart, setAutoStart] = useState(localStorage.getItem("visionbridge_auto_camera") === "1");
  const [saved, setSaved] = useState(false);
  const [message, setMessage] = useState("");
  const [loadError, setLoadError] = useState("");
  const refresh = () => Promise.all([api.me(), api.letterAdapters()]).then(([u,a]) => { setUser(u); setAdapters(a); setLoadError(""); }).catch((err) => { setUser(undefined); setAdapters([]); setLoadError(err instanceof Error ? err.message : "Settings could not be loaded."); });
  useEffect(() => { void refresh(); }, []);
  const savePreferences = () => {
    localStorage.setItem("visionbridge_theme", theme);
    localStorage.setItem("visionbridge_camera_fps", cameraFps);
    localStorage.setItem("visionbridge_contrast", document.documentElement.dataset.contrast === "high" ? "high" : "normal");
    localStorage.setItem("visionbridge_auto_camera", autoStart ? "1" : "0");
    document.documentElement.dataset.theme = theme;
    setSaved(true); window.setTimeout(() => setSaved(false), 1800);
  };
  return <Page title="Settings" subtitle="Adjust how VisionBridge looks and behaves.">{loadError && <div className="alert error" role="alert">{loadError}</div>}
    <div className="settings-grid">
      <section className="panel"><div className="eyebrow">ACCOUNT</div><h2>{user?.username || "Loading…"}</h2><p className="muted">{user?.email || "No email on this account yet."}</p><p className="account-id">Your account is signed in securely.</p><div className="button-row"><button className="ghost-btn" onClick={() => void api.logout().then(() => { clearLocalAuth(); clearSessionHint(); window.location.assign("/login"); }).catch((err) => setMessage(err instanceof Error ? err.message : "Sign out failed. Please try again."))}>Sign out</button></div></section>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">APPEARANCE</div><h2>Make it comfortable</h2></div></div><div className="settings-form">
        <label>Theme<select value={theme} onChange={e => setTheme(e.target.value)}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
        <label>Text contrast<select value={localStorage.getItem("visionbridge_contrast") || "normal"} onChange={e => { localStorage.setItem("visionbridge_contrast", e.target.value); document.documentElement.dataset.contrast = e.target.value; }}><option value="normal">Standard</option><option value="high">High contrast</option></select></label>
        <label>Camera smoothness<select value={cameraFps} onChange={e => setCameraFps(e.target.value)}><option value="15">15 frames · lower battery use</option><option value="30">30 frames · balanced</option><option value="60">60 frames · smoother motion</option></select></label>
        <label className="setting-toggle"><input type="checkbox" checked={autoStart} onChange={e => setAutoStart(e.target.checked)} /> Start camera automatically on Translate</label>
        <button className="primary-btn" onClick={savePreferences}>{saved ? "Saved" : "Save settings"}</button>
      </div></section>
      <section className="panel"><div className="eyebrow">PERSONALIZATION</div><h2>Make recognition more personal</h2>{adapters.length ? <div className="adapter-list">{adapters.map(a => <div className="adapter-row" key={a.id}><div><strong>Adapter #{a.id}</strong><span>{a.letters ? a.letters.join(" · ") : ((a.calibration_seconds ?? 0)+"s")}</span></div><button className="ghost-btn" onClick={() => api.deleteAdapter(a.id).then(refresh)}>Delete</button></div>)}</div> : <Empty text="No personalized profiles yet. You can personalize recognition from the calibration page." />}<div className="button-row"><Link to="/personalization" className="text-btn">OPEN MY PROFILE →</Link><Link to="/calibration" className="text-btn">CALIBRATE A SIGNER →</Link></div></section>
      <section className="panel"><div className="eyebrow">PRIVACY</div><h2>Your browser permissions</h2><p className="muted">Authentication uses an HttpOnly session cookie. Recognition history and signer adapters are stored against your account on the production backend.</p><div className="settings-checks"><span>Session cookie · HttpOnly</span><span>CSRF protection · enabled</span><span>Camera · browser permission required</span></div></section>
    </div>
  </Page>;
}