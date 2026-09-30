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
  const refresh = () => Promise.all([api.me(), api.letterAdapters()]).then(([u,a]) => { setUser(u); setAdapters(a); }).catch(() => { setUser(undefined); setAdapters([]); });
  useEffect(() => { void refresh(); }, []);
  const savePreferences = () => {
    localStorage.setItem("visionbridge_theme", theme);
    localStorage.setItem("visionbridge_camera_fps", cameraFps);
    localStorage.setItem("visionbridge_auto_camera", autoStart ? "1" : "0");
    document.documentElement.dataset.theme = theme;
    setSaved(true); window.setTimeout(() => setSaved(false), 1800);
  };
  return <Page title="Settings" subtitle="Account, appearance, camera behavior, and signer profile controls.">
    <div className="settings-grid">
      <section className="panel"><div className="eyebrow">ACCOUNT</div><h2>{user?.username || "Loading…"}</h2><p className="muted">{user?.email || "No email on this account yet."}</p><p className="mono">Account ID {user?.id ?? "—"}</p><div className="button-row"><button className="ghost-btn" onClick={() => void api.logout().then(() => { clearLocalAuth(); clearSessionHint(); window.location.assign("/login"); }).catch((err) => setMessage(err instanceof Error ? err.message : "Sign out failed. Please try again."))}>Sign out</button></div></section>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">APPEARANCE</div><h2>Interface</h2></div></div><div className="settings-form">
        <label>Theme<select value={theme} onChange={e => setTheme(e.target.value)}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
        <label>Recognition FPS<select value={cameraFps} onChange={e => setCameraFps(e.target.value)}><option value="15">15 FPS · battery friendly</option><option value="30">30 FPS · balanced</option><option value="60">60 FPS · high performance</option></select></label>
        <label className="setting-toggle"><input type="checkbox" checked={autoStart} onChange={e => setAutoStart(e.target.checked)} /> Auto-start camera on Live Translate</label>
        <button className="primary-btn" onClick={savePreferences}>{saved ? "Saved" : "Save settings"}</button>
      </div></section>
      <section className="panel"><div className="eyebrow">SIGNER PROFILES</div><h2>Few-shot adapters</h2>{adapters.length ? <div className="adapter-list">{adapters.map(a => <div className="adapter-row" key={a.id}><div><strong>Adapter #{a.id}</strong><span>{a.letters ? a.letters.join(" · ") : ((a.calibration_seconds ?? 0)+"s")}</span></div><button className="ghost-btn" onClick={() => api.deleteAdapter(a.id).then(refresh)}>Delete</button></div>)}</div> : <Empty text="No adapters yet. Calibrate a few letters first." />}<div className="button-row"><Link to="/personalization" className="text-btn">OPEN MY PROFILE →</Link><Link to="/calibration" className="text-btn">CALIBRATE A SIGNER →</Link></div></section>
      <section className="panel"><div className="eyebrow">PRIVACY & SESSION</div><h2>Browser controls</h2><p className="muted">Authentication uses an HttpOnly session cookie. Recognition history and signer adapters are stored against your account on the production backend.</p><div className="settings-checks"><span>Session cookie · HttpOnly</span><span>CSRF protection · enabled</span><span>Camera · browser permission required</span></div></section>
    </div>
  </Page>;
}