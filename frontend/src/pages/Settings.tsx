import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, clearLocalAuth, clearSessionHint } from "../api";
import { Empty, Page } from "../components/Page";
import { ErrorState } from "../components/SystemStates";
import { applyThemePreference, getThemePreference, type ThemePreference } from "../theme";

export default function Settings() {
  const [user, setUser] = useState<any>();
  const [adapters, setAdapters] = useState<any[]>([]);
  const [theme, setTheme] = useState<ThemePreference>(() => getThemePreference());
  const [contrast, setContrast] = useState(localStorage.getItem("visionbridge_contrast") || "normal");
  const [cameraFps, setCameraFps] = useState(localStorage.getItem("visionbridge_camera_fps") || "30");
  const [autoStart, setAutoStart] = useState(localStorage.getItem("visionbridge_auto_camera") === "1");
  const [saved, setSaved] = useState(false);
  const [message, setMessage] = useState("");
  const [loadError, setLoadError] = useState("");
  const refresh = () => Promise.all([api.me(), api.letterAdapters()]).then(([u,a]) => { setUser(u); setAdapters(a); setLoadError(""); }).catch((err) => { setUser(undefined); setAdapters([]); setLoadError(err instanceof Error ? err.message : "Settings could not be loaded."); });
  useEffect(() => { void refresh(); }, []);
  const chooseTheme = (next: ThemePreference) => {
    setTheme(next);
    applyThemePreference(next);
    setSaved(false);
  };

  const savePreferences = () => {
    applyThemePreference(theme);
    localStorage.setItem("visionbridge_camera_fps", cameraFps);
    localStorage.setItem("visionbridge_contrast", contrast);
    localStorage.setItem("visionbridge_auto_camera", autoStart ? "1" : "0");
    document.documentElement.dataset.contrast = contrast;
    setSaved(true); window.setTimeout(() => setSaved(false), 1800);
  };
  return <Page title="Settings" subtitle="Manage appearance, camera settings, account access, and recognition personalization.">{loadError && <ErrorState title="Settings could not be loaded" message="Check your connection and try again. No settings were changed." onRetry={() => void refresh()} home />}
    <div className="settings-grid">
      <section className="panel"><div className="eyebrow">ACCOUNT</div><h2>{user?.username || "Loading…"}</h2><p className="muted">{user?.email || "No email on this account yet."}</p><p className="account-id">Your account uses a server-side session.</p><div className="button-row"><button className="ghost-btn" onClick={() => void api.logout().then(() => { clearLocalAuth(); clearSessionHint(); window.location.assign("/login"); }).catch((err) => setMessage(err instanceof Error ? err.message : "Sign out failed. Please try again."))}>Sign out</button></div></section>
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">APPEARANCE</div><h2>Appearance</h2></div></div><div className="settings-form">
        <div className="settings-field">
          <span className="settings-field-label">Theme</span>
          <div className="theme-picker" role="radiogroup" aria-label="Theme preference">
            {([
              ["light", "Light", "Always use the light workspace"],
              ["dark", "Dark", "Always use the dark workspace"],
              ["system", "System", "Follow your operating system"],
            ] as const).map(([value, label, description]) => (
              <button key={value} type="button" role="radio" aria-checked={theme === value}
                className={theme === value ? "theme-option active" : "theme-option"}
                onClick={() => chooseTheme(value)}>
                <span className="theme-option-indicator" aria-hidden="true" />
                <span className="theme-option-title">{label}</span>
                <span className="theme-option-description">{description}</span>
              </button>
            ))}
          </div>
          <p className="settings-theme-note">Theme changes apply immediately and persist across reloads.</p>
        </div>
        <label>Text contrast<select value={contrast} onChange={e => { setContrast(e.target.value); document.documentElement.dataset.contrast = e.target.value; }}><option value="normal">Standard</option><option value="high">High contrast</option></select></label>
        <label>Camera frame rate<select value={cameraFps} onChange={e => setCameraFps(e.target.value)}><option value="15">15 fps · lower processing load</option><option value="30">30 fps · balanced</option><option value="60">60 fps · higher frame rate</option></select></label>
        <label className="setting-toggle"><input type="checkbox" checked={autoStart} onChange={e => setAutoStart(e.target.checked)} /> Start the camera automatically on Camera translation</label>
        <button className="primary-btn" onClick={savePreferences}>{saved ? "Saved" : "Save settings"}</button>
      </div></section>
      <section className="panel"><div className="eyebrow">PERSONALIZATION</div><h2>Recognition personalization</h2>{adapters.length ? <div className="adapter-list">{adapters.map(a => <div className="adapter-row" key={a.id}><div><strong>Adapter #{a.id}</strong><span>{a.letters ? a.letters.join(" · ") : ((a.calibration_seconds ?? 0)+"s")}</span></div><button className="ghost-btn" onClick={() => api.deleteAdapter(a.id).then(refresh)}>Delete</button></div>)}</div> : <Empty text="No personalized profiles yet. You can personalize recognition from the calibration page." />}<div className="button-row"><Link to="/personalization" className="text-btn">OPEN MY PROFILE →</Link><Link to="/calibration" className="text-btn">CALIBRATE A SIGNER →</Link></div></section>
      <section className="panel"><div className="eyebrow">PRIVACY</div><h2>Your browser permissions</h2><p className="muted">Camera and microphone access are controlled by your browser. You can change these permissions from the browser's site settings.</p><div className="settings-checks"><span>Camera · required for Camera translation</span><span>Microphone · required for Speak</span></div><details className="advanced-details"><summary>Security details</summary><div className="assistive-copy">Your signed-in session and recognition data are protected by the application's server-side session and request security controls.</div></details></section>
    </div>
  </Page>;
}