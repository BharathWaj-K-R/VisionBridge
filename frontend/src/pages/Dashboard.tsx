import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Empty, Loading, Metric, Page } from "../components/Page";
import Speakable from "../components/Speakable";

export default function Dashboard() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api.dashboard().then(setData).catch((e) => setError(e instanceof Error ? e.message : "We could not load your home screen."));
  }, []);

  return (
    <Page title="Welcome to VisionBridge" subtitle="Choose how you want to communicate today.">
      {error ? <div className="alert error" role="alert">{error}</div> : !data ? <Loading /> : (
        <>
          <section className="welcome-panel" aria-labelledby="welcome-title">
            <div>
              <div className="eyebrow">START HERE</div>
              <h2 id="welcome-title">How would you like to communicate?</h2>
              <p>Use the camera for hand signs, or speak naturally and turn your words into a sequence of sign references.</p>
            </div>
            <div className="communication-actions">
              <Link to="/translate" className="action-card">
                <span className="action-icon" aria-hidden="true">⌁</span>
                <span><h3>Translate with camera</h3><p>Show your hand signs inside the frame and VisionBridge will recognize the letter.</p></span>
                <span className="action-link">Open camera →</span>
              </Link>
              <Link to="/voice-to-sign" className="action-card">
                <span className="action-icon" aria-hidden="true">◉</span>
                <span><h3>Speak</h3><p>Speak a sentence, then play it back as sign references.</p></span>
                <span className="action-link">Start speaking →</span>
              </Link>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <div><div className="eyebrow">GETTING STARTED</div><h2>Three simple tips</h2></div>
              <Link to="/settings" className="text-btn">Settings</Link>
            </div>
            <div className="getting-started-grid">
              <article className="tip-card"><span className="tip-number">1</span><strong>Allow access</strong><p>Camera mode needs camera permission. Speak mode needs microphone permission.</p></article>
              <article className="tip-card"><span className="tip-number">2</span><strong>Use a clear view</strong><p>Good light and hands fully inside the camera frame make recognition easier.</p></article>
              <article className="tip-card"><span className="tip-number">3</span><strong>Personalize later</strong><p>Calibration is optional. Use it when you want recognition tuned to your signing.</p></article>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <div><div className="eyebrow">RECENT ACTIVITY</div><h2>Recent letters</h2></div>
              <Link to="/history" className="text-btn">View history</Link>
            </div>
            {data.recent_activity?.length ? (
              <div className="activity-list">
                {data.recent_activity.map((item: any) => (
                  <div className="activity-row" key={item.id}>
                    <div><Speakable text={item.predicted_text} className="activity-speakable" /><span>{new Date(item.created_at).toLocaleString()}</span></div>
                    <span className="activity-meta">{Math.round((item.confidence || 0) * 100)}%</span>
                  </div>
                ))}
              </div>
            ) : <Empty text="No letters recognized yet. Start with the camera." />}
          </section>

          <details className="panel advanced-details">
            <summary>Show recognition details</summary>
            <div className="metric-grid">
              <Metric label="Recognized" value={data.usage?.translation_events ?? 0} detail="recent recognition events" />
              <Metric label="Confidence" value={data.usage?.average_confidence != null ? Math.round(data.usage.average_confidence * 100) + "%" : "—"} detail="recent average" />
              <Metric label="Response time" value={data.usage?.average_latency_ms != null ? Math.round(data.usage.average_latency_ms) + " ms" : "—"} detail="recent average" />
            </div>
          </details>
        </>
      )}
    </Page>
  );
}
