import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Empty, Loading, Metric, Page } from "../components/Page";
import { ErrorState } from "../components/SystemStates";
import Speakable from "../components/Speakable";

export default function Dashboard() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api.dashboard().then(setData).catch((e) => setError(e instanceof Error ? e.message : "We could not load your home screen."));
  }, []);

  return (
    <Page title="VisionBridge home" subtitle="Choose a communication method or review recent recognition.">
      {error ? <ErrorState title="Home could not be loaded" message="Check your connection and try again. Your saved data has not been changed." onRetry={() => { setError(""); api.dashboard().then(setData).catch((e) => setError(e instanceof Error ? e.message : "Home could not be loaded.")); }} home /> : !data ? <Loading /> : (
        <>
          <section className="welcome-panel" aria-labelledby="welcome-title">
            <div>
              <div className="eyebrow">Start here</div>
              <h2 id="welcome-title">Choose a communication method</h2>
              <p>Use the camera to recognize A–Z signs, or speak a sentence to build a sequence of sign references.</p>
            </div>
            <div className="communication-actions">
              <Link to="/translate" className="action-card">
                <span className="action-icon" aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false"><rect x="3.5" y="5" width="17" height="14" rx="1.5"/><path d="M8 16c1.2-1.8 2.4-2.7 3.6-2.7S13.9 14.2 15 16M7 9.5h.01M17 9.5h.01"/></svg></span>
                <span><h3>Camera translation</h3><p>Place a hand sign inside the camera frame to recognize an A–Z letter.</p></span>
                <span className="action-link">Open camera →</span>
              </Link>
              <Link to="/voice-to-sign" className="action-card">
                <span className="action-icon" aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false"><path d="M7 18.5 4.5 20l.7-2.9A7.5 7.5 0 1 1 19.5 12 7.5 7.5 0 0 1 7 18.5Z"/><path d="M8.5 10.5h7M8.5 13.5h5"/></svg></span>
                <span><h3>Speak</h3><p>Speak a sentence, then play it back as sign references.</p></span>
                <span className="action-link">Start speaking →</span>
              </Link>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <div><div className="eyebrow">Getting started</div><h2>Before you start</h2></div>
              <Link to="/settings" className="text-btn">Settings</Link>
            </div>
            <div className="getting-started-grid">
              <article className="tip-card"><span className="tip-number">1</span><strong>Allow device access</strong><p>Camera translation requires camera access. Speak requires microphone access.</p></article>
              <article className="tip-card"><span className="tip-number">2</span><strong>Use a clear camera view</strong><p>Use steady lighting and keep the full hand inside the frame.</p></article>
              <article className="tip-card"><span className="tip-number">3</span><strong>Calibrate when useful</strong><p>Calibration is optional. Use it when the base model needs to adapt to a signer.</p></article>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <div><div className="eyebrow">Recent activity</div><h2>Recent recognition</h2></div>
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
            ) : <Empty text="No recognition events yet. Open camera translation to begin." />}
          </section>

          <details className="panel advanced-details">
            <summary>Show recognition details</summary>
            <div className="metric-grid">
              <Metric label="Recognized" value={data.usage?.translation_events ?? 0} detail="Recent recognition events" />
              <Metric label="Confidence" value={data.usage?.average_confidence != null ? Math.round(data.usage.average_confidence * 100) + "%" : "—"} detail="Recent average" />
              <Metric label="Response time" value={data.usage?.average_latency_ms != null ? Math.round(data.usage.average_latency_ms) + " ms" : "—"} detail="Recent average" />
            </div>
          </details>
        </>
      )}
    </Page>
  );
}
