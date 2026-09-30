import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Empty, Loading, Metric, Page } from "../components/Page";
import Speakable from "../components/Speakable";

export default function Dashboard() {
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
      <section className="panel"><div className="panel-head"><div><div className="eyebrow">RECENT</div><h2>Recognition events</h2></div><Link to="/history" className="text-btn">View history</Link></div>{data.recent_activity?.length ? <div className="activity-list">{data.recent_activity.map((item: any) => <div className="activity-row" key={item.id}><div><Speakable text={item.predicted_text} className="activity-speakable" /><span>{new Date(item.created_at).toLocaleString()}</span></div><span className="activity-meta">{Math.round((item.confidence || 0) * 100)}%</span></div>)}</div> : <Empty text="No letter predictions yet." />}</section>
    </>}
  </Page>;
}