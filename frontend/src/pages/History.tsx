import { useEffect, useState } from "react";
import { api } from "../api";
import { Empty, Loading, Page } from "../components/Page";

export default function History() {
  const [data, setData] = useState<any>(); const [query, setQuery] = useState(""); const [exporting, setExporting] = useState(false);
  useEffect(() => { api.history().then(setData).catch(() => setData({ items: [] })); }, []);
  async function downloadCsv() { setExporting(true); try { const blob = await api.exportHistoryCsv(); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = "visionbridge-letter-history.csv"; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url); } finally { setExporting(false); } }
  const rows = data?.items?.filter((x: any) => !query || String(x.predicted_text).toLowerCase().includes(query.toLowerCase())) || [];
  return <Page title="History" subtitle="Recorded letter predictions from the current signer session."><section className="panel"><div className="panel-head"><div><div className="eyebrow">EVENTS</div><h2>History</h2></div><div style={{ display: "flex", gap: ".5rem" }}><input className="compact-input" placeholder="Search letters" value={query} onChange={e => setQuery(e.target.value)} /><button type="button" className="ghost-btn" onClick={downloadCsv} disabled={exporting}>{exporting ? "Exporting…" : "Export CSV"}</button></div></div>{!data ? <Loading /> : rows.length ? <div className="table-wrap"><table><thead><tr><th>Time</th><th>Letter</th><th>Confidence</th><th>Latency</th></tr></thead><tbody>{rows.map((row: any) => <tr key={row.id}><td>{new Date(row.created_at).toLocaleString()}</td><td><strong>{row.predicted_text}</strong></td><td>{Math.round((row.confidence || 0) * 100)}%</td><td>{Math.round(row.latency_ms || 0)} ms</td></tr>)}</tbody></table></div> : <Empty text="No letter predictions yet." />}</section></Page>;
}