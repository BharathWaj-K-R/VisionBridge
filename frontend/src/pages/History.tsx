import { useEffect, useState } from "react";
import { api } from "../api";
import { Empty, Loading, Page } from "../components/Page";
import { EmptyState, ErrorState } from "../components/SystemStates";
import Speakable from "../components/Speakable";

export default function History() {
  const [data, setData] = useState<any>();
  const [query, setQuery] = useState("");
  const [exporting, setExporting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [message, setMessage] = useState("");
  const [loadError, setLoadError] = useState("");

  const refresh = () => {
    setLoadError("");
    return api.history().then(setData).catch((err) => {
      setData(undefined);
      setLoadError(err instanceof Error ? err.message : "History could not be loaded.");
    });
  };

  useEffect(() => { void refresh(); }, []);

  async function downloadCsv() {
    setExporting(true);
    setMessage("");
    try {
      const blob = await api.exportHistoryCsv();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "visionbridge-letter-history.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      setMessage("History CSV exported.");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "History CSV could not be exported.");
    } finally {
      setExporting(false);
    }
  }

  async function clearHistory() {
    if (!window.confirm("Clear all recognition history for this account? This permanently deletes the database history in API mode.")) return;
    setClearing(true);
    setMessage("");
    try {
      const result = await api.clearHistory();
      setData({ items: [] });
      setMessage(
        result.storage === "database"
          ? result.deleted + " history record" + (result.deleted === 1 ? "" : "s") + " deleted from the database."
          : result.deleted + " local history record" + (result.deleted === 1 ? "" : "s") + " cleared from this browser."
      );
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "History could not be cleared.");
    } finally {
      setClearing(false);
    }
  }

  const rows = data?.items?.filter((x: any) =>
    !query || String(x.predicted_text).toLowerCase().includes(query.toLowerCase())
  ) || [];

  return (
    <Page title="History" subtitle="Recorded letter predictions from the current signer session.">
      <section className="panel">
        <div className="panel-head">
          <div>
            <div className="eyebrow">EVENTS</div>
            <h2>History</h2>
          </div>
          <div className="history-actions">
            <input className="compact-input" placeholder="Search letters" value={query} onChange={e => setQuery(e.target.value)} />
            <button type="button" className="ghost-btn" onClick={downloadCsv} disabled={exporting || clearing}>
              {exporting ? "Exporting…" : "Export CSV"}
            </button>
            <button type="button" className="danger-btn" onClick={clearHistory} disabled={exporting || clearing || !rows.length}>
              {clearing ? "Clearing…" : "Clear history"}
            </button>
          </div>
        </div>

        {message && <div className="alert history-message" role="status">{message}</div>}

        {loadError ? <ErrorState title="History couldn't be loaded" message="Check your connection and try again. Your saved history has not been changed." onRetry={() => void refresh()} /> : !data ? <Loading /> : rows.length ? (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Time</th><th>Letter</th><th>Confidence</th><th>Latency</th></tr></thead>
              <tbody>
                {rows.map((row: any) => (
                  <tr key={row.id}>
                    <td>{new Date(row.created_at).toLocaleString()}</td>
                    <td><Speakable text={row.predicted_text} className="history-speakable" /></td>
                    <td>{Math.round((row.confidence || 0) * 100)}%</td>
                    <td>{Math.round(row.latency_ms || 0)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <EmptyState title="No history yet" message="Your recognized letters will appear here after you use Translate." to="/translate" actionLabel="Start translating" />}
      </section>
    </Page>
  );
}
