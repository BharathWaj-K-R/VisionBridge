(function () {
  "use strict";
  const main = document.querySelector("main");
  if (!main || !window.VB_API_FETCH) return;

  function esc(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[c]));
  }
  function pct(value) { return value == null ? "—" : `${Math.round(value * 100)}%`; }

  async function load() {
    try {
      const response = await window.VB_API_FETCH("/dashboard");
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      const modelReady = data.model?.available === true;
      const avgLatency = data.usage?.average_latency_ms;
      const avgConfidence = data.usage?.average_confidence;
      const activity = data.recent_activity || [];
      main.querySelector("h1")?.replaceChildren(document.createTextNode(`Welcome back, ${data.user.username}.`));
      main.innerHTML = `<header class="d-flex flex-wrap justify-content-between align-items-end gap-3 mb-5"><div><div class="text-muted-2 small">Workspace</div><h1 class="h2 mb-1">Welcome back, ${esc(data.user.username)}.</h1><p class="text-muted-2 mb-0">Live data from your VisionBridge account.</p></div><div class="d-flex gap-2"><a href="/pages/translate.html" class="btn btn-primary-sb">Start Translation</a><a href="/pages/calibration.html" class="btn btn-ghost-sb">Calibration</a></div></header>
      <section class="row g-4 mb-5"><div class="col-lg-4"><div class="sb-card h-100"><div class="card-body"><span class="text-muted-2 small">Model status</span><h3 class="h4 mb-2">${modelReady ? "Ready" : "Unavailable"}</h3><span class="sb-badge ${modelReady ? "success" : "warning"}">${esc(data.model?.status || "unknown")}</span>${data.adapter ? `<p class="text-muted-2 small mt-3 mb-0">Adapter #${data.adapter.id} · ${Math.round(data.adapter.calibration_seconds)} s calibration</p>` : '<p class="text-muted-2 small mt-3 mb-0">No signer adapter has been created.</p>'}</div></div></div>
      <div class="col-lg-4"><div class="sb-card h-100"><div class="card-body"><span class="text-muted-2 small">Translation events</span><h3 class="h4 mb-1">${data.usage.translation_events}</h3><p class="text-muted-2 small mb-0">Persisted inference requests for this account.</p><div class="mt-4 small">Average confidence: <strong>${pct(avgConfidence)}</strong></div><div class="small">Average model latency: <strong>${avgLatency == null ? "—" : `${Math.round(avgLatency)} ms`}</strong></div></div></div></div>
      <div class="col-lg-4"><div class="sb-card h-100"><div class="card-body"><span class="text-muted-2 small">Privacy boundary</span><h3 class="h4 mb-2">Keypoints only</h3><p class="text-muted-2 small mb-0">The translation API stores transcript telemetry, not webcam frames. Browser extraction remains separate from the backend.</p></div></div></div></section>
      <section><div class="d-flex justify-content-between align-items-center mb-3"><h2 class="h5 mb-0">Recent translation events</h2><a href="/pages/history.html" class="small">View all</a></div><div class="sb-card"><div class="card-body p-0"><div class="table-responsive"><table class="sb-table"><thead><tr><th>Time</th><th>Prediction</th><th>Confidence</th><th>Latency</th><th>Mode</th></tr></thead><tbody>${activity.length ? activity.map((row) => `<tr><td>${esc(new Date(row.created_at).toLocaleString())}</td><td>${esc(row.predicted_text)}</td><td>${pct(row.confidence)}</td><td>${row.latency_ms == null ? "—" : `${Math.round(row.latency_ms)} ms`}</td><td>${row.used_adapter ? "Personalized" : "Base"}</td></tr>`).join("") : '<tr><td colspan="5" class="text-muted-2">No translation events yet.</td></tr>'}</tbody></table></div></div></div></section>`;
    } catch (error) {
      main.insertAdjacentHTML("afterbegin", `<div class="alert alert-danger">Dashboard data could not be loaded: ${esc(error.message || error)}</div>`);
    }
  }
  load();
})();
