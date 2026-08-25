(function () {
  "use strict";
  const main = document.querySelector("main");
  if (!main || !window.VB_API_FETCH) return;
  const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[c]));

  async function load() {
    const response = await window.VB_API_FETCH("/evaluation");
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    const modelReady = data.model?.available === true;
    const adapterRows = data.adapters?.length ? data.adapters.map((adapter) => `<tr><td>#${adapter.id}</td><td>${Math.round(adapter.calibration_seconds)} s</td><td>${adapter.param_count ?? "—"}</td><td>${adapter.accuracy_gain_pct == null ? "Not measured" : `${adapter.accuracy_gain_pct}%`}</td></tr>`).join("") : '<tr><td colspan="4" class="text-muted-2">No persisted adapters.</td></tr>';
    main.innerHTML = `<header class="mb-5"><div class="text-muted-2 small">Evaluation</div><h1 class="h2 mb-2">Measured system state</h1><p class="text-muted-2 mb-0">This page reports evidence that actually exists in the repository/database. It does not invent benchmark numbers.</p></header>
      <div class="row g-4 mb-5"><div class="col-lg-6"><div class="sb-card h-100"><div class="card-body"><span class="text-muted-2 small">Base model</span><h2 class="h4 mt-1">${modelReady ? "Checkpoint available" : "Checkpoint unavailable"}</h2><span class="sb-badge ${modelReady ? "success" : "warning"}">${esc(data.model?.status || "unknown")}</span><p class="text-muted-2 small mt-3 mb-0">Model contract and availability are checked from the configured checkpoint and vocabulary.</p></div></div></div><div class="col-lg-6"><div class="sb-card h-100"><div class="card-body"><span class="text-muted-2 small">Benchmark evidence</span><h2 class="h4 mt-1">${data.evaluation_data_available ? "Available" : "Not available"}</h2><p class="text-muted-2 small mb-0">${esc(data.message)}</p></div></div></div></div>
      <section class="sb-card"><div class="card-body"><h2 class="h5 mb-3">Persisted adapter telemetry</h2><div class="table-responsive"><table class="sb-table"><thead><tr><th>Adapter</th><th>Calibration</th><th>Parameters</th><th>Measured accuracy gain</th></tr></thead><tbody>${adapterRows}</tbody></table></div></div></section>`;
  }
  load().catch((error) => main.insertAdjacentHTML("afterbegin", `<div class="alert alert-danger">Evaluation data could not be loaded: ${esc(error.message || error)}</div>`));
})();
