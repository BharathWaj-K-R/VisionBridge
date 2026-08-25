(function () {
  "use strict";
  const main = document.querySelector("main");
  if (!main || !window.VB_API_FETCH) return;

  const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[c]));
  const confidence = (v) => v == null ? "—" : `${Math.round(v * 100)}%`;

  function controls() {
    return `<div class="sb-card mb-4"><div class="card-body p-3"><div class="row g-3 align-items-center"><div class="col-md-6"><label class="visually-hidden" for="sbSearch">Search translations</label><input id="sbSearch" class="form-control" placeholder="Search transcripts or keywords" /></div><div class="col-md-3"><select id="sbDate" class="form-select"><option value="7d">Last 7 days</option><option value="30d">Last 30 days</option><option value="90d">Last 90 days</option><option value="all">All time</option></select></div><div class="col-md-3"><select id="sbSort" class="form-select"><option value="newest">Newest</option><option value="confidence">Confidence</option><option value="length">Length</option></select></div></div></div></div>`;
  }

  async function load() {
    const search = encodeURIComponent(document.querySelector("#sbSearch")?.value || "");
    const range = document.querySelector("#sbDate")?.value || "7d";
    const sort = document.querySelector("#sbSort")?.value || "newest";
    const response = await window.VB_API_FETCH(`/history?range=${range}&sort=${sort}&search=${search}`);
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    const items = data.items || [];
    const cards = items.length ? items.map((item) => `<div class="col-md-6 col-lg-4"><article class="sb-card h-100"><div class="card-body"><div class="d-flex justify-content-between"><span class="text-muted-2 small">${esc(new Date(item.created_at).toLocaleString())}</span><span class="sb-badge ${item.confidence >= .9 ? "success" : item.confidence >= .75 ? "" : "warning"}">${confidence(item.confidence)}</span></div><h3 class="h6 mt-3 mb-2">Translation #${item.id}</h3><p class="text-muted-2 small mb-3">${esc(item.predicted_text)}</p><div class="d-flex justify-content-between text-muted-2 small"><span>${item.used_adapter ? "Personalized" : "Base model"}</span><span>${item.latency_ms == null ? "—" : `${Math.round(item.latency_ms)} ms`}</span></div></div></article></div>`).join("") : '<div class="col-12"><div class="sb-card"><div class="card-body text-muted-2">No persisted translation events match these filters.</div></div></div>';
    main.innerHTML = `<header class="d-flex flex-wrap justify-content-between align-items-end gap-3 mb-4"><div><div class="text-muted-2 small">Archive</div><h1 class="h2 mb-0">Translation history</h1><p class="text-muted-2 mb-0">Persisted server-side transcript telemetry for your account.</p></div><button class="btn btn-ghost-sb" data-export-csv type="button">Export CSV</button></header>${controls()}<div class="row g-3">${cards}</div>`;
    main.querySelector("[data-export-csv]")?.addEventListener("click", () => exportCsv().catch(showError));
    ["#sbSearch", "#sbDate", "#sbSort"].forEach((selector) => document.querySelector(selector)?.addEventListener(selector === "#sbSearch" ? "input" : "change", () => load().catch(showError)));
  }

  async function exportCsv() {
    // /history/export.csv requires Bearer auth (read from localStorage, not
    // a cookie) — a plain <a href> download never sends that header and
    // would 401. Fetch it through VB_API_FETCH (which does attach it), then
    // trigger the download client-side from the response body.
    const response = await window.VB_API_FETCH("/history/export.csv");
    if (!response.ok) throw new Error(await response.text());
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "visionbridge-history.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }
  function showError(error) { main.insertAdjacentHTML("afterbegin", `<div class="alert alert-danger">History could not be loaded: ${esc(error.message || error)}</div>`); }
  load().catch(showError);
})();
