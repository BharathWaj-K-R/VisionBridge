(function () {
  "use strict";
  const main = document.querySelector("main");
  if (!main || !window.VB_API_FETCH) return;
  const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[c]));

  async function load() {
    const [meResponse, adaptersResponse] = await Promise.all([
      window.VB_API_FETCH("/users/me"),
      window.VB_API_FETCH("/users/me/adapters"),
    ]);
    if (!meResponse.ok || !adaptersResponse.ok) throw new Error("Unable to load signer profile data");
    const me = await meResponse.json();
    const adapters = await adaptersResponse.json();
    const cards = adapters.length ? adapters.map((adapter) => `<div class="col-md-6 col-lg-4"><article class="sb-card h-100"><div class="card-body"><div class="d-flex align-items-center gap-3"><div class="sb-avatar">${esc(me.username.slice(0, 2).toUpperCase())}</div><div><h3 class="h6 mb-0">${esc(me.username)}</h3><div class="text-muted-2 small">Adapter #${adapter.id}</div></div></div><div class="mt-4 small text-muted-2">Calibration: <strong>${Math.round(adapter.calibration_seconds)} s</strong></div><div class="small text-muted-2">Parameters: <strong>${adapter.param_count ?? "—"}</strong></div><div class="small text-muted-2">Measured gain: <strong>${adapter.accuracy_gain_pct == null ? "Not measured" : `${adapter.accuracy_gain_pct}%`}</strong></div><button class="btn btn-ghost-sb btn-sm mt-4 w-100" data-delete-adapter="${adapter.id}">Delete adapter</button></div></article></div>`).join("") : '<div class="col-12"><div class="sb-card"><div class="card-body text-muted-2">No personal adapters exist yet. Start calibration to create one.</div></div></div>';
    main.innerHTML = `<header class="d-flex flex-wrap justify-content-between align-items-end gap-3 mb-4"><div><div class="text-muted-2 small">Signer profile</div><h1 class="h2 mb-1">${esc(me.username)}</h1><p class="text-muted-2 mb-0">These are the adapters actually persisted for your account.</p></div><a class="btn btn-primary-sb" href="/pages/calibration.html">Start calibration</a></header><div class="row g-3">${cards}</div>`;
    main.querySelectorAll("[data-delete-adapter]").forEach((button) => button.addEventListener("click", async () => {
      if (!window.confirm("Delete this adapter? Its stored weights will no longer be selectable.")) return;
      const id = button.dataset.deleteAdapter;
      const response = await window.VB_API_FETCH(`/users/me/adapters/${id}`, { method: "DELETE" });
      if (!response.ok) throw new Error(await response.text());
      load().catch(showError);
    }));
  }
  function showError(error) { main.insertAdjacentHTML("afterbegin", `<div class="alert alert-danger">Profile data could not be loaded: ${esc(error.message || error)}</div>`); }
  load().catch(showError);
})();
