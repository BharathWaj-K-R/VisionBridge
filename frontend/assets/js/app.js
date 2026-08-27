/* VisionBridge — shared UI behavior
   No jQuery, no packages. Vanilla JS only. */

(function () {
  "use strict";

  const NAV_LINKS = [
    { href: "/index.html", label: "Home" },
    { href: "/pages/dashboard.html", label: "Dashboard" },
    { href: "/pages/translate.html", label: "Translate" },
    { href: "/pages/calibration.html", label: "Calibration" },
    { href: "/pages/history.html", label: "History" },
    { href: "/pages/users.html", label: "Profiles" },
    { href: "/pages/evaluation.html", label: "Evaluation" },
    { href: "/pages/settings.html", label: "Settings" },
  ];
  const PRIVATE_PAGES = new Set(["dashboard.html", "calibration.html", "history.html", "users.html", "evaluation.html"]);
  const API_TIMEOUT_MS = 20_000;

  function authToken() {
    try { return localStorage.getItem("visionbridge.accessToken"); } catch { return null; }
  }
  function authHeaders() {
    const token = authToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  }
  async function apiFetch(path, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
    const requestOptions = {
      ...options,
      headers: { ...(options.headers || {}), ...authHeaders() },
    };
    if (!requestOptions.signal) requestOptions.signal = controller.signal;

    try {
      const response = await fetch(`${window.VB_API_BASE_URL}${path}`, requestOptions);
      if (response.status === 401) {
        try { localStorage.removeItem("visionbridge.accessToken"); } catch { /* ignore */ }
        if (PRIVATE_PAGES.has(window.location.pathname.split("/").pop())) window.location.href = "/pages/auth.html";
      }
      return response;
    } finally {
      clearTimeout(timeoutId);
    }
  }
  window.VB_AUTH_TOKEN = authToken;
  window.VB_API_FETCH = apiFetch;

  function renderNav() {
    const mount = document.querySelector("[data-sb-nav]");
    if (!mount) return;
    const currentPath = window.location.pathname.replace(/\/$/, "");
    const links = NAV_LINKS.map((link) => {
      const isActive = currentPath.endsWith(link.href.split("/").pop());
      return `<li class="nav-item"><a class="nav-link ${isActive ? "active" : ""}" href="${link.href}">${link.label}</a></li>`;
    }).join("");
    const authenticated = Boolean(authToken());
    mount.innerHTML = `<nav class="sb-nav" aria-label="Primary">
      <div class="container-sb d-flex align-items-center justify-content-between py-3">
        <a class="navbar-brand" href="/index.html" aria-label="VisionBridge home"><span class="sb-logo-mark" aria-hidden="true"><i class="bi bi-soundwave"></i></span> VisionBridge</a>
        <button class="btn btn-ghost-sb d-lg-none btn-icon" type="button" aria-expanded="false" aria-controls="sbNavList" data-sb-nav-toggle><i class="bi bi-list"></i><span class="visually-hidden">Toggle navigation</span></button>
        <ul id="sbNavList" class="nav align-items-center gap-1 d-none d-lg-flex" role="menubar">${links}</ul>
        <div class="d-none d-lg-flex align-items-center gap-2"><a href="/pages/translate.html" class="btn btn-primary-sb btn-icon btn-ripple"><i class="bi bi-broadcast"></i> Start Translation</a>${authenticated ? '<button class="btn btn-ghost-sb btn-icon" data-sb-logout type="button"><i class="bi bi-box-arrow-right"></i> Sign out</button>' : '<a href="/pages/auth.html" class="btn btn-ghost-sb btn-icon"><i class="bi bi-person"></i> Sign in</a>'}</div>
      </div>
      <div class="container-sb d-lg-none pb-3 d-none" data-sb-nav-mobile><ul class="nav flex-column gap-1">${links}</ul><a href="/pages/translate.html" class="btn btn-primary-sb w-100 mt-2">Start Translation</a>${authenticated ? '<button class="btn btn-ghost-sb w-100 mt-2" data-sb-logout type="button">Sign out</button>' : '<a href="/pages/auth.html" class="btn btn-ghost-sb w-100 mt-2">Sign in</a>'}</div>
    </nav>`;
    const toggle = mount.querySelector("[data-sb-nav-toggle]");
    const mobile = mount.querySelector("[data-sb-nav-mobile]");
    if (toggle && mobile) toggle.addEventListener("click", () => { const open = mobile.classList.toggle("d-none") === false; toggle.setAttribute("aria-expanded", String(open)); });
    mount.querySelectorAll("[data-sb-logout]").forEach((button) => button.addEventListener("click", () => { try { localStorage.removeItem("visionbridge.accessToken"); } catch { /* ignore */ } window.location.href = "/index.html"; }));
    const navEl = mount.querySelector(".sb-nav");
    if (navEl) { const onScroll = () => navEl.classList.toggle("scrolled", window.scrollY > 4); onScroll(); window.addEventListener("scroll", onScroll, { passive: true }); }
  }

  function enforcePrivatePage() {
    const page = window.location.pathname.split("/").pop();
    if (PRIVATE_PAGES.has(page) && !authToken()) window.location.replace("/pages/auth.html");
  }
  function renderFooter() {
    const mount = document.querySelector("[data-sb-footer]");
    if (!mount) return;
    mount.innerHTML = `<footer class="sb-footer"><div class="container-sb d-flex flex-wrap gap-3 justify-content-between align-items-center"><div class="d-flex align-items-center gap-2"><span class="sb-logo-mark"><i class="bi bi-soundwave"></i></span><span>&copy; ${new Date().getFullYear()} VisionBridge. Designed for accessible communication.</span></div><div class="d-flex gap-3"><a href="/pages/settings.html">Settings</a><a href="#">Privacy</a><a href="#">Terms</a></div></div></footer>`;
  }
  function initRipples() {
    document.addEventListener("click", (e) => { const btn = e.target.closest(".btn-ripple"); if (!btn) return; const rect = btn.getBoundingClientRect(); btn.style.setProperty("--x", `${((e.clientX - rect.left) / rect.width) * 100}%`); btn.style.setProperty("--y", `${((e.clientY - rect.top) / rect.height) * 100}%`); btn.classList.remove("is-rippling"); void btn.offsetWidth; btn.classList.add("is-rippling"); setTimeout(() => btn.classList.remove("is-rippling"), 500); });
  }
  function initReveal() {
    const items = document.querySelectorAll("[data-sb-reveal]"); if (!items.length || !("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((entries) => entries.forEach((entry) => { if (entry.isIntersecting) { entry.target.classList.add("sb-fade-up"); io.unobserve(entry.target); } }), { threshold: 0.12 }); items.forEach((el) => io.observe(el));
  }
  function initRings() { document.querySelectorAll(".sb-progress-ring[data-value]").forEach((el) => { const val = Math.max(0, Math.min(100, Number(el.dataset.value) || 0)); el.style.setProperty("--value", val); const valNode = el.querySelector(".val"); if (valNode && !valNode.dataset.static) valNode.textContent = `${val}%`; }); }
  function initCopy() { document.querySelectorAll("[data-sb-copy]").forEach((btn) => btn.addEventListener("click", async () => { const target = document.querySelector(btn.getAttribute("data-sb-copy")); const text = target ? target.innerText : btn.getAttribute("data-copy-text") || ""; try { await navigator.clipboard.writeText(text); const original = btn.innerHTML; btn.innerHTML = '<i class="bi bi-check2"></i> Copied'; setTimeout(() => (btn.innerHTML = original), 1400); } catch (err) { console.warn("Copy failed", err); } })); }
  function initApiEndpointSettings() {
    const input = document.querySelector("#apiUrl"), button = document.querySelector("[data-sb-save-api-url]"), status = document.querySelector("[data-sb-api-url-status]"); if (!input || !button) return;
    input.value = window.VB_API_BASE_URL || input.value;
    button.addEventListener("click", () => { let endpoint; try { endpoint = new URL(input.value); } catch { if (status) status.textContent = "Enter a valid http(s) endpoint URL."; return; } if (!/^https?:$/.test(endpoint.protocol)) { if (status) status.textContent = "Endpoint URL must use http or https."; return; } const normalized = endpoint.toString().replace(/\/$/, ""); try { localStorage.setItem("visionbridge.apiBaseUrl", normalized); window.VB_API_BASE_URL = normalized; if (status) status.textContent = "Endpoint saved for future translation requests."; } catch { if (status) status.textContent = "Browser storage is unavailable; the endpoint was not saved."; } });
  }
  document.addEventListener("DOMContentLoaded", () => { enforcePrivatePage(); renderNav(); renderFooter(); initRipples(); initReveal(); initRings(); initCopy(); initApiEndpointSettings(); });
})();
