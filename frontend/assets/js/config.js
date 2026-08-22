/**
 * Backend API base URL.
 * Local dev: FastAPI dev server. Production: the backend's Render URL.
 * Update the production fallback below once you know your actual
 * silentbridge-backend Render URL (shown on its dashboard page).
 *
 * NOTE: this fallback URL intentionally still says "silentbridge" — it is
 * the literal hostname of the currently-deployed Render service. If/when
 * that service is renamed on Render's side, update this string (and
 * render.yaml's `name:` + ALLOWED_ORIGINS) to match at the same time.
 */
window.VB_API_BASE_URL =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000/api/v1"
    : "https://silentbridge-backend-qpsn.onrender.com/api/v1";
