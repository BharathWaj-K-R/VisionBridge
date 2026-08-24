/**
 * Backend API base URL.
 * Local dev: FastAPI dev server. Production: the backend's Render URL.
 * Update the production fallback below once you know your actual
 * visionbridge-backend Render URL (shown on its dashboard page).
 */
const VB_DEFAULT_API_BASE_URL =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000/api/v1"
    : "https://visionbridge-backend.onrender.com/api/v1";

try {
  const configuredUrl = window.localStorage.getItem("visionbridge.apiBaseUrl");
  window.VB_API_BASE_URL = configuredUrl || VB_DEFAULT_API_BASE_URL;
} catch {
  // Storage can be unavailable in restrictive browser privacy modes.
  window.VB_API_BASE_URL = VB_DEFAULT_API_BASE_URL;
}
