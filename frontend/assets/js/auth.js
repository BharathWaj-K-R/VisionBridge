(function () {
  "use strict";

  const form = document.querySelector("[data-auth-form]");
  if (!form) return;

  const modeButtons = document.querySelectorAll("[data-auth-mode]");
  const submit = document.querySelector("[data-auth-submit]");
  const status = document.querySelector("[data-auth-status]");
  const password = document.querySelector("#authPassword");
  let mode = "login";

  function setMode(next) {
    mode = next;
    modeButtons.forEach((button) => {
      const active = button.dataset.authMode === mode;
      button.classList.toggle("btn-primary-sb", active);
      button.classList.toggle("btn-ghost-sb", !active);
    });
    submit.textContent = mode === "login" ? "Sign in" : "Create account";
    password.autocomplete = mode === "login" ? "current-password" : "new-password";
    if (status) status.textContent = "";
  }

  async function request(path, payload) {
    const response = await fetch(`${window.VB_API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const text = await response.text();
    let body = {};
    try { body = text ? JSON.parse(text) : {}; } catch { body = { detail: text }; }
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body));
    return body;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (status) status.textContent = "Working…";
    submit.disabled = true;
    const payload = {
      username: document.querySelector("#authUsername").value.trim(),
      password: password.value,
    };
    try {
      if (mode === "register") {
        await request("/auth/register", payload);
      }
      const token = mode === "login" ? (await request("/auth/login", payload)).access_token : (await request("/auth/login", payload)).access_token;
      localStorage.setItem("visionbridge.accessToken", token);
      if (status) status.textContent = "Authenticated. Redirecting…";
      window.location.href = "/pages/dashboard.html";
    } catch (error) {
      if (status) status.textContent = error.message || "Authentication failed.";
    } finally {
      submit.disabled = false;
    }
  });

  modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.authMode)));
  setMode("login");
})();
