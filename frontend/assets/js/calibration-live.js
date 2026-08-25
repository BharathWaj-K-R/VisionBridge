(function () {
  "use strict";
  const main = document.querySelector("main");
  if (!main || !window.VB_API_FETCH) return;
  const state = { running: false, startedAt: 0, pose: [], face: [], camera: null, holistic: null, lastSample: 0, userId: null, duration: 300 };
  const POSE_DIM = 132, FACE_DIM = 1404, SAMPLE_MS = 333;
  const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[c]));

  function flattenPose(landmarks) {
    const out = [];
    (landmarks || []).forEach((lm) => out.push(lm.x || 0, lm.y || 0, lm.z || 0, lm.visibility || 0));
    while (out.length < POSE_DIM) out.push(0);
    return out.slice(0, POSE_DIM);
  }
  function flattenFace(landmarks) {
    const out = [];
    (landmarks || []).forEach((lm) => out.push(lm.x || 0, lm.y || 0, lm.z || 0));
    while (out.length < FACE_DIM) out.push(0);
    return out.slice(0, FACE_DIM);
  }
  function render(status = "Ready to begin") {
    const elapsed = state.startedAt ? Math.min(state.duration, (performance.now() - state.startedAt) / 1000) : 0;
    const progress = Math.round((elapsed / state.duration) * 100);
    const left = Math.max(0, Math.ceil(state.duration - elapsed));
    const progressNode = document.querySelector("[data-cal-progress]");
    if (progressNode) progressNode.style.width = `${progress}%`;
    const timeNode = document.querySelector("[data-cal-time]");
    if (timeNode) timeNode.textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
    const statusNode = document.querySelector("[data-cal-status]");
    if (statusNode) statusNode.textContent = status;
    const samplesNode = document.querySelector("[data-cal-samples]");
    if (samplesNode) samplesNode.textContent = `${state.pose.length} samples`;
  }

  async function loadHolistic() {
    const load = (src) => new Promise((resolve, reject) => { const s = document.createElement("script"); s.src = src; s.onload = resolve; s.onerror = reject; document.head.appendChild(s); });
    await Promise.all([load("https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"), load("https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js")]);
    state.holistic = new Holistic({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}` });
    state.holistic.setOptions({ modelComplexity: 0, smoothLandmarks: true, refineFaceLandmarks: false });
    state.holistic.onResults((results) => {
      const now = performance.now();
      if (!state.running || now - state.lastSample < SAMPLE_MS || state.pose.length >= 1024) return;
      state.lastSample = now;
      state.pose.push(flattenPose(results.poseLandmarks));
      state.face.push(flattenFace(results.faceLandmarks));
      render("Collecting landmark samples…");
    });
  }

  async function start() {
    const targetText = document.querySelector("#calTarget")?.value.trim();
    if (!targetText) { render("Enter the phrase being signed before starting."); return; }
    const meResponse = await window.VB_API_FETCH("/users/me");
    if (!meResponse.ok) throw new Error("Authentication required");
    state.userId = (await meResponse.json()).id;
    state.pose = []; state.face = []; state.startedAt = performance.now(); state.lastSample = 0; state.running = true;
    await loadHolistic();
    const video = document.querySelector("#calVideo");
    state.camera = new Camera(video, { width: 640, height: 480, onFrame: async () => { if (state.running) await state.holistic.send({ image: video }); } });
    await state.camera.start();
    render("Calibration running. Keep signing the entered phrase naturally.");
    const timer = setInterval(async () => {
      render();
      if (!state.running || performance.now() - state.startedAt >= state.duration * 1000) {
        clearInterval(timer); await finish(targetText);
      }
    }, 500);
  }

  async function finish(targetText) {
    if (!state.running && !state.startedAt) return;
    state.running = false; if (state.camera) state.camera.stop();
    if (state.pose.length < 10 || state.pose.length !== state.face.length) { render("Not enough valid synchronized samples were captured."); return; }
    render("Uploading the captured keypoints and fitting your adapter…");
    const response = await window.VB_API_FETCH("/calibration", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ user_id: state.userId, calibration_seconds: state.duration, pose_keypoints: state.pose, face_keypoints: state.face, target_text: targetText }) });
    const text = await response.text();
    let body; try { body = JSON.parse(text); } catch { body = { detail: text }; }
    if (!response.ok) throw new Error(body.detail || "Calibration failed");
    render(`Calibration complete. Adapter #${body.adapter_id} was persisted.`);
    document.querySelector("[data-cal-start]")?.setAttribute("disabled", "true");
  }

  document.querySelector("[data-cal-start]")?.addEventListener("click", () => start().catch((error) => render(error.message || String(error))));
  render();
})();
