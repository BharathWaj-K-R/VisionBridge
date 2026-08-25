/* VisionBridge — real translate wiring (webcam + MediaPipe Holistic + backend API)
   Uses real browser keypoints and the real backend model. Failure states are
   surfaced to the user; there is no fake/demo translation fallback. */

(function () {
  "use strict";

  const captionEl = document.querySelector("[data-sb-caption]");
  if (!captionEl) return;

  const confBar = document.querySelector("[data-sb-confidence-bar]");
  const confVal = document.querySelectorAll("[data-sb-confidence-val]");
  const latencyEl = document.querySelector("[data-sb-latency]");
  const fpsEl = document.querySelector("[data-sb-fps]");
  const startBtn = document.querySelector("[data-sb-start]");
  const pauseBtn = document.querySelector("[data-sb-pause]");
  const stopBtn = document.querySelector("[data-sb-stop]");
  const cameraFrame = document.querySelector(".sb-camera-frame");
  const placeholder = cameraFrame ? cameraFrame.querySelector(".placeholder") : null;

  const API_BASE = window.VB_API_BASE_URL || "http://localhost:8000/api/v1";
  const FRAME_WINDOW = 50;
  const TARGET_FPS = 15;

  const EXPECTED_POSE_DIM = 33 * 4;
  const EXPECTED_FACE_DIM = 468 * 3;

  let video, holistic, camera;
  let poseBuffer = [];
  let faceBuffer = [];
  let running = false;
  let initialized = false;
  let lastFrameTime = 0;
  let sendInFlight = false;

  function setCaption(text, confidencePct = 0, latencyMs = 0) {
    captionEl.textContent = text;
    if (confBar) confBar.style.width = `${confidencePct}%`;
    confVal.forEach((el) => (el.textContent = `${confidencePct}%`));
    if (latencyEl && Number.isFinite(latencyMs)) latencyEl.textContent = `${Math.round(latencyMs)} ms`;
  }

  function flattenPose(landmarks) {
    const out = [];
    if (Array.isArray(landmarks)) {
      landmarks.forEach((lm) => {
        out.push(Number.isFinite(lm.x) ? lm.x : 0);
        out.push(Number.isFinite(lm.y) ? lm.y : 0);
        out.push(Number.isFinite(lm.z) ? lm.z : 0);
        out.push(Number.isFinite(lm.visibility) ? lm.visibility : 0);
      });
    }
    while (out.length < EXPECTED_POSE_DIM) out.push(0);
    return out.slice(0, EXPECTED_POSE_DIM);
  }

  function flattenFace(landmarks) {
    const out = [];
    if (Array.isArray(landmarks)) {
      landmarks.forEach((lm) => {
        out.push(Number.isFinite(lm.x) ? lm.x : 0);
        out.push(Number.isFinite(lm.y) ? lm.y : 0);
        out.push(Number.isFinite(lm.z) ? lm.z : 0);
      });
    }
    while (out.length < EXPECTED_FACE_DIM) out.push(0);
    return out.slice(0, EXPECTED_FACE_DIM);
  }

  function validatePayload(posePayload, facePayload) {
    if (posePayload.length === 0 || facePayload.length === 0) return "empty pose/face buffer";
    if (posePayload.length !== facePayload.length) {
      return `pose/face frame count mismatch: ${posePayload.length} vs ${facePayload.length}`;
    }
    if (posePayload.length < FRAME_WINDOW) {
      return `not enough frames buffered: ${posePayload.length} < ${FRAME_WINDOW}`;
    }
    for (let i = 0; i < posePayload.length; i++) {
      if (posePayload[i].length !== EXPECTED_POSE_DIM) {
        return `pose frame ${i} has ${posePayload[i].length} dims, expected ${EXPECTED_POSE_DIM}`;
      }
      if (facePayload[i].length !== EXPECTED_FACE_DIM) {
        return `face frame ${i} has ${facePayload[i].length} dims, expected ${EXPECTED_FACE_DIM}`;
      }
    }
    return null;
  }

  async function extractErrorDetail(res) {
    let bodyText = "";
    try {
      bodyText = await res.text();
    } catch {
      return `HTTP ${res.status}`;
    }
    try {
      const body = JSON.parse(bodyText);
      if (typeof body.detail === "string") return body.detail;
      if (Array.isArray(body.detail)) {
        return body.detail.map((e) => (e.loc ? `${e.loc.join(".")}: ${e.msg}` : e.msg || JSON.stringify(e))).join("; ");
      }
    } catch {
      // fall through to raw text
    }
    return bodyText || `HTTP ${res.status}`;
  }

  async function sendToBackend() {
    if (sendInFlight || poseBuffer.length < FRAME_WINDOW) return;

    const posePayload = poseBuffer.slice(-FRAME_WINDOW);
    const facePayload = faceBuffer.slice(-FRAME_WINDOW);
    poseBuffer = [];
    faceBuffer = [];

    const validationError = validatePayload(posePayload, facePayload);
    if (validationError) {
      console.error(`VisionBridge: refusing to send invalid payload — ${validationError}`);
      setCaption(`Validation error: ${validationError}`, 0, 0);
      return;
    }

    sendInFlight = true;
    const start = performance.now();
    try {
      const res = await fetch(`${API_BASE}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: null,
          adapter_id: null,
          pose_keypoints: posePayload,
          face_keypoints: facePayload,
        }),
      });
      if (!res.ok) {
        const detail = await extractErrorDetail(res);
        const label = res.status >= 500 ? `Server error (${res.status})` : `Request rejected (${res.status})`;
        throw new Error(`${label}: ${detail}`);
      }
      const result = await res.json();
      const latency = Number.isFinite(result.latency_ms) ? result.latency_ms : performance.now() - start;
      const confidence = Math.max(0, Math.min(1, Number(result.confidence) || 0));
      setCaption(result.predicted_text || "(no sign detected)", Math.round(confidence * 100), latency);
    } catch (err) {
      console.warn("VisionBridge: backend call failed.", err);
      const isNetworkFailure = err instanceof TypeError;
      setCaption(
        isNetworkFailure
          ? "Live translation unavailable — backend connection failed."
          : `Live translation error: ${err.message}`,
        0,
        0
      );
    } finally {
      sendInFlight = false;
    }
  }

  function onHolisticResults(results) {
    poseBuffer.push(flattenPose(results.poseLandmarks));
    faceBuffer.push(flattenFace(results.faceLandmarks));

    const now = performance.now();
    if (fpsEl && lastFrameTime) {
      const dt = now - lastFrameTime;
      if (dt > 0) fpsEl.textContent = `${Math.round(1000 / dt)} fps`;
    }
    lastFrameTime = now;

    if (poseBuffer.length >= FRAME_WINDOW) void sendToBackend();
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) return resolve();
      const script = document.createElement("script");
      script.src = src;
      script.crossOrigin = "anonymous";
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(script);
    });
  }

  async function loadHolistic() {
    await Promise.all([
      loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"),
      loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"),
    ]);

    holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });
    holistic.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      refineFaceLandmarks: false,
    });
    holistic.onResults(onHolisticResults);
  }

  function mountVideoIntoPlaceholder() {
    if (!placeholder) return false;
    video = document.createElement("video");
    video.setAttribute("playsinline", "");
    video.autoplay = true;
    video.muted = true;
    video.style.width = "100%";
    video.style.height = "100%";
    video.style.objectFit = "cover";
    video.style.borderRadius = "inherit";
    placeholder.innerHTML = "";
    placeholder.appendChild(video);
    return true;
  }

  async function startRealTranslation() {
    if (!mountVideoIntoPlaceholder()) throw new Error("No camera placeholder found on page");
    await loadHolistic();

    camera = new Camera(video, {
      onFrame: async () => {
        if (!running) return;
        await holistic.send({ image: video });
      },
      width: 640,
      height: 480,
      fps: TARGET_FPS,
    });
    await camera.start();
    running = true;
    setCaption("Camera live — sign to translate.", 0, 0);
  }

  function stopRealTranslation() {
    running = false;
    poseBuffer = [];
    faceBuffer = [];
    if (camera) camera.stop();
    if (video && video.srcObject) {
      video.srcObject.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    }
    setCaption("Translation stopped. Press Start to resume.", 0, 0);
  }

  async function handleStart() {
    if (initialized) {
      running = true;
      return;
    }
    initialized = true;
    try {
      await startRealTranslation();
    } catch (err) {
      initialized = false;
      console.error("VisionBridge: real pipeline failed to start.", err);
      setCaption(`Live translation unavailable: ${err.message}`, 0, 0);
    }
  }

  startBtn && startBtn.addEventListener("click", handleStart);
  pauseBtn && pauseBtn.addEventListener("click", () => { running = false; });
  stopBtn && stopBtn.addEventListener("click", stopRealTranslation);

  setCaption("Press Start to begin live translation.", 0, 0);
})();
