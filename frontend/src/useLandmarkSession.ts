import { useCallback, useEffect, useRef, useState } from "react";
import {
  createHands,
  drawHands,
  frameFromResults,
  type LandmarkFrame,
  getAdaptiveTrackerColors,
  type TrackerOverlayMeta,
  type TrackerSettings,
} from "./landmarks";

const TRACE_POINTS = 28;

export function useLandmarkSession(
  sampleFps: number,
  trackerSettings: TrackerSettings = { mode: "adaptive", fixedColor: "#00E5FF" },
  trackerMeta: TrackerOverlayMeta = {},
) {
  const targetFps = Number.isFinite(sampleFps) ? Math.min(60, Math.max(1, sampleFps)) : 30;
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handsRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const activeRef = useRef(false);
  const startingRef = useRef(false);
  const startGenerationRef = useRef(0);
  const lastSampleRef = useRef(0);
  const framesRef = useRef<LandmarkFrame[]>([]);
  const latestFrameRef = useRef<LandmarkFrame | null>(null);
  const traceRef = useRef<{
    left: Array<[number, number]>;
    right: Array<[number, number]>;
  }>({ left: [], right: [] });
  const trackerSettingsRef = useRef<TrackerSettings>(trackerSettings);
  const trackerMetaRef = useRef<TrackerOverlayMeta>(trackerMeta);
  const adaptiveColorsRef = useRef<{ left?: string; right?: string }>({});
  const lastAdaptiveColorSampleRef = useRef(0);

  useEffect(() => {
    trackerSettingsRef.current = trackerSettings;
  }, [trackerSettings]);

  useEffect(() => {
    trackerMetaRef.current = trackerMeta;
  }, [trackerMeta]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [fps, setFps] = useState(0);

  const stop = useCallback(() => {
    startGenerationRef.current += 1;
    activeRef.current = false;
    setRunning(false);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    handsRef.current?.close?.();
    handsRef.current = null;
    framesRef.current = [];
    latestFrameRef.current = null;
    lastSampleRef.current = 0;
    traceRef.current = { left: [], right: [] };
    setStatus("Stopped");
  }, []);

  const start = useCallback(async () => {
    if (activeRef.current || startingRef.current) return;
    startingRef.current = true;
    const startGeneration = startGenerationRef.current + 1;
    startGenerationRef.current = startGeneration;

    let hands: Awaited<ReturnType<typeof createHands>> | null = null;
    let stream: MediaStream | null = null;

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera API unavailable. Use HTTPS and a supported browser.");
      }

      if (!videoRef.current) {
        throw new Error("Camera preview is unavailable.");
      }

      setStatus("Requesting camera permission…");
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640, max: 960 },
            height: { ideal: 480, max: 720 },
            facingMode: "user",
            frameRate: { ideal: 30, max: 60 },
          },
          audio: false,
        });
      } catch (firstError) {
        const name = firstError instanceof DOMException ? firstError.name : "";
        if (name !== "OverconstrainedError" && name !== "NotFoundError") {
          throw firstError;
        }
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }

      if (startGeneration !== startGenerationRef.current || !videoRef.current) {
        throw new Error("Camera start cancelled");
      }

      const video = videoRef.current;
      streamRef.current = stream;
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play();

      if (startGeneration !== startGenerationRef.current || videoRef.current !== video) {
        throw new Error("Camera start cancelled");
      }

      setStatus("Loading hand tracker…");
      hands = await createHands((results) => {
        if (startGeneration !== startGenerationRef.current || !activeRef.current) {
          return;
        }

        const currentVideo = videoRef.current;
        const canvas = canvasRef.current;
        if (!currentVideo) return;

        const frame = frameFromResults(results);
        latestFrameRef.current = frame;

        if (frame.leftVisible && frame.leftLandmarks?.[0]) {
          traceRef.current.left.push([frame.leftLandmarks[0].x, frame.leftLandmarks[0].y]);
          if (traceRef.current.left.length > TRACE_POINTS) traceRef.current.left.shift();
        }
        if (frame.rightVisible && frame.rightLandmarks?.[0]) {
          traceRef.current.right.push([frame.rightLandmarks[0].x, frame.rightLandmarks[0].y]);
          if (traceRef.current.right.length > TRACE_POINTS) traceRef.current.right.shift();
        }

        const width = currentVideo.videoWidth || 640;
        const height = currentVideo.videoHeight || 480;
        if (canvas) {
          if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
          }
          const currentSettings = trackerSettingsRef.current;
          let trackerMeta = trackerMetaRef.current;
          const colorNow = performance.now();
          if (
            currentSettings.mode === "adaptive" &&
            colorNow - lastAdaptiveColorSampleRef.current >= 140
          ) {
            adaptiveColorsRef.current = getAdaptiveTrackerColors(
              currentVideo,
              frame.leftLandmarks,
              frame.rightLandmarks,
            );
            lastAdaptiveColorSampleRef.current = colorNow;
          }
          if (currentSettings.mode !== "adaptive") {
            adaptiveColorsRef.current = {};
          }
          trackerMeta = {
            ...trackerMeta,
            leftColor: adaptiveColorsRef.current.left,
            rightColor: adaptiveColorsRef.current.right,
          };
          drawHands(
            canvas,
            frame.leftLandmarks,
            frame.rightLandmarks,
            traceRef.current,
            true,
            currentSettings,
            trackerMeta,
          );
        }

        const now = performance.now();
        const interval = 1000 / targetFps;
        if (now - lastSampleRef.current < interval) return;
        lastSampleRef.current = now;

        framesRef.current.push(frame);
        if (framesRef.current.length > 60) framesRef.current.shift();
      });

      handsRef.current = hands;
      activeRef.current = true;
      setRunning(true);
      setStatus("Live · hand tracking");

      let frames = 0;
      let tick = performance.now();
      let processing = false;
      let lastInference = 0;
      const inferenceInterval = 1000 / targetFps;

      const loop = async (now: number) => {
        if (!activeRef.current || !videoRef.current || !handsRef.current) return;

        if (!processing && (lastInference === 0 || now - lastInference >= inferenceInterval)) {
          lastInference = now;
          processing = true;
          try {
            await handsRef.current.send({ image: videoRef.current });
            frames += 1;
          } catch (error) {
            if (startGeneration !== startGenerationRef.current) {
              return;
            }
            activeRef.current = false;
            setRunning(false);
            setStatus(error instanceof Error ? error.message : "Hand tracking failed");
            streamRef.current?.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
            handsRef.current?.close?.();
            handsRef.current = null;
            return;
          } finally {
            processing = false;
          }
        }

        const current = performance.now();
        if (current - tick >= 1000) {
          setFps(frames);
          frames = 0;
          tick = current;
        }

        requestAnimationFrame(loop);
      };

      requestAnimationFrame(loop);
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      hands?.close?.();

      if (startGeneration !== startGenerationRef.current) {
        return;
      }

      stop();
      if (error instanceof DOMException) {
        if (error.name === "NotAllowedError" || error.name === "SecurityError") {
          setStatus("Camera permission denied. Allow camera access for VisionBridge, then try again.");
        } else if (error.name === "NotFoundError") {
          setStatus("No camera was found. Check the camera connection and browser permissions.");
        } else if (error.name === "NotReadableError") {
          setStatus("Camera is busy or blocked by another application.");
        } else {
          setStatus(error.message || "Camera start failed");
        }
      } else {
        setStatus(error instanceof Error ? error.message : "Camera start failed");
      }
      throw error;
    } finally {
      startingRef.current = false;
    }
  }, [targetFps, stop]);

  const snapshot = useCallback(() => [...framesRef.current], []);
  const latestFrame = useCallback(() => latestFrameRef.current, []);
  const clear = useCallback(() => {
    framesRef.current = [];
    latestFrameRef.current = null;
    traceRef.current = { left: [], right: [] };
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  useEffect(() => stop, [stop]);

  return {
    videoRef,
    canvasRef,
    running,
    status,
    fps,
    start,
    stop,
    snapshot,
    latestFrame,
    clear,
  };
}
