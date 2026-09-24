import { useCallback, useEffect, useRef, useState } from "react";
import {
  createHands,
  drawHands,
  frameFromResults,
  type LandmarkFrame,
} from "./landmarks";

const TRACE_POINTS = 28;

export function useLandmarkSession(sampleFps: number) {
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
      setStatus("Loading hand tracker…");
      hands = await createHands((results) => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video) return;

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

        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;
        if (canvas) {
          if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
          }
          drawHands(canvas, frame.leftLandmarks, frame.rightLandmarks, traceRef.current);
        }

        const now = performance.now();
        const interval = 1000 / sampleFps;
        if (now - lastSampleRef.current < interval) return;
        lastSampleRef.current = now;

        framesRef.current.push(frame);
        if (framesRef.current.length > 60) framesRef.current.shift();
      });

      handsRef.current = hands;

      if (startGeneration !== startGenerationRef.current || !videoRef.current) {
        throw new Error("Camera start cancelled");
      }

      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640, max: 960 },
          height: { ideal: 480, max: 720 },
          facingMode: "user",
          frameRate: { ideal: 30, max: 60 },
        },
        audio: false,
      });

      if (
        startGeneration !== startGenerationRef.current
        || !videoRef.current
      ) {
        throw new Error("Camera start cancelled");
      }

      const video = videoRef.current;
      if (!video) throw new Error("Camera preview is unavailable");

      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();

      if (startGeneration !== startGenerationRef.current || videoRef.current !== video) {
        throw new Error("Camera start cancelled");
      }

      activeRef.current = true;
      setRunning(true);
      setStatus("Live · hand tracking");

      let frames = 0;
      let tick = performance.now();
      let processing = false;

      const loop = async () => {
        if (!activeRef.current || !videoRef.current || !handsRef.current) return;

        if (!processing) {
          processing = true;
          try {
            await handsRef.current.send({ image: videoRef.current });
            frames += 1;
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
      setStatus(error instanceof Error ? error.message : "Camera start failed");
      throw error;
    } finally {
      startingRef.current = false;
    }
  }, [sampleFps, stop]);

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
