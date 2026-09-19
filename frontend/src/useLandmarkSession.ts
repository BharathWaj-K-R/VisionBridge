import { useCallback, useEffect, useRef, useState } from "react";
import { createHolistic, drawHands, frameFromResults, type LandmarkFrame } from "./landmarks";

export function useLandmarkSession(sampleFps: number) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const holisticRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const activeRef = useRef(false);
  const lastSampleRef = useRef(0);
  const framesRef = useRef<LandmarkFrame[]>([]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [fps, setFps] = useState(0);

  const stop = useCallback(() => {
    activeRef.current = false;
    setRunning(false);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    holisticRef.current?.close?.();
    holisticRef.current = null;
    framesRef.current = [];
    lastSampleRef.current = 0;
    setStatus("Stopped");
  }, []);

  const start = useCallback(async () => {
    if (activeRef.current) return;
    try {
      setStatus("Loading landmark engine…");
      const holistic = await createHolistic((results) => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video) return;
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;
        if (canvas) {
          canvas.width = width;
          canvas.height = height;
          drawHands(canvas, results.leftHandLandmarks, results.rightHandLandmarks);
        }

        const now = performance.now();
        const interval = 1000 / sampleFps;
        if (now - lastSampleRef.current < interval) return;
        lastSampleRef.current = now;
        framesRef.current.push(frameFromResults(results));
      });

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      if (!videoRef.current) throw new Error("Camera preview is unavailable");
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      holisticRef.current = holistic;
      activeRef.current = true;
      setRunning(true);
      setStatus("Live");

      let frames = 0;
      let tick = performance.now();
      const loop = async () => {
        if (!activeRef.current || !videoRef.current || !holisticRef.current) return;
        await holisticRef.current.send({ image: videoRef.current });
        frames += 1;
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
      stop();
      setStatus(error instanceof Error ? error.message : "Camera start failed");
      throw error;
    }
  }, [sampleFps, stop]);

  const snapshot = useCallback(() => [...framesRef.current], []);
  const clear = useCallback(() => { framesRef.current = []; }, []);

  useEffect(() => stop, [stop]);

  return { videoRef, canvasRef, running, status, fps, start, stop, snapshot, clear };
}
