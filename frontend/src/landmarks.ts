export const POSE_DIM = 132;
export const FACE_DIM = 1404;
export const HAND_DIM = 63;
export const FRAME_WINDOW = 50;

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export type LandmarkFrame = {
  pose: number[];
  face: number[];
  leftHand: number[];
  rightHand: number[];
  leftVisible: boolean;
  rightVisible: boolean;
};

export function loadMediaPipeHolistic(): Promise<any> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-vb-mediapipe="holistic"]') as HTMLScriptElement | null;
    if (existing) {
      if ((window as any).Holistic) return resolve((window as any).Holistic);
      existing.addEventListener("load", () => resolve((window as any).Holistic));
      existing.addEventListener("error", () => reject(new Error("MediaPipe Holistic failed to load")));
      return;
    }
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js";
    script.crossOrigin = "anonymous";
    script.dataset.vbMediapipe = "holistic";
    script.onload = () => {
      const Holistic = (window as any).Holistic;
      if (!Holistic) reject(new Error("MediaPipe Holistic global is unavailable"));
      else resolve(Holistic);
    };
    script.onerror = () => reject(new Error("Failed to load MediaPipe Holistic"));
    document.head.appendChild(script);
  });
}

function finite(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function flattenLandmarks(landmarks: any[] | undefined, dimensions: number, visibility: boolean): number[] {
  const values: number[] = [];
  for (const landmark of landmarks || []) {
    values.push(finite(landmark.x), finite(landmark.y), finite(landmark.z));
    if (visibility) values.push(finite(landmark.visibility));
  }
  while (values.length < dimensions) values.push(0);
  return values.slice(0, dimensions);
}

export function frameFromResults(results: any): LandmarkFrame {
  return {
    pose: flattenLandmarks(results.poseLandmarks, POSE_DIM, true),
    face: flattenLandmarks(results.faceLandmarks, FACE_DIM, false),
    leftHand: flattenLandmarks(results.leftHandLandmarks, HAND_DIM, false),
    rightHand: flattenLandmarks(results.rightHandLandmarks, HAND_DIM, false),
    leftVisible: Boolean(results.leftHandLandmarks?.length),
    rightVisible: Boolean(results.rightHandLandmarks?.length),
  };
}

export function drawHands(
  canvas: HTMLCanvasElement,
  left: any[] | undefined,
  right: any[] | undefined,
): void {
  const width = canvas.width;
  const height = canvas.height;
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, width, height);
  context.lineWidth = 2;
  context.strokeStyle = "#ffffff";
  context.fillStyle = "#ffffff";

  const draw = (landmarks: any[] | undefined, label: string, xLabel: number) => {
    if (!landmarks?.length) return;
    context.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      const first = landmarks[a];
      const second = landmarks[b];
      if (!first || !second) continue;
      context.moveTo(first.x * width, first.y * height);
      context.lineTo(second.x * width, second.y * height);
    }
    context.stroke();
    for (const point of landmarks) {
      context.beginPath();
      context.arc(point.x * width, point.y * height, 2.5, 0, Math.PI * 2);
      context.fill();
    }
    context.font = "600 12px Inter, sans-serif";
    context.fillText(label, xLabel, 18);
  };

  draw(left, "LEFT", 12);
  draw(right, "RIGHT", Math.max(12, width - 54));
}

export async function createHolistic(
  onResults: (results: any) => void,
): Promise<any> {
  const Holistic = await loadMediaPipeHolistic();
  const holistic = new Holistic({
    locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
  });
  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  holistic.onResults(onResults);
  return holistic;
}
