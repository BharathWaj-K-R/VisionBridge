export const HAND_DIM = 63;
export const COMBINED_HAND_DIM = 126;

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export type LandmarkFrame = {
  leftHand: number[];
  rightHand: number[];
  leftVisible: boolean;
  rightVisible: boolean;
};

function finite(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function flattenHandLandmarks(landmarks: any[] | undefined): number[] {
  const values: number[] = [];
  for (const landmark of landmarks || []) {
    values.push(finite(landmark.x), finite(landmark.y), finite(landmark.z));
  }
  while (values.length < HAND_DIM) values.push(0);
  return values.slice(0, HAND_DIM);
}

export function frameFromResults(results: any): LandmarkFrame {
  return {
    leftHand: flattenHandLandmarks(results.leftHandLandmarks),
    rightHand: flattenHandLandmarks(results.rightHandLandmarks),
    leftVisible: Boolean(results.leftHandLandmarks?.length),
    rightVisible: Boolean(results.rightHandLandmarks?.length),
  };
}

function normalizeSingleHand(values: number[]): number[] {
  if (values.length !== HAND_DIM) throw new Error("Expected 63 hand features");
  let active = false;
  for (const value of values) {
    if (value !== 0) { active = true; break; }
  }
  if (!active) return new Array(HAND_DIM).fill(0);

  const wrist = [values[0], values[1], values[2]];
  const centered: number[] = [];
  let scale = 0;
  for (let index = 0; index < 21; index += 1) {
    const offset = index * 3;
    const x = values[offset] - wrist[0];
    const y = values[offset + 1] - wrist[1];
    const z = values[offset + 2] - wrist[2];
    centered.push(x, y, z);
    scale = Math.max(scale, Math.hypot(x, y, z));
  }
  if (scale < 1e-6) return new Array(HAND_DIM).fill(0);
  return centered.map((value) => value / scale);
}

export function normalizeHandPair(values: number[]): number[] {
  if (values.length !== COMBINED_HAND_DIM) throw new Error("Expected 126 combined hand features");
  return normalizeSingleHandPair(values);
}

function normalizeSingleHandPair(values: number[]): number[] {
  return [
    ...normalizeSingleHand(values.slice(0, HAND_DIM)),
    ...normalizeSingleHand(values.slice(HAND_DIM)),
  ];
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

export async function createHolistic(
  onResults: (results: any) => void,
): Promise<any> {
  const Holistic = await loadMediaPipeHolistic();
  const holistic = new Holistic({
    locateFile: (file: string) => "https://cdn.jsdelivr.net/npm/@mediapipe/holistic/" + file,
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
