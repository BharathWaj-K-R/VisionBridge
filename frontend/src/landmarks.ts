import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

export const HAND_DIM = 63;
export const COMBINED_HAND_DIM = 126;

export const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const LEFT = "left";
const RIGHT = "right";
const TASKS_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export type LandmarkPoint = { x: number; y: number; z?: number };
export type LandmarkFrame = {
  leftHand: number[];
  rightHand: number[];
  leftVisible: boolean;
  rightVisible: boolean;
  leftLandmarks?: LandmarkPoint[];
  rightLandmarks?: LandmarkPoint[];
  timestamp: number;
};

export type TrackerVisual = "anatomy" | "neon" | "holographic";

type HandLandmarkerResultLike = {
  landmarks?: LandmarkPoint[][];
  handedness?: Array<Array<{ categoryName?: string; category_name?: string; label?: string }>>;
  handednesses?: Array<Array<{ categoryName?: string; category_name?: string; label?: string }>>;
  multiHandLandmarks?: LandmarkPoint[][];
  multiHandedness?: Array<Array<{ categoryName?: string; category_name?: string; label?: string }>>;
};

function finite(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function flattenHandLandmarks(landmarks: LandmarkPoint[] | undefined): number[] {
  const values: number[] = [];
  for (const landmark of landmarks || []) {
    values.push(finite(landmark.x), finite(landmark.y), finite(landmark.z));
  }
  while (values.length < HAND_DIM) values.push(0);
  return values.slice(0, HAND_DIM);
}

type HandednessEntry = { categoryName?: string; category_name?: string; label?: string };
function handedLabel(entry: HandednessEntry | undefined): string {
  // Training and browser inference both consume the raw MediaPipe Tasks
  // handedness result. Do not add a mirror-specific swap in only one path.
  return String(
    entry?.categoryName ??
      entry?.category_name ??
      entry?.label ??
      "",
  ).trim().toLowerCase();
}

export function frameFromResults(results: HandLandmarkerResultLike): LandmarkFrame {
  const multi = Array.isArray(results?.landmarks)
    ? results.landmarks
    : Array.isArray(results?.multiHandLandmarks)
      ? results.multiHandLandmarks
      : [];
  const handedness = Array.isArray(results?.handedness)
    ? results.handedness
    : Array.isArray(results?.handednesses)
      ? results.handednesses
      : Array.isArray(results?.multiHandedness)
        ? results.multiHandedness
        : [];

  const leftIndex = handedness.findIndex((entry) => handedLabel(entry?.[0]) === LEFT);
  const rightIndex = handedness.findIndex((entry) => handedLabel(entry?.[0]) === RIGHT);

  const leftLandmarks = leftIndex >= 0 ? multi[leftIndex] : undefined;
  const rightLandmarks = rightIndex >= 0 ? multi[rightIndex] : undefined;

  return {
    leftHand: flattenHandLandmarks(leftLandmarks),
    rightHand: flattenHandLandmarks(rightLandmarks),
    leftVisible: Boolean(leftLandmarks?.length),
    rightVisible: Boolean(rightLandmarks?.length),
    leftLandmarks,
    rightLandmarks,
    timestamp: performance.now(),
  };
}

function normalizeSingleHand(values: number[]): number[] {
  if (values.length !== HAND_DIM) throw new Error("Expected 63 hand features");
  let active = false;
  for (const value of values) {
    if (value !== 0) {
      active = true;
      break;
    }
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
  if (values.length !== COMBINED_HAND_DIM) {
    throw new Error("Expected 126 combined hand features");
  }
  return [
    ...normalizeSingleHand(values.slice(0, HAND_DIM)),
    ...normalizeSingleHand(values.slice(HAND_DIM)),
  ];
}

function drawHand(
  context: CanvasRenderingContext2D,
  landmarks: LandmarkPoint[] | undefined,
  width: number,
  height: number,
  label: string,
  lineWidth: number,
  jointRadius: number,
  mirrorX: boolean,
  visual: TrackerVisual,
): void {
  if (!landmarks?.length) return;

  const point = (index: number): [number, number] => {
    const landmark = landmarks[index];
    return [
      (mirrorX ? 1 - landmark.x : landmark.x) * width,
      landmark.y * height,
    ];
  };

  if (visual === "anatomy") {
    context.save();
    context.strokeStyle = "#00f3ff";
    context.fillStyle = "#d9fbff";
    context.shadowColor = "#00f3ff";
    context.shadowBlur = 8;
    context.lineWidth = lineWidth + 0.5;

    for (const [a, b] of HAND_CONNECTIONS) {
      const first = point(a);
      const second = point(b);
      context.beginPath();
      context.moveTo(first[0], first[1]);
      context.lineTo(second[0], second[1]);
      context.stroke();
    }

    context.shadowBlur = 12;
    for (let index = 0; index < landmarks.length; index += 1) {
      const [x, y] = point(index);
      context.beginPath();
      context.arc(x, y, index === 0 ? jointRadius + 2 : jointRadius, 0, Math.PI * 2);
      context.fill();
      context.stroke();
    }

    for (const index of [2, 5, 9, 13, 17]) {
      const [x, y] = point(index);
      context.beginPath();
      context.arc(x, y, jointRadius + 5, 0, Math.PI * 2);
      context.globalAlpha = 0.35;
      context.stroke();
      context.globalAlpha = 1;
    }
    context.shadowBlur = 0;
    context.font = "700 10px Avenir Next, Helvetica, sans-serif";
    context.fillText(label, label === "LEFT" ? 12 : Math.max(12, width - 48), 18);
    context.restore();
    return;
  }

  if (visual === "holographic") {
    context.save();
    context.strokeStyle = "#c084fc";
    context.fillStyle = "#f0abfc";
    context.shadowColor = "#a855f7";
    context.shadowBlur = 10;
    context.lineWidth = lineWidth;

    for (const [a, b] of HAND_CONNECTIONS) {
      const first = point(a);
      const second = point(b);
      context.beginPath();
      context.moveTo(first[0], first[1]);
      context.lineTo(second[0], second[1]);
      context.stroke();
    }

    context.globalAlpha = 0.18;
    for (const [a, b] of HAND_CONNECTIONS) {
      const first = point(a);
      const second = point(b);
      const dx = second[0] - first[0];
      const dy = second[1] - first[1];
      context.beginPath();
      context.moveTo(first[0], first[1]);
      context.lineTo(first[0] + dx * 0.82 - dy * 0.12, first[1] + dy * 0.82 + dx * 0.12);
      context.lineTo(second[0], second[1]);
      context.closePath();
      context.fill();
    }

    context.globalAlpha = 0.9;
    for (let index = 0; index < landmarks.length; index += 1) {
      const [x, y] = point(index);
      context.beginPath();
      context.arc(x, y, index === 0 ? jointRadius + 2 : Math.max(1.5, jointRadius - 0.2), 0, Math.PI * 2);
      context.fill();
    }
    context.globalAlpha = 1;
    context.shadowBlur = 0;
    context.font = "700 10px Avenir Next, Helvetica, sans-serif";
    context.fillText(label, label === "LEFT" ? 12 : Math.max(12, width - 48), 18);
    context.restore();
    return;
  }

  context.save();
  context.strokeStyle = "#39ff14";
  context.fillStyle = "#b8ff9e";
  context.shadowColor = "#39ff14";
  context.shadowBlur = 10;
  context.lineWidth = lineWidth;

  for (const [a, b] of HAND_CONNECTIONS) {
    const first = point(a);
    const second = point(b);
    context.beginPath();
    context.moveTo(first[0], first[1]);
    context.lineTo(second[0], second[1]);
    context.stroke();
  }

  for (let index = 0; index < landmarks.length; index += 1) {
    const [x, y] = point(index);
    context.beginPath();
    context.arc(x, y, index === 0 ? jointRadius + 1.5 : jointRadius, 0, Math.PI * 2);
    context.fill();
  }

  context.shadowBlur = 0;
  context.font = "700 10px Avenir Next, Helvetica, sans-serif";
  context.fillText(label, label === "LEFT" ? 12 : Math.max(12, width - 48), 18);
  context.restore();
}

export function drawHands(
  canvas: HTMLCanvasElement,
  left: LandmarkPoint[] | undefined,
  right: LandmarkPoint[] | undefined,
  traces: {
    left: Array<[number, number]>;
    right: Array<[number, number]>;
  } = { left: [], right: [] },
  mirrorX = true,
  visual: TrackerVisual = "neon",
): void {
  const width = canvas.width;
  const height = canvas.height;
  const context = canvas.getContext("2d");
  if (!context) return;

  context.clearRect(0, 0, width, height);
  context.lineCap = "round";
  context.lineJoin = "round";

  const drawTrace = (points: Array<[number, number]>, color: string) => {
    if (points.length < 2) return;
    context.save();
    context.beginPath();
    for (let index = 1; index < points.length; index += 1) {
      const previous = points[index - 1];
      const current = points[index];
      context.moveTo((mirrorX ? 1 - previous[0] : previous[0]) * width, previous[1] * height);
      context.lineTo((mirrorX ? 1 - current[0] : current[0]) * width, current[1] * height);
    }
    context.strokeStyle = color;
    context.lineWidth = visual === "anatomy" ? 2 : visual === "holographic" ? 4 : 3;
    context.globalAlpha = visual === "holographic" ? 0.18 : 0.3;
    context.shadowColor = color;
    context.shadowBlur = visual === "neon" ? 12 : 5;
    context.stroke();
    context.restore();
  };

  if (visual === "anatomy") {
    drawTrace(traces.left, "#00f3ff");
    drawTrace(traces.right, "#7dd3fc");
  } else if (visual === "holographic") {
    drawTrace(traces.left, "#ec4899");
    drawTrace(traces.right, "#a855f7");
  } else {
    drawTrace(traces.left, "#39ff14");
    drawTrace(traces.right, "#9cff80");
  }

  drawHand(context, left, width, height, "LEFT", 2.5, 3.0, mirrorX, visual);
  drawHand(context, right, width, height, "RIGHT", 2.5, 3.0, mirrorX, visual);
}

export async function createHands(
  onResults: (results: HandLandmarkerResultLike) => void,
): Promise<{
  send: (payload: { image: HTMLVideoElement }) => Promise<void>;
  close: () => void;
}> {
  const vision = await FilesetResolver.forVisionTasks(TASKS_WASM_URL);
  const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  let lastTimestamp = -1;

  return {
    async send({ image }: { image: HTMLVideoElement }) {
      const timestamp = Math.max(Math.round(performance.now()), lastTimestamp + 1);
      lastTimestamp = timestamp;
      const result = handLandmarker.detectForVideo(image, timestamp);
      onResults(result as unknown as HandLandmarkerResultLike);
    },
    close() {
      handLandmarker.close();
    },
  };
}