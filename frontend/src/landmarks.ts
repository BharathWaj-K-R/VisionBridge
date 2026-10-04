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

export type TrackerMode = "adaptive" | "fixed" | "off";
export type TrackerSettings = {
  mode: TrackerMode;
  fixedColor: string;
};

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


export type TrackerOverlayMeta = {
  prediction?: string;
  confidence?: number;
  leftColor?: string;
  rightColor?: string;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function hexToRgba(hex: string, alpha: number): string {
  const clean = hex.replace("#", "");
  const normalized = clean.length === 3
    ? clean.split("").map((char) => char + char).join("")
    : clean.padEnd(6, "0").slice(0, 6);
  const value = Number.parseInt(normalized, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
}

function relativeLuminance(r: number, g: number, b: number): number {
  const linear = (channel: number) => {
    const normalized = channel / 255;
    return normalized <= 0.03928
      ? normalized / 12.92
      : Math.pow((normalized + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
}

function contrastRatio(backgroundLuminance: number, candidate: string): number {
  const clean = candidate.replace("#", "");
  const value = Number.parseInt(clean, 16);
  const foreground = relativeLuminance((value >> 16) & 255, (value >> 8) & 255, value & 255);
  const lighter = Math.max(backgroundLuminance, foreground);
  const darker = Math.min(backgroundLuminance, foreground);
  return (lighter + 0.05) / (darker + 0.05);
}

const TRACKER_COLORS = [
  "#00E5FF",
  "#39FF14",
  "#FFE600",
  "#FF3BD4",
  "#FFFFFF",
  "#111827",
  "#8B5CF6",
];

let adaptiveCanvas: HTMLCanvasElement | null = null;
let adaptiveContext: CanvasRenderingContext2D | null = null;

function ensureAdaptiveSampler(): CanvasRenderingContext2D | null {
  if (!adaptiveCanvas) {
    adaptiveCanvas = document.createElement("canvas");
    adaptiveCanvas.width = 64;
    adaptiveCanvas.height = 48;
    adaptiveContext = adaptiveCanvas.getContext("2d", { willReadFrequently: true });
  }
  return adaptiveContext;
}

function sampleHandBackgroundLuminance(
  video: HTMLVideoElement,
  landmarks: LandmarkPoint[] | undefined,
): number {
  const context = ensureAdaptiveSampler();
  if (!context || !video.videoWidth || !video.videoHeight || !landmarks?.length) return 0.18;

  context.drawImage(video, 0, 0, 64, 48);
  const pixels = context.getImageData(0, 0, 64, 48).data;

  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const landmark of landmarks) {
    minX = Math.min(minX, landmark.x);
    minY = Math.min(minY, landmark.y);
    maxX = Math.max(maxX, landmark.x);
    maxY = Math.max(maxY, landmark.y);
  }

  const left = clamp(Math.floor(minX * 64) - 4, 0, 63);
  const right = clamp(Math.ceil(maxX * 64) + 4, 0, 63);
  const top = clamp(Math.floor(minY * 48) - 4, 0, 47);
  const bottom = clamp(Math.ceil(maxY * 48) + 4, 0, 47);
  const samples: number[] = [];

  for (let x = left; x <= right; x += 3) {
    for (const y of [top, bottom]) {
      const index = (y * 64 + x) * 4;
      samples.push(relativeLuminance(pixels[index], pixels[index + 1], pixels[index + 2]));
    }
  }
  for (let y = top; y <= bottom; y += 3) {
    for (const x of [left, right]) {
      const index = (y * 64 + x) * 4;
      samples.push(relativeLuminance(pixels[index], pixels[index + 1], pixels[index + 2]));
    }
  }

  if (!samples.length) return 0.18;
  return samples.reduce((sum, value) => sum + value, 0) / samples.length;
}

export function getAdaptiveTrackerColors(
  video: HTMLVideoElement,
  left: LandmarkPoint[] | undefined,
  right: LandmarkPoint[] | undefined,
): { left?: string; right?: string } {
  const choose = (landmarks: LandmarkPoint[] | undefined): string | undefined => {
    if (!landmarks?.length) return undefined;
    const background = sampleHandBackgroundLuminance(video, landmarks);
    return TRACKER_COLORS.reduce(
      (best, candidate) =>
        contrastRatio(background, candidate) > contrastRatio(background, best) ? candidate : best,
      TRACKER_COLORS[0],
    );
  };
  return { left: choose(left), right: choose(right) };
}

function drawRoundRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): void {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function drawGlowStroke(
  context: CanvasRenderingContext2D,
  draw: () => void,
  color: string,
  width: number,
  glow: number,
  alpha = 1,
): void {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = width;
  context.globalAlpha = alpha;
  context.shadowColor = color;
  context.shadowBlur = glow;
  draw();
  context.stroke();
  context.restore();
}

function convexHull(points: Array<[number, number]>): Array<[number, number]> {
  if (points.length < 3) return points;
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Array<[number, number]> = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper: Array<[number, number]> = [];
  for (let index = sorted.length - 1; index >= 0; index -= 1) {
    const point = sorted[index];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function drawMotionArrow(
  context: CanvasRenderingContext2D,
  points: Array<[number, number]>,
  width: number,
  height: number,
  mirrorX: boolean,
  color: string,
): void {
  if (points.length < 2) return;
  const previous = points[points.length - 2];
  const current = points[points.length - 1];
  const x1 = (mirrorX ? 1 - previous[0] : previous[0]) * width;
  const y1 = previous[1] * height;
  const x2 = (mirrorX ? 1 - current[0] : current[0]) * width;
  const y2 = current[1] * height;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.hypot(dx, dy);
  if (length < 3) return;
  const ux = dx / length;
  const uy = dy / length;
  const tipX = x2 + ux * 18;
  const tipY = y2 + uy * 18;
  drawGlowStroke(context, () => {
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(tipX, tipY);
    context.moveTo(tipX, tipY);
    context.lineTo(tipX - ux * 10 - uy * 5, tipY - uy * 10 + ux * 5);
    context.moveTo(tipX, tipY);
    context.lineTo(tipX - ux * 10 + uy * 5, tipY - uy * 10 - ux * 5);
  }, color, 2.5, 10, 0.78);
}

function drawTrackerConnections(
  context: CanvasRenderingContext2D,
  point: (index: number) => [number, number],
  color: string,
): void {
  // Explicitly render every MediaPipe bone with a high-contrast under-stroke.
  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.setLineDash([]);

  const drawLayer = (stroke: string, width: number, alpha: number, glow: number): void => {
    context.strokeStyle = stroke;
    context.lineWidth = width;
    context.globalAlpha = alpha;
    context.shadowColor = stroke;
    context.shadowBlur = glow;

    for (const [a, b] of HAND_CONNECTIONS) {
      const first = point(a);
      const second = point(b);
      context.beginPath();
      context.moveTo(first[0], first[1]);
      context.lineTo(second[0], second[1]);
      context.stroke();
    }
  };

  drawLayer("#000000", 7.5, 0.78, 4);
  drawLayer(color, 3.4, 1, 9);
  context.restore();
}

function drawTrackerHand(
  context: CanvasRenderingContext2D,
  landmarks: LandmarkPoint[],
  trace: Array<[number, number]>,
  width: number,
  height: number,
  label: string,
  mirrorX: boolean,
  color: string,
  meta: TrackerOverlayMeta,
): void {
  if (landmarks.length < 21) return;
  const point = (index: number): [number, number] => [
    (mirrorX ? 1 - landmarks[index].x : landmarks[index].x) * width,
    landmarks[index].y * height,
  ];

  const cloud = landmarks.map((landmark) => [
    (mirrorX ? 1 - landmark.x : landmark.x) * width,
    landmark.y * height,
  ] as [number, number]);
  const hull = convexHull(cloud);

  context.save();

  if (hull.length >= 3) {
    context.beginPath();
    context.moveTo(hull[0][0], hull[0][1]);
    for (let index = 1; index < hull.length; index += 1) context.lineTo(hull[index][0], hull[index][1]);
    context.closePath();
    context.fillStyle = hexToRgba(color, 0.045);
    context.fill();
    context.strokeStyle = hexToRgba(color, 0.2);
    context.lineWidth = 1;
    context.stroke();
  }

  drawTrackerConnections(context, point, color);

  for (let index = 0; index < 21; index += 1) {
    const [x, y] = point(index);
    context.beginPath();
    context.arc(x, y, index === 0 ? 5.5 : 3.8, 0, Math.PI * 2);
    context.fillStyle = color;
    context.shadowColor = color;
    context.shadowBlur = 15;
    context.fill();
    context.shadowBlur = 0;
    context.beginPath();
    context.arc(x, y, index === 0 ? 2.5 : 1.6, 0, Math.PI * 2);
    context.fillStyle = "#ffffff";
    context.fill();
  }

  for (const index of [0, 5, 9, 13, 17]) {
    const [x, y] = point(index);
    context.beginPath();
    context.arc(x, y, index === 0 ? 14 : 9, 0, Math.PI * 2);
    context.strokeStyle = hexToRgba(color, 0.45);
    context.lineWidth = 1;
    context.setLineDash([3, 4]);
    context.stroke();
    context.setLineDash([]);
  }

  if (trace.length >= 2) {
    drawGlowStroke(context, () => {
      context.beginPath();
      for (let index = 1; index < trace.length; index += 1) {
        const previous = trace[index - 1];
        const current = trace[index];
        context.moveTo((mirrorX ? 1 - previous[0] : previous[0]) * width, previous[1] * height);
        context.lineTo((mirrorX ? 1 - current[0] : current[0]) * width, current[1] * height);
      }
    }, color, 2, 9, 0.38);
    drawMotionArrow(context, trace, width, height, mirrorX, color);
  }

  const wrist = point(0);
  const badgeWidth = 92;
  const badgeHeight = 25;
  const badgeX = clamp(wrist[0] + 12, 6, width - badgeWidth - 6);
  const badgeY = clamp(wrist[1] - 36, 6, height - badgeHeight - 6);
  context.fillStyle = "rgba(3,9,14,.76)";
  drawRoundRect(context, badgeX, badgeY, badgeWidth, badgeHeight, 5);
  context.fill();
  context.strokeStyle = hexToRgba(color, 0.8);
  context.lineWidth = 1;
  context.stroke();
  context.fillStyle = color;
  context.font = "700 8px Space Mono, monospace";
  context.fillText(label + " · TRACKING", badgeX + 7, badgeY + 10);
  context.fillStyle = "#ffffff";
  context.font = "700 9px Space Mono, monospace";
  const prediction = String(meta.prediction || "LIVE").trim() || "LIVE";
  context.fillText(prediction, badgeX + 7, badgeY + 20);
  context.fillStyle = "#b8cad3";
  context.font = "700 7px Space Mono, monospace";
  context.fillText(Math.round(clamp(Number(meta.confidence || 0), 0, 1) * 100) + "%", badgeX + 55, badgeY + 20);

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
  settings: TrackerSettings = { mode: "adaptive", fixedColor: "#00E5FF" },
  meta: TrackerOverlayMeta = {},
): void {
  const width = canvas.width;
  const height = canvas.height;
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, width, height);

  if (settings.mode === "off") return;

  const fallback = settings.fixedColor || "#00E5FF";
  const leftColor = settings.mode === "adaptive" ? (meta.leftColor || fallback) : fallback;
  const rightColor = settings.mode === "adaptive" ? (meta.rightColor || fallback) : fallback;

  if (left?.length) drawTrackerHand(context, left, traces.left, width, height, "LEFT", mirrorX, leftColor, meta);
  if (right?.length) drawTrackerHand(context, right, traces.right, width, height, "RIGHT", mirrorX, rightColor, meta);
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