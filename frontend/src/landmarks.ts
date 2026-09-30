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


export type TrackerOverlayMeta = {
  prediction?: string;
  confidence?: number;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
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
  context.restore();
}

function angleAt(
  points: LandmarkPoint[],
  a: number,
  center: number,
  b: number,
): number {
  const pa = points[a];
  const pc = points[center];
  const pb = points[b];
  if (!pa || !pc || !pb) return 0;
  const v1x = pa.x - pc.x;
  const v1y = pa.y - pc.y;
  const v2x = pb.x - pc.x;
  const v2y = pb.y - pc.y;
  const dot = v1x * v2x + v1y * v2y;
  const mag = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
  if (mag < 1e-6) return 0;
  return Math.acos(clamp(dot / mag, -1, 1)) * (180 / Math.PI);
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
  const size = 10;
  drawGlowStroke(context, () => {
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2 + ux * 18, y2 + uy * 18);
    context.stroke();
    context.beginPath();
    context.moveTo(x2 + ux * 18, y2 + uy * 18);
    context.lineTo(x2 + ux * 18 - ux * size - uy * size * 0.55, y2 + uy * 18 - uy * size + ux * size * 0.55);
    context.moveTo(x2 + ux * 18, y2 + uy * 18);
    context.lineTo(x2 + ux * 18 - ux * size + uy * size * 0.55, y2 + uy * 18 - uy * size - ux * size * 0.55);
    context.stroke();
  }, color, 2.5, 12);
}

function drawAnatomyHand(
  context: CanvasRenderingContext2D,
  landmarks: LandmarkPoint[],
  width: number,
  height: number,
  label: string,
  mirrorX: boolean,
): void {
  const point = (index: number): [number, number] => {
    const landmark = landmarks[index];
    return [
      (mirrorX ? 1 - landmark.x : landmark.x) * width,
      landmark.y * height,
    ];
  };

  context.save();
  drawGlowStroke(context, () => {
    for (const [a, b] of HAND_CONNECTIONS) {
      const first = point(a);
      const second = point(b);
      context.beginPath();
      context.moveTo(first[0], first[1]);
      context.lineTo(second[0], second[1]);
      context.stroke();
    }
  }, "#00f3ff", 9, 14, 0.16);

  drawGlowStroke(context, () => {
    for (const [a, b] of HAND_CONNECTIONS) {
      const first = point(a);
      const second = point(b);
      context.beginPath();
      context.moveTo(first[0], first[1]);
      context.lineTo(second[0], second[1]);
      context.stroke();
    }
  }, "#d9fbff", 4.2, 8);

  context.shadowColor = "#00f3ff";
  context.shadowBlur = 12;
  for (let index = 0; index < landmarks.length; index += 1) {
    const [x, y] = point(index);
    context.beginPath();
    context.arc(x, y, index === 0 ? 5 : 3.7, 0, Math.PI * 2);
    context.fillStyle = "#e9fdff";
    context.fill();
    context.strokeStyle = "#00f3ff";
    context.lineWidth = 1.4;
    context.stroke();
  }

  const ringIndices = [0, 5, 9, 13, 17];
  for (const index of ringIndices) {
    const [x, y] = point(index);
    for (const radius of [8, 13]) {
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.strokeStyle = "#00f3ff";
      context.lineWidth = 1;
      context.globalAlpha = radius === 8 ? 0.62 : 0.28;
      context.stroke();
    }
    context.beginPath();
    context.moveTo(x - 16, y);
    context.lineTo(x + 16, y);
    context.moveTo(x, y - 16);
    context.lineTo(x, y + 16);
    context.strokeStyle = "#7dd3fc";
    context.globalAlpha = 0.2;
    context.stroke();
  }

  const callouts: Array<[number, number, number, number]> = [
    [2, 1, 3, 17],
    [6, 5, 7, 9],
    [10, 9, 11, 13],
    [14, 13, 15, 17],
  ];
  context.font = "700 9px Space Mono, monospace";
  for (const [a, center, b, offsetSign] of callouts) {
    const angle = angleAt(landmarks, a, center, b);
    const [x, y] = point(center);
    const direction = x < width / 2 ? -1 : 1;
    const offsetY = (center % 2 === 0 ? -1 : 1) * 22;
    const boxX = clamp(x + direction * 28, 8, width - 84);
    const boxY = clamp(y + offsetY, 28, height - 18);
    context.globalAlpha = 1;
    context.strokeStyle = "#5ee7f5";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(boxX + (direction < 0 ? 84 : 0), boxY);
    context.stroke();

    context.fillStyle = "rgba(0,12,18,.78)";
    drawRoundRect(context, boxX, boxY - 11, 84, 20, 3);
    context.fill();
    context.strokeStyle = "rgba(0,243,255,.55)";
    context.stroke();
    context.fillStyle = "#bffcff";
    context.fillText("θ " + Math.round(angle) + "°", boxX + 7, boxY + 3);
  }

  context.font = "700 10px Space Mono, monospace";
  context.fillStyle = "#c8fbff";
  context.fillText(label + " · ANATOMY", label === "LEFT" ? 14 : Math.max(14, width - 104), 18);
  context.restore();
}

function drawNeonHand(
  context: CanvasRenderingContext2D,
  landmarks: LandmarkPoint[],
  width: number,
  height: number,
  label: string,
  mirrorX: boolean,
  trace: Array<[number, number]>,
): void {
  const point = (index: number): [number, number] => {
    const landmark = landmarks[index];
    return [
      (mirrorX ? 1 - landmark.x : landmark.x) * width,
      landmark.y * height,
    ];
  };

  context.save();
  drawGlowStroke(context, () => {
    for (const [a, b] of HAND_CONNECTIONS) {
      const first = point(a);
      const second = point(b);
      context.beginPath();
      context.moveTo(first[0], first[1]);
      context.lineTo(second[0], second[1]);
      context.stroke();
    }
  }, "#39ff14", 2.6, 12);

  for (let index = 0; index < landmarks.length; index += 1) {
    const [x, y] = point(index);
    context.beginPath();
    context.arc(x, y, index === 0 ? 5.2 : 4, 0, Math.PI * 2);
    context.fillStyle = "#8dff72";
    context.shadowColor = "#39ff14";
    context.shadowBlur = 18;
    context.fill();
    context.shadowBlur = 0;
    context.beginPath();
    context.arc(x, y, index === 0 ? 2.5 : 2, 0, Math.PI * 2);
    context.fillStyle = "#39ff14";
    context.fill();
  }

  drawMotionArrow(context, trace, width, height, mirrorX, "#39ff14");

  if (trace.length >= 4) {
    const last = trace[trace.length - 1];
    const prev = trace[trace.length - 4];
    const startX = (mirrorX ? 1 - last[0] : last[0]) * width;
    const startY = last[1] * height;
    const vx = (last[0] - prev[0]) * width;
    const vy = (last[1] - prev[1]) * height;
    const projectedX = startX - vx * 2.8;
    const projectedY = startY + vy * 2.8;
    const cpX = (startX + projectedX) / 2 + 34;
    const cpY = (startY + projectedY) / 2 - 34;

    drawGlowStroke(context, () => {
      context.beginPath();
      context.moveTo(startX, startY);
      context.quadraticCurveTo(cpX, cpY, projectedX, projectedY);
      context.stroke();
    }, "#8dff72", 2.2, 12, 0.85);

    context.beginPath();
    context.arc(projectedX, projectedY, 7, 0, Math.PI * 2);
    context.strokeStyle = "#8dff72";
    context.lineWidth = 1.5;
    context.setLineDash([3, 3]);
    context.stroke();
    context.setLineDash([]);
  }

  context.font = "700 10px Space Mono, monospace";
  context.fillStyle = "#d9ffd0";
  context.fillText(label + " · LIVE VECTOR", label === "LEFT" ? 14 : Math.max(14, width - 118), 18);
  context.restore();
}

function drawHolographicHand(
  context: CanvasRenderingContext2D,
  landmarks: LandmarkPoint[],
  width: number,
  height: number,
  label: string,
  mirrorX: boolean,
  meta: TrackerOverlayMeta,
): void {
  const point = (index: number): [number, number] => {
    const landmark = landmarks[index];
    return [
      (mirrorX ? 1 - landmark.x : landmark.x) * width,
      landmark.y * height,
    ];
  };

  const cloud = landmarks.map((landmark) => [
    (mirrorX ? 1 - landmark.x : landmark.x) * width,
    landmark.y * height,
  ] as [number, number]);
  const hull = convexHull(cloud);
  context.save();

  if (hull.length >= 3) {
    const centroid = hull.reduce(
      (sum, item) => [sum[0] + item[0], sum[1] + item[1]] as [number, number],
      [0, 0] as [number, number],
    );
    centroid[0] /= hull.length;
    centroid[1] /= hull.length;

    for (let index = 0; index < hull.length; index += 1) {
      const a = hull[index];
      const b = hull[(index + 1) % hull.length];
      const centerX = (a[0] + b[0] + centroid[0]) / 3;
      const centerY = (a[1] + b[1] + centroid[1]) / 3;
      context.beginPath();
      context.moveTo(a[0], a[1]);
      context.lineTo(b[0], b[1]);
      context.lineTo(centroid[0], centroid[1]);
      context.closePath();
      context.fillStyle = index % 2 ? "rgba(236,72,153,.12)" : "rgba(168,85,247,.14)";
      context.fill();
      context.strokeStyle = index % 2 ? "rgba(236,72,153,.36)" : "rgba(168,85,247,.42)";
      context.lineWidth = 1;
      context.stroke();

      context.beginPath();
      context.arc(centerX, centerY, 1.8, 0, Math.PI * 2);
      context.fillStyle = "#f0abfc";
      context.fill();
    }

    context.beginPath();
    context.moveTo(hull[0][0], hull[0][1]);
    for (let index = 1; index < hull.length; index += 1) {
      context.lineTo(hull[index][0], hull[index][1]);
    }
    context.closePath();
    context.strokeStyle = "#ec4899";
    context.lineWidth = 2;
    context.shadowColor = "#a855f7";
    context.shadowBlur = 14;
    context.stroke();
  }

  for (let index = 0; index < landmarks.length; index += 1) {
    const [x, y] = point(index);
    context.beginPath();
    context.arc(x, y, index === 0 ? 4.4 : 3.2, 0, Math.PI * 2);
    context.fillStyle = "#f4c6ff";
    context.shadowColor = "#ec4899";
    context.shadowBlur = 14;
    context.fill();
  }
  context.shadowBlur = 0;

  const wrist = point(0);
  const confidence = clamp(Number(meta.confidence || 0), 0, 1);
  const prediction = String(meta.prediction || "").trim() || "LIVE";
  const badgeWidth = 134;
  const badgeHeight = 45;
  const badgeX = clamp(wrist[0] + 16, 8, width - badgeWidth - 8);
  const badgeY = clamp(wrist[1] - 58, 8, height - badgeHeight - 8);

  context.fillStyle = "rgba(12,6,22,.84)";
  drawRoundRect(context, badgeX, badgeY, badgeWidth, badgeHeight, 6);
  context.fill();
  context.strokeStyle = "rgba(236,72,153,.7)";
  context.lineWidth = 1.2;
  context.stroke();

  context.fillStyle = "#f0abfc";
  context.font = "700 8px Space Mono, monospace";
  context.fillText(label + " · GESTURE", badgeX + 9, badgeY + 12);
  context.fillStyle = "#ffffff";
  context.font = "700 17px Space Mono, monospace";
  context.fillText(prediction === "—" ? "LIVE" : prediction, badgeX + 9, badgeY + 30);
  context.fillStyle = "#c3a7ca";
  context.font = "700 8px Space Mono, monospace";
  context.fillText(Math.round(confidence * 100) + "% CONFIDENCE", badgeX + 64, badgeY + 30);

  context.font = "700 9px Space Mono, monospace";
  context.fillStyle = "#e7c8f7";
  context.fillText(label + " · HOLOGRAPHIC MESH", label === "LEFT" ? 14 : Math.max(14, width - 146), 18);
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
  meta: TrackerOverlayMeta = {},
): void {
  const width = canvas.width;
  const height = canvas.height;
  const context = canvas.getContext("2d");
  if (!context) return;

  context.clearRect(0, 0, width, height);
  context.lineCap = "round";
  context.lineJoin = "round";

  if (visual === "anatomy") {
    drawAnatomyHand(context, left || [], width, height, "LEFT", mirrorX);
    drawAnatomyHand(context, right || [], width, height, "RIGHT", mirrorX);
  } else if (visual === "holographic") {
    drawHolographicHand(context, left || [], width, height, "LEFT", mirrorX, meta);
    drawHolographicHand(context, right || [], width, height, "RIGHT", mirrorX, meta);
  } else {
    if (left?.length) drawNeonHand(context, left, width, height, "LEFT", mirrorX, traces.left);
    if (right?.length) drawNeonHand(context, right, width, height, "RIGHT", mirrorX, traces.right);
  }
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