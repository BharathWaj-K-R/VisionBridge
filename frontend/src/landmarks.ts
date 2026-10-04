function drawTrackerConnections(
  context: CanvasRenderingContext2D,
  point: (index: number) => [number, number],
  color: string,
): void {
  // Draw every MediaPipe bone as an explicit segment. A dark under-stroke keeps
  // the skeleton readable against both light skin and dark camera backgrounds.
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