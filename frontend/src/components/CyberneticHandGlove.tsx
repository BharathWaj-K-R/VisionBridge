import { useEffect, useRef, useState } from "react";
// @ts-ignore -- the current frontend intentionally uses Three.js without its optional declaration graph.
import * as THREE from "three";
import type { LandmarkFrame, LandmarkPoint } from "../landmarks";
import "./CyberneticHandGlove.css";

type Props = {
  readFrame: () => LandmarkFrame | null;
  running: boolean;
  prediction: string;
  confidence: number;
};

const CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

type HandRig = {
  root: any;
  joints: any[];
  rings: any[];
  bones: any[];
};

const WRIST_AXIS = new THREE.Vector3(0, 1, 0);
const NODE_BASES = new Set([0, 5, 9, 13, 17]);
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);

function createHandRig(side: "left" | "right", color: string): HandRig {
  const root = new THREE.Group();
  root.name = "live-" + side + "-glove";
  root.visible = false;

  const boneMaterial = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.72,
    metalness: 0.62,
    roughness: 0.28,
    transparent: true,
    opacity: 0.92,
  });
  const palmMaterial = new THREE.MeshStandardMaterial({
    color: "#9b8afb",
    emissive: "#7359ef",
    emissiveIntensity: 0.55,
    metalness: 0.65,
    roughness: 0.3,
    transparent: true,
    opacity: 0.9,
  });
  const jointMaterial = new THREE.MeshStandardMaterial({
    color: "#b8fbff",
    emissive: color,
    emissiveIntensity: 1.05,
    metalness: 0.3,
    roughness: 0.22,
  });
  const baseMaterial = new THREE.MeshStandardMaterial({
    color: "#c4b5fd",
    emissive: "#8b5cf6",
    emissiveIntensity: 0.95,
    metalness: 0.3,
    roughness: 0.2,
  });
  const tipMaterial = new THREE.MeshStandardMaterial({
    color: "#e6feff",
    emissive: "#65f5ff",
    emissiveIntensity: 1.4,
    metalness: 0.2,
    roughness: 0.18,
  });

  const joints: any[] = [];
  const rings: any[] = [];
  const bones: any[] = [];

  for (let index = 0; index < 21; index += 1) {
    const radius = index === 0 ? 0.052 : FINGERTIPS.has(index) ? 0.042 : 0.031;
    const material = FINGERTIPS.has(index) ? tipMaterial : NODE_BASES.has(index) ? baseMaterial : jointMaterial;
    const joint = new THREE.Mesh(new THREE.SphereGeometry(radius, 12, 10), material);
    joint.name = side + "-landmark-" + index;
    root.add(joint);
    joints.push(joint);

    if (NODE_BASES.has(index) || FINGERTIPS.has(index)) {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius * 1.6, 0.006, 6, 18),
        FINGERTIPS.has(index) ? tipMaterial : baseMaterial,
      );
      ring.name = side + "-joint-ring-" + index;
      root.add(ring);
      rings.push({ mesh: ring, index });
    }
  }

  for (let index = 0; index < CONNECTIONS.length; index += 1) {
    const [start, end] = CONNECTIONS[index];
    const isPalm = start === 0 || (start >= 5 && end >= 9 && end <= 17);
    const material = isPalm ? palmMaterial : boneMaterial;
    const bone = new THREE.Mesh(
      new THREE.CylinderGeometry(isPalm ? 0.025 : 0.018, isPalm ? 0.032 : 0.023, 1, 8, 1, false),
      material,
    );
    bone.name = side + "-segment-" + start + "-" + end;
    root.add(bone);
    bones.push(bone);
  }

  return { root, joints, rings, bones };
}

function toWorldPoint(point: LandmarkPoint, horizontalRange: number) {
  const depth = Number.isFinite(point.z) ? Number(point.z) : 0;
  return new THREE.Vector3(
    (0.5 - point.x) * horizontalRange,
    (0.5 - point.y) * 2.55,
    THREE.MathUtils.clamp(-depth * 2.5, -0.65, 0.65),
  );
}

function updateHandRig(rig: HandRig, landmarks: LandmarkPoint[] | undefined, horizontalRange: number) {
  if (!landmarks || landmarks.length < 21) {
    rig.root.visible = false;
    return;
  }

  const points = landmarks.slice(0, 21);
  if (points.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y))) {
    rig.root.visible = false;
    return;
  }

  rig.root.visible = true;
  const worldPoints = points.map((point) => toWorldPoint(point, horizontalRange));

  rig.joints.forEach((joint, index) => joint.position.copy(worldPoints[index]));
  rig.rings.forEach(({ mesh, index }) => mesh.position.copy(worldPoints[index]));

  rig.bones.forEach((bone, index) => {
    const [startIndex, endIndex] = CONNECTIONS[index];
    const start = worldPoints[startIndex];
    const end = worldPoints[endIndex];
    const direction = new THREE.Vector3().subVectors(end, start);
    const length = direction.length();
    bone.position.copy(start).add(end).multiplyScalar(0.5);
    bone.quaternion.setFromUnitVectors(WRIST_AXIS, direction.normalize());
    bone.scale.set(1, Math.max(length, 0.001), 1);
  });
}

function SvgHand({ landmarks, color }: { landmarks?: LandmarkPoint[]; color: string }) {
  if (!landmarks || landmarks.length < 21) return null;
  const safePoints = landmarks.slice(0, 21);
  if (safePoints.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y))) return null;

  return (
    <g>
      {CONNECTIONS.map(([start, end], index) => (
        <line
          key={"bone-" + index}
          x1={(1 - safePoints[start].x) * 160}
          y1={safePoints[start].y * 100}
          x2={(1 - safePoints[end].x) * 160}
          y2={safePoints[end].y * 100}
          stroke={index >= 4 && index <= 7 ? "#b9a7ff" : color}
          strokeWidth="1.25"
          strokeLinecap="round"
          opacity="0.9"
        />
      ))}
      {safePoints.map((point, index) => (
        <g key={"node-" + index}>
          {[0, 5, 9, 13, 17, 4, 8, 12, 16, 20].includes(index) && (
            <circle
              cx={(1 - point.x) * 160}
              cy={point.y * 100}
              r={index === 0 ? 2.2 : 1.7}
              fill="none"
              stroke={index >= 4 && index % 4 === 0 ? "#b9a7ff" : color}
              strokeWidth="0.65"
              opacity="0.8"
            />
          )}
          <circle
            cx={(1 - point.x) * 160}
            cy={point.y * 100}
            r={[4, 8, 12, 16, 20].includes(index) ? 1.25 : 0.9}
            fill={[4, 8, 12, 16, 20].includes(index) ? "#e9fdff" : color}
          />
        </g>
      ))}
    </g>
  );
}

function handCount(frame: LandmarkFrame | null): number {
  if (!frame) return 0;
  const left = Boolean(frame.leftVisible && frame.leftLandmarks?.length >= 21);
  const right = Boolean(frame.rightVisible && frame.rightLandmarks?.length >= 21);
  return Number(left) + Number(right);
}

export default function CyberneticHandGlove({ readFrame, running, prediction, confidence }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<LandmarkFrame | null>(null);
  const [renderMode, setRenderMode] = useState<"starting" | "3d" | "2d">("starting");
  const [visibleHands, setVisibleHands] = useState(0);
  const [fallbackFrame, setFallbackFrame] = useState<LandmarkFrame | null>(null);

  useEffect(() => {
    const timer = window.setInterval(() => {
      const frame = running ? readFrame() : null;
      frameRef.current = frame;

      const nextCount = handCount(frame);
      setVisibleHands((previous) => previous === nextCount ? previous : nextCount);
      if (renderMode === "2d") setFallbackFrame(frame);
    }, 40);

    return () => window.clearInterval(timer);
  }, [readFrame, running, renderMode]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let renderer: any;
    let animationFrame = 0;
    let observer: ResizeObserver | null = null;
    let disposed = false;

    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
      renderer.setClearColor(0x000000, 0);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      mount.appendChild(renderer.domElement);
    } catch {
      setRenderMode("2d");
      return undefined;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 50);
    camera.position.set(0, 0, 5.2);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.AmbientLight("#a7c7ff", 1.15));
    const cyanLight = new THREE.PointLight("#19d9f5", 22, 9);
    cyanLight.position.set(-2.2, 2.8, 3.5);
    scene.add(cyanLight);
    const violetLight = new THREE.PointLight("#9b73ff", 18, 8);
    violetLight.position.set(2.4, -1.6, 2.8);
    scene.add(violetLight);

    const leftRig = createHandRig("left", "#22d3ee");
    const rightRig = createHandRig("right", "#8b7cff");
    scene.add(leftRig.root, rightRig.root);

    const resize = () => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();
    observer = new ResizeObserver(resize);
    observer.observe(mount);

    const tick = () => {
      if (disposed) return;
      animationFrame = requestAnimationFrame(tick);

      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      const horizontalRange = Math.max(1.65, Math.min(3.1, (width / height) * 2.05));
      const frame = frameRef.current;
      updateHandRig(leftRig, frame?.leftVisible ? frame.leftLandmarks : undefined, horizontalRange);
      updateHandRig(rightRig, frame?.rightVisible ? frame.rightLandmarks : undefined, horizontalRange);
      renderer.render(scene, camera);
    };

    setRenderMode("3d");
    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      observer?.disconnect();
      scene.traverse((object: any) => {
        if (object.geometry?.dispose) object.geometry.dispose();
        const material = object.material;
        if (Array.isArray(material)) material.forEach((entry: any) => entry.dispose?.());
        else material?.dispose?.();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  const statusText = !running
    ? "Camera inactive"
    : visibleHands === 2
      ? "Both hands linked"
      : visibleHands === 1
        ? "Hand linked"
        : "Searching for hands";
  const countText = visibleHands === 1 ? "1 hand" : visibleHands + " hands";

  return (
    <section className="panel cybernetic-glove-panel" aria-labelledby="cybernetic-glove-title">
      <div className="cybernetic-glove-header">
        <div>
          <div className="eyebrow">Live digital twin</div>
          <h2 id="cybernetic-glove-title">Cybernetic glove</h2>
          <p>Linked to the live hand landmarks, not a prerecorded animation.</p>
        </div>
        <span className={"cybernetic-link-state" + (running && visibleHands > 0 ? " connected" : "")} role="status" aria-live="polite">
          <i aria-hidden="true" />
          {statusText}
        </span>
      </div>

      <div className="cybernetic-glove-viewport">
        <div className="cybernetic-glove-orbit orbit-one" aria-hidden="true" />
        <div className="cybernetic-glove-orbit orbit-two" aria-hidden="true" />
        <div ref={mountRef} className="cybernetic-glove-canvas" aria-hidden="true" />
        {renderMode === "2d" && (
          <svg className="cybernetic-glove-fallback" viewBox="0 0 160 100" preserveAspectRatio="xMidYMid meet" aria-label="Live hand landmark wireframe fallback" role="img">
            <SvgHand landmarks={fallbackFrame?.leftLandmarks} color="#32e5f4" />
            <SvgHand landmarks={fallbackFrame?.rightLandmarks} color="#a78bfa" />
          </svg>
        )}
        {visibleHands === 0 && (
          <div className="cybernetic-glove-empty">
            <span className="cybernetic-empty-mark" aria-hidden="true"><i /><i /><i /></span>
            <strong>{running ? "Looking for your hands" : "Ready when you are"}</strong>
            <span>{running ? "Move a hand fully into the camera frame." : "Start the camera to bring the glove online."}</span>
          </div>
        )}
        <span className="cybernetic-glove-corner top-left">LANDMARK LINK / {renderMode === "2d" ? "2D FALLBACK" : "3D LIVE"}</span>
        <span className="cybernetic-glove-corner bottom-left">21 NODES / HAND</span>
        <span className="cybernetic-glove-corner bottom-right">{countText.toUpperCase()}</span>
      </div>

      <div className="cybernetic-glove-telemetry">
        <div>
          <span>Tracked hands</span>
          <strong>{visibleHands} / 2</strong>
        </div>
        <div>
          <span>Visible landmarks</span>
          <strong>{visibleHands * 21} / 42</strong>
        </div>
        <div>
          <span>Display mode</span>
          <strong>{renderMode === "2d" ? "2D fallback" : renderMode === "3d" ? "3D live" : "Starting"}</strong>
        </div>
        {prediction !== "—" && confidence > 0 && (
          <div className="cybernetic-glove-prediction">
            <span>Latest model result</span>
            <strong>{prediction} · {Math.round(confidence * 100)}%</strong>
          </div>
        )}
      </div>
      <p className="cybernetic-glove-footnote">The glove disappears when a hand is not detected. Its pose is refreshed from MediaPipe landmarks.</p>
    </section>
  );
}
