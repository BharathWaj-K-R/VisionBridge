import { useEffect, useRef } from "react";
// Three.js ships as the runtime dependency; this file intentionally treats the renderer API as runtime-only.
// @ts-ignore -- the project does not carry the large optional Three.js declaration graph.
import * as THREE from "three";
import type { AvatarPreferences } from "./AvatarCustomizer";
import { poseFor, type HandFinger } from "./avatarRig";

type Props = {
  letter: string;
  preferences: AvatarPreferences;
  playing: boolean;
  view: "full" | "close";
  expression: "neutral" | "question" | "emphasis";
};

const DEG = Math.PI / 180;

function makeMaterial(color: string, opacity: number, wireframe = false) {
  return new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.8,
    transparent: true,
    opacity,
    roughness: 0.32,
    metalness: 0.15,
    wireframe,
    depthWrite: opacity > 0.7,
  });
}

function addJoint(parent: any, position: any, color: string, size = 0.075) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(size, 12, 12),
    makeMaterial(color, 0.95),
  );
  mesh.position.copy(position);
  parent.add(mesh);
  return mesh;
}

function addSegment(parent: any, length: number, radius: number, color: string) {
  const mesh = new THREE.Mesh(
    new THREE.CapsuleGeometry(radius, Math.max(0.05, length - radius * 2), 6, 12),
    makeMaterial(color, 0.92),
  );
  mesh.rotation.z = Math.PI;
  mesh.position.y = -length / 2;
  parent.add(mesh);
  return mesh;
}

function buildHand(side: "left" | "right", preferences: AvatarPreferences) {
  const root = new THREE.Group();
  root.name = side + "-hand";

  const skin = makeMaterial(preferences.skinTone, 0.88);
  const outline = makeMaterial(preferences.highContrast ? "#38d9ff" : "#7dd3fc", 0.7);
  const palm = new THREE.Mesh(new THREE.CapsuleGeometry(0.34, 0.42, 6, 12), skin);
  palm.scale.set(1, 0.72, 0.72);
  palm.rotation.z = Math.PI / 2;
  root.add(palm);

  const fingers: Array<[keyof HandFinger, number, number, number]> = [
    ["index", -0.22, 0.34, 0],
    ["middle", -0.07, 0.39, 0],
    ["ring", 0.08, 0.36, 0],
    ["little", 0.22, 0.31, 0],
  ];

  for (const [name, x, length, spread] of fingers) {
    const finger = new THREE.Group();
    finger.name = side + "-" + name;
    finger.position.set(x, 0.12, 0);
    finger.rotation.z = spread;
    addSegment(finger, length, 0.055, preferences.skinTone);
    addJoint(finger, new THREE.Vector3(0, -length * 0.52, 0), preferences.highContrast ? "#38d9ff" : "#a855f7", 0.038);
    root.add(finger);
  }

  const thumb = new THREE.Group();
  thumb.name = side + "-thumb";
  thumb.position.set(side === "left" ? -0.31 : 0.31, -0.04, 0.02);
  thumb.rotation.z = side === "left" ? -0.62 : 0.62;
  addSegment(thumb, 0.3, 0.065, preferences.skinTone);
  root.add(thumb);

  const wire = new THREE.Mesh(
    new THREE.SphereGeometry(0.48, 8, 6),
    makeMaterial(preferences.highContrast ? "#38d9ff" : "#a855f7", 0.18, true),
  );
  wire.scale.set(1, 0.76, 0.7);
  root.add(wire);

  return root;
}

function buildArm(side: "left" | "right", preferences: AvatarPreferences) {
  const root = new THREE.Group();
  root.name = side + "-arm";
  const sign = side === "left" ? -1 : 1;
  root.position.set(sign * 0.9, 1.72, 0);

  const shoulder = addJoint(root, new THREE.Vector3(0, 0, 0), preferences.highContrast ? "#38d9ff" : "#39ff14", 0.11);
  shoulder.name = side + "-shoulder";

  const upper = new THREE.Group();
  upper.name = side + "-upper";
  upper.position.set(0, 0, 0);
  addSegment(upper, 0.78, 0.13, preferences.shirtColor);
  root.add(upper);

  const elbow = new THREE.Group();
  elbow.name = side + "-elbow";
  elbow.position.set(0, -0.78, 0);
  addJoint(elbow, new THREE.Vector3(0, 0, 0), preferences.highContrast ? "#38d9ff" : "#39ff14", 0.085);
  upper.add(elbow);

  const forearm = new THREE.Group();
  forearm.name = side + "-forearm";
  addSegment(forearm, 0.78, 0.105, preferences.skinTone);
  elbow.add(forearm);

  const wrist = new THREE.Group();
  wrist.name = side + "-wrist";
  wrist.position.set(0, -0.78, 0);
  addJoint(wrist, new THREE.Vector3(0, 0, 0), preferences.highContrast ? "#38d9ff" : "#ec4899", 0.07);
  forearm.add(wrist);

  const hand = buildHand(side, preferences);
  hand.position.set(0, -0.06, 0.03);
  wrist.add(hand);

  return root;
}

function buildAvatar(preferences: AvatarPreferences) {
  const root = new THREE.Group();
  root.name = "visionbridge-3d-avatar";

  const glowColor = preferences.highContrast ? "#38d9ff" : "#a855f7";
  const accent = preferences.highContrast ? "#38d9ff" : "#39ff14";

  const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.78, 1.28, 8, 16),
    makeMaterial(preferences.shirtColor, 0.84),
  );
  torso.scale.z = 0.62;
  torso.position.y = 0.62;
  root.add(torso);

  const torsoWire = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.82, 1.35, 6, 12),
    makeMaterial(glowColor, 0.13, true),
  );
  torsoWire.scale.z = 0.66;
  torsoWire.position.copy(torso.position);
  root.add(torsoWire);

  const neck = new THREE.Mesh(
    new THREE.CylinderGeometry(0.2, 0.24, 0.28, 12),
    makeMaterial(preferences.skinTone, 0.9),
  );
  neck.position.y = 1.47;
  root.add(neck);

  const head = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.7, 2),
    makeMaterial(preferences.skinTone, 0.9),
  );
  head.name = "head";
  head.position.y = 2.18;
  head.scale.set(0.92, 1.05, 0.86);
  root.add(head);

  const headWire = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.73, 2),
    makeMaterial(glowColor, 0.16, true),
  );
  headWire.position.copy(head.position);
  headWire.scale.copy(head.scale);
  root.add(headWire);

  const hair = new THREE.Mesh(
    new THREE.SphereGeometry(0.69, 16, 12),
    makeMaterial(preferences.hairColor, 0.88),
  );
  hair.name = "hair";
  hair.position.set(0, 2.55, -0.02);
  hair.scale.set(0.94, 0.52, 0.88);
  root.add(hair);

  const eyeMaterial = makeMaterial("#e9fbff", 0.95);
  for (const x of [-0.24, 0.24]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.055, 8, 8), eyeMaterial);
    eye.position.set(x, 2.22, 0.66);
    root.add(eye);
  }

  const chestNode = addJoint(root, new THREE.Vector3(0, 1.12, 0.66), accent, 0.055);
  chestNode.name = "chest-node";

  root.add(buildArm("left", preferences));
  root.add(buildArm("right", preferences));

  const halo = new THREE.Mesh(
    new THREE.IcosahedronGeometry(2.2, 2),
    makeMaterial(glowColor, 0.055, true),
  );
  halo.name = "holographic-envelope";
  halo.position.y = 1.2;
  root.add(halo);

  const nodeGroup = new THREE.Group();
  nodeGroup.name = "data-nodes";
  for (let i = 0; i < 26; i += 1) {
    const a = (i / 26) * Math.PI * 2;
    const node = new THREE.Mesh(
      new THREE.SphereGeometry(0.018 + (i % 3) * 0.008, 8, 8),
      makeMaterial(i % 2 ? glowColor : accent, 0.8),
    );
    node.position.set(Math.cos(a) * 1.75, 1.1 + Math.sin(a * 1.7) * 1.25, Math.sin(a) * 0.55);
    nodeGroup.add(node);
  }
  root.add(nodeGroup);

  return root;
}

function applyFingerPose(root: any, side: "left" | "right", hand: HandFinger) {
  const names: Array<keyof HandFinger> = ["index", "middle", "ring", "little", "thumb"];
  for (const name of names) {
    const object = root.getObjectByName(side + "-" + name);
    if (!object) continue;
    const value = hand[name] ?? 0;
    object.rotation.x = value * DEG * 0.72;
    object.rotation.z += (name === "thumb" ? (side === "left" ? -1 : 1) : 0) * 0.04;
  }
}

export default function SignAvatar3D({ letter, preferences, playing, view, expression }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef({ letter, preferences, playing, view, expression });

  useEffect(() => {
    stateRef.current = { letter, preferences, playing, view, expression };
  }, [letter, preferences, playing, view, expression]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#071018");

    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
    camera.position.set(0, 1.65, 7.2);
    camera.lookAt(0, 1.25, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight("#9bc7ff", 1.8);
    scene.add(ambient);
    const key = new THREE.PointLight("#a855f7", 28, 8);
    key.position.set(-2.5, 4.5, 3.5);
    scene.add(key);
    const rim = new THREE.PointLight("#39ff14", 20, 7);
    rim.position.set(2.8, 2.2, -2);
    scene.add(rim);

    const grid = new THREE.GridHelper(8, 28, "#31536a", "#183041");
    grid.position.y = -0.62;
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.32;
    scene.add(grid);

    const avatar = buildAvatar(stateRef.current.preferences);
    scene.add(avatar);

    const targetPose = poseFor(letter);
    const current = {
      leftArm: targetPose.leftArm,
      rightArm: targetPose.rightArm,
      leftElbow: targetPose.leftElbow,
      rightElbow: targetPose.rightElbow,
      leftWrist: targetPose.leftWrist,
      rightWrist: targetPose.rightWrist,
    };

    const clock = new THREE.Clock();

    const resize = () => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const state = stateRef.current;
      const pose = poseFor(state.letter);
      const lerp = state.playing ? 0.12 : 0.075;

      current.leftArm = THREE.MathUtils.lerp(current.leftArm, pose.leftArm, lerp);
      current.rightArm = THREE.MathUtils.lerp(current.rightArm, pose.rightArm, lerp);
      current.leftElbow = THREE.MathUtils.lerp(current.leftElbow, pose.leftElbow, lerp);
      current.rightElbow = THREE.MathUtils.lerp(current.rightElbow, pose.rightElbow, lerp);
      current.leftWrist = THREE.MathUtils.lerp(current.leftWrist, pose.leftWrist, lerp);
      current.rightWrist = THREE.MathUtils.lerp(current.rightWrist, pose.rightWrist, lerp);

      const left = avatar.getObjectByName("left-arm");
      const right = avatar.getObjectByName("right-arm");
      if (left && right) {
        left.rotation.z = current.leftArm * DEG;
        right.rotation.z = -current.rightArm * DEG;
        const le = avatar.getObjectByName("left-elbow");
        const re = avatar.getObjectByName("right-elbow");
        if (le) le.rotation.z = current.leftElbow * DEG;
        if (re) re.rotation.z = -current.rightElbow * DEG;
        const lw = avatar.getObjectByName("left-wrist");
        const rw = avatar.getObjectByName("right-wrist");
        if (lw) lw.rotation.z = current.leftWrist * DEG;
        if (rw) rw.rotation.z = -current.rightWrist * DEG;
      }

      applyFingerPose(avatar, "left", pose.leftHand);
      applyFingerPose(avatar, "right", pose.rightHand);

      const t = clock.getElapsedTime();
      const breathe = state.playing ? Math.sin(t * 3.2) * 0.025 : Math.sin(t * 1.4) * 0.012;
      avatar.position.y = breathe;
      avatar.rotation.y = Math.sin(t * 0.42) * 0.045;

      const targetZ = state.view === "close" ? 5.4 : 7.2;
      camera.position.z = THREE.MathUtils.lerp(camera.position.z, targetZ, 0.06);
      camera.position.y = THREE.MathUtils.lerp(camera.position.y, state.view === "close" ? 1.7 : 1.65, 0.06);

      const head = avatar.getObjectByName("head");
      if (head) {
        head.rotation.z = state.expression === "question" ? -0.08 : state.expression === "emphasis" ? 0.055 : 0;
      }

      const halo = avatar.getObjectByName("holographic-envelope");
      if (halo) {
        halo.rotation.x = t * 0.08;
        halo.rotation.y = -t * 0.12;
        halo.scale.setScalar(1 + Math.sin(t * 1.8) * 0.025);
      }

      const nodes = avatar.getObjectByName("data-nodes");
      if (nodes) {
        nodes.rotation.y = t * 0.18;
        nodes.rotation.x = Math.sin(t * 0.35) * 0.08;
      }

      renderer.render(scene, camera);
    };
    tick();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.dispose();
      scene.traverse((object: any) => {
        const mesh = object as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const material = mesh.material;
        if (Array.isArray(material)) material.forEach((item) => item.dispose());
        else if (material) material.dispose();
      });
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div className="sign-avatar-3d">
      <div className="avatar-3d-hud">
        <div>
          <span className="eyebrow">3D signing avatar</span>
          <strong>{letter === " " ? "SPACE" : letter || "READY"}</strong>
        </div>
        <span>{playing ? "Live motion" : "Ready · 3D"}</span>
      </div>
      <div ref={mountRef} className="avatar-3d-canvas" />
      <div className="avatar-3d-overlay avatar-3d-overlay-left">126D landmarks · sign rig</div>
      <div className="avatar-3d-overlay avatar-3d-overlay-right">{preferences.highContrast ? "High contrast" : "Holographic mesh"}</div>
      <div className="avatar-3d-scanline" />
    </div>
  );
}
