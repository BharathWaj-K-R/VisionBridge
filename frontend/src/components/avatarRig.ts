export type HandFinger = {
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  little: number;
};

export type AvatarPose = {
  leftArm: number;
  rightArm: number;
  leftElbow: number;
  rightElbow: number;
  leftWrist: number;
  rightWrist: number;
  leftHand: HandFinger;
  rightHand: HandFinger;
};

const OPEN: HandFinger = { thumb: 38, index: 0, middle: 0, ring: 0, little: 0 };
const RELAXED: HandFinger = { thumb: 58, index: 28, middle: 18, ring: 20, little: 24 };
const CLOSED: HandFinger = { thumb: 72, index: 78, middle: 82, ring: 80, little: 74 };
const PINCH: HandFinger = { thumb: 18, index: 58, middle: 18, ring: 18, little: 24 };
const POINT: HandFinger = { thumb: 44, index: 2, middle: 72, ring: 70, little: 68 };
const V: HandFinger = { thumb: 42, index: -8, middle: 10, ring: 70, little: 66 };
const W: HandFinger = { thumb: 38, index: -4, middle: -2, ring: -2, little: 34 };
const Y: HandFinger = { thumb: -4, index: 68, middle: 72, ring: 70, little: -8 };

const base = (leftArm: number, rightArm: number, hand = RELAXED): AvatarPose => ({
  leftArm,
  rightArm,
  leftElbow: 8,
  rightElbow: -8,
  leftWrist: 0,
  rightWrist: 0,
  leftHand: hand,
  rightHand: hand,
});

export const LETTER_POSES: Record<string, AvatarPose> = {
  A: base(-10, 10, CLOSED),
  B: base(-24, 24, OPEN),
  C: base(-38, 38, RELAXED),
  D: base(-52, 52, POINT),
  E: base(-62, 62, CLOSED),
  F: base(-76, 76, PINCH),
  G: base(-34, 34, POINT),
  H: base(-22, 22, V),
  I: base(8, -8, Y),
  J: base(16, -16, Y),
  K: base(-46, 46, POINT),
  L: base(-58, 58, OPEN),
  M: base(-70, 70, CLOSED),
  N: base(-82, 82, CLOSED),
  O: base(-94, 94, PINCH),
  P: base(-48, 48, POINT),
  Q: base(-38, 38, POINT),
  R: base(-28, 28, V),
  S: base(-8, 8, CLOSED),
  T: base(-86, 86, PINCH),
  U: base(-24, 24, V),
  V: base(-44, 44, V),
  W: base(-60, 60, W),
  X: base(-70, 70, POINT),
  Y: base(18, -18, Y),
  Z: base(82, -82, POINT),
};

export const DEFAULT_POSE: AvatarPose = {
  leftArm: -8,
  rightArm: 8,
  leftElbow: 0,
  rightElbow: 0,
  leftWrist: 0,
  rightWrist: 0,
  leftHand: RELAXED,
  rightHand: RELAXED,
};

export function poseFor(letter: string): AvatarPose {
  return LETTER_POSES[letter] || DEFAULT_POSE;
}

export function catmullRom(t: number, p0: number, p1: number, p2: number, p3: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (
    (2 * p1) +
    (-p0 + p2) * t +
    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
  );
}

export function interpolateAngle(current: number, target: number, t: number): number {
  return catmullRom(t, current, current, target, target);
}
