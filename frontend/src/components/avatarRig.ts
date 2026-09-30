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
const FOLDED_THUMB: HandFinger = { thumb: 78, index: 75, middle: 80, ring: 82, little: 76 };
const CURL: HandFinger = { thumb: 30, index: 46, middle: 55, ring: 55, little: 48 };

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
  A: { ...base(-12, 12, CLOSED), leftWrist: 6, rightWrist: -6 },
  B: { ...base(-26, 26, OPEN), leftWrist: -4, rightWrist: 4 },
  C: { ...base(-40, 40, CURL), leftWrist: 12, rightWrist: -12 },
  D: { ...base(-54, 54, POINT), leftWrist: 9, rightWrist: -9 },
  E: { ...base(-64, 64, FOLDED_THUMB), leftWrist: -8, rightWrist: 8 },
  F: { ...base(-78, 78, PINCH), leftWrist: -14, rightWrist: 14 },
  G: { ...base(-36, 36, POINT), leftWrist: -18, rightWrist: 18 },
  H: { ...base(-24, 24, V), leftWrist: 8, rightWrist: -8 },
  I: { ...base(8, -8, Y), leftWrist: -7, rightWrist: 7 },
  J: { ...base(18, -18, Y), leftWrist: 18, rightWrist: -18 },
  K: { ...base(-48, 48, POINT), leftWrist: 16, rightWrist: -16 },
  L: { ...base(-60, 60, OPEN), leftWrist: -12, rightWrist: 12 },
  M: { ...base(-72, 72, CLOSED), leftWrist: 10, rightWrist: -10 },
  N: { ...base(-84, 84, CLOSED), leftWrist: -6, rightWrist: 6 },
  O: { ...base(-96, 96, PINCH), leftWrist: 14, rightWrist: -14 },
  P: { ...base(-50, 50, POINT), leftWrist: 22, rightWrist: -22 },
  Q: { ...base(-40, 40, POINT), leftWrist: -22, rightWrist: 22 },
  R: { ...base(-30, 30, V), leftWrist: 20, rightWrist: -20 },
  S: { ...base(-10, 10, CLOSED), leftWrist: -10, rightWrist: 10 },
  T: { ...base(-88, 88, FOLDED_THUMB), leftWrist: 8, rightWrist: -8 },
  U: { ...base(-26, 26, V), leftWrist: -5, rightWrist: 5 },
  V: { ...base(-46, 46, V), leftWrist: -13, rightWrist: 13 },
  W: { ...base(-62, 62, W), leftWrist: 6, rightWrist: -6 },
  X: { ...base(-72, 72, POINT), leftWrist: 25, rightWrist: -25 },
  Y: { ...base(20, -20, Y), leftWrist: -16, rightWrist: 16 },
  Z: { ...base(84, -84, POINT), leftWrist: 28, rightWrist: -28 },
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
