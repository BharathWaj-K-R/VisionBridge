import type { SignPlaybackItem, SignTokenKind } from "./signTypes";

const BASE_DURATIONS: Record<SignTokenKind, number> = {
  letter: 760,
  word: 1050,
  phrase: 1500,
  unsupported: 900,
};

export function durationForToken(kind: SignTokenKind, speed: number): number {
  const safeSpeed = Number.isFinite(speed) && speed > 0 ? speed : 1;
  return Math.max(220, BASE_DURATIONS[kind] / safeSpeed);
}

export function withPlaybackDurations(items: SignPlaybackItem[], speed: number): SignPlaybackItem[] {
  return items.map((item) => ({
    ...item,
    durationBaseMs: durationForToken(item.kind, speed),
  }));
}
