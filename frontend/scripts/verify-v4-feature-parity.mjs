import { build } from "esbuild";
import { readFileSync, unlinkSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const frontendDir = resolve(scriptDir, "..");
const fixturePath = resolve(frontendDir, "../backend/tests/fixtures/v4_feature_parity.json");
const bundlePath = resolve(frontendDir, ".v4-feature-parity.bundle.mjs");
const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));

try {
  await build({
    entryPoints: [resolve(frontendDir, "scripts/v4-feature-entry.mjs")],
    bundle: true,
    platform: "node",
    format: "esm",
    target: "node20",
    external: ["@mediapipe/tasks-vision"],
    outfile: bundlePath,
    logLevel: "silent",
  });
  const module = await import(pathToFileURL(bundlePath).href);
  let maxAbsError = 0;
  let comparisons = 0;
  for (const frame of fixture.frames) {
    const actual = module.normalizeHandPairV2(frame.raw, frame.left_present, frame.right_present);
    if (actual.length !== fixture.output_dim) {
      throw new Error(frame.id + ": expected " + fixture.output_dim + " outputs, got " + actual.length);
    }
    for (let i = 0; i < fixture.output_dim; i += 1) {
      const error = Math.abs(actual[i] - frame.expected[i]);
      maxAbsError = Math.max(maxAbsError, error);
      comparisons += 1;
      if (!Number.isFinite(actual[i]) || error > fixture.atol) {
        throw new Error(frame.id + "[" + i + "] expected " + frame.expected[i] + ", got " + actual[i] + ", abs_error=" + error);
      }
    }
    if (!frame.left_present && actual.slice(0, 64).some(value => value !== 0)) {
      throw new Error(frame.id + ": absent left slot is not zero-filled");
    }
    if (!frame.right_present && actual.slice(64, 128).some(value => value !== 0)) {
      throw new Error(frame.id + ": absent right slot is not zero-filled");
    }
    if ((!frame.left_present || !frame.right_present) && actual.slice(128).some(value => value !== 0)) {
      throw new Error(frame.id + ": missing-hand geometry is not zero-filled");
    }
  }
  console.log(JSON.stringify({
    fixture_frames: fixture.frames.length,
    values_compared: comparisons,
    max_abs_error: maxAbsError,
    tolerance: fixture.atol,
    passed: true,
  }, null, 2));
} finally {
  try { unlinkSync(bundlePath); } catch {}
}
