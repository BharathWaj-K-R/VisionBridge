import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const dist = resolve("dist");
const indexPath = resolve(dist, "index.html");
const markerPath = resolve(dist, "deploy-marker.txt");

if (!existsSync(indexPath)) {
  throw new Error("Production build verification failed: dist/index.html is missing");
}

if (!existsSync(markerPath)) {
  throw new Error("Production build verification failed: deploy-marker.txt is missing from dist");
}

const marker = readFileSync(markerPath, "utf8").trim();
if (marker !== "VISIONBRIDGE_DIST_ARTIFACT") {
  throw new Error("Production build verification failed: unexpected dist artifact marker");
}

const html = readFileSync(indexPath, "utf8");

if (/\/src\/main\.tsx\b/.test(html) || /src=["']\/src\//.test(html)) {
  throw new Error(
    "Production build verification failed: dist/index.html still references the TypeScript source tree",
  );
}

if (!/assets\/.+\.(?:js|css)/.test(html)) {
  throw new Error(
    "Production build verification failed: dist/index.html does not reference compiled Vite assets",
  );
}

console.log("Production build verified: dist/index.html references compiled assets only.");
