import { BrowserLetterModel, type BrowserAdapterPayload, type BrowserModelPayload } from "./browserModel";
import { normalizeHandPair } from "./landmarks";

export type ApiError = Error & { status?: number };

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");
const LOCAL_MODE = import.meta.env.VITE_LOCAL_MODE !== "false";
const LOCAL_USER_KEY = "visionbridge_user";
const LOCAL_LETTER_ADAPTERS_KEY = "visionbridge_letter_adapters";
const LOCAL_HISTORY_KEY = "visionbridge_letter_history";

export function getToken(): string | null { return localStorage.getItem("visionbridge_token"); }
export function setToken(token: string): void { localStorage.setItem("visionbridge_token", token); }
export function clearToken(): void { localStorage.removeItem("visionbridge_token"); localStorage.removeItem(LOCAL_USER_KEY); }

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type") && init.body) headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) headers.set("Authorization", "Bearer " + token);
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 20_000);
  try {
    const response = await fetch(API_BASE + path, { ...init, headers, signal: init.signal || controller.signal });
    if (!response.ok) {
      let detail = "Request failed (" + response.status + ")";
      try {
        const payload = await response.json();
        if (typeof payload?.detail === "string") detail = payload.detail;
      } catch {}
      const error = new Error(detail) as ApiError;
      error.status = response.status;
      throw error;
    }
    return (await response.json()) as T;
  } finally {
    window.clearTimeout(timer);
  }
}

export type User = { id: number; username: string; created_at: string };
export type Token = { access_token: string; token_type: string };
export type LetterSample = { letter: string; hand_keypoints: number[] };
export type LetterCalibrationResult = { adapter_id: number; letters: string[]; shots: Record<string, number>; param_count: number };
export type LetterPredictionResult = { predicted_letter: string; confidence: number; latency_ms: number; adapter_id: number };
export type LetterRecognitionEvent = { user_id: number; adapter_id: number; predicted_letter: string; confidence: number; latency_ms: number; };

function localUser(username?: string): User {
  const raw = localStorage.getItem(LOCAL_USER_KEY);
  if (raw) {
    try { return JSON.parse(raw) as User; } catch {}
  }
  const user: User = { id: 1, username: username || "Local User", created_at: new Date().toISOString() };
  localStorage.setItem(LOCAL_USER_KEY, JSON.stringify(user));
  return user;
}

function localLetterAdapters(): any[] {
  try { return JSON.parse(localStorage.getItem(LOCAL_LETTER_ADAPTERS_KEY) || "[]") as any[]; } catch { return []; }
}
function saveLocalLetterAdapters(items: any[]): void { localStorage.setItem(LOCAL_LETTER_ADAPTERS_KEY, JSON.stringify(items)); }
function localHistory(): any[] {
  try { return JSON.parse(localStorage.getItem(LOCAL_HISTORY_KEY) || "[]") as any[]; } catch { return []; }
}
function saveLocalHistory(items: any[]): void { localStorage.setItem(LOCAL_HISTORY_KEY, JSON.stringify(items.slice(-100))); }

function localFit(samples: LetterSample[]) {
  const grouped: Record<string, number[][]> = {};
  for (const sample of samples) {
    const letter = sample.letter.toUpperCase();
    grouped[letter] = grouped[letter] || [];
    grouped[letter].push(normalizeHandPair(sample.hand_keypoints));
  }
  const entries = Object.entries(grouped);
  if (entries.length < 2) throw new Error("Calibrate at least two different letters.");
  const prototypes: Record<string, number[]> = {};
  const shots: Record<string, number> = {};
  for (const entry of entries) {
    const letter = entry[0];
    const values = entry[1];
    const mean = values[0].map((_, index) => values.reduce((sum, value) => sum + value[index], 0) / values.length);
    const norm = Math.hypot(...mean);
    prototypes[letter] = norm > 1e-8 ? mean.map((value) => value / norm) : mean;
    shots[letter] = values.length;
  }
  return { prototypes, shots, letters: entries.map((entry) => entry[0]) };
}

function localPredict(adapter: any, raw: number[]) {
  const query = normalizeHandPair(raw);
  const queryNorm = Math.hypot(...query) || 1;
  const scores = Object.entries(adapter.prototypes).map(([letter, prototype]: [string, any]) => {
    const score = query.reduce((sum, value, index) => sum + (value / queryNorm) * Number(prototype[index]), 0);
    return { letter, score };
  }).sort((a, b) => b.score - a.score);
  const logits = scores.map((item) => Math.exp((item.score - scores[0].score) * 10));
  const total = logits.reduce((sum, value) => sum + value, 0) || 1;
  const confidence = logits[0] / total;
  return { predicted_letter: scores[0].score < 0.35 ? "?" : scores[0].letter, confidence };
}

export const api = {
  register: async (username: string, password: string): Promise<User> =>
    LOCAL_MODE ? localUser(username) : request<User>("/auth/register", { method: "POST", body: JSON.stringify({ username, password }) }),
  login: async (username: string, password: string): Promise<Token> => {
    if (LOCAL_MODE) { localUser(username); return { access_token: "visionbridge-local-token", token_type: "bearer" }; }
    return request<Token>("/auth/login", { method: "POST", body: JSON.stringify({ username, password }) });
  },
  me: async (): Promise<User> => LOCAL_MODE ? localUser() : request<User>("/users/me"),
  dashboard: async () => {
    if (LOCAL_MODE) {
      const history = localHistory();
      const averageConfidence = history.length ? history.reduce((sum, item) => sum + item.confidence, 0) / history.length : null;
      const averageLatency = history.length ? history.reduce((sum, item) => sum + item.latency_ms, 0) / history.length : null;
      return {
        model: { status: localLetterAdapters().length ? "ready" : "calibrate first", modality: "hand-only few-shot letters" },
        usage: { translation_events: history.length, average_confidence: averageConfidence, average_latency_ms: averageLatency },
        recent_activity: history.slice(-5).reverse(),
      };
    }
    return request<any>("/dashboard");
  },
  history: async (_params = "") => LOCAL_MODE ? { items: localHistory().reverse() } : request<any>("/history" + (_params ? "?" + _params : "")),
  exportHistoryCsv: async (): Promise<Blob> => {
    if (LOCAL_MODE) {
      const rows = localHistory();
      const header = "id,created_at,predicted_text,confidence,latency_ms,used_adapter\n";
      const body = rows.map((row) => [row.id, row.created_at, row.predicted_text, row.confidence, row.latency_ms, row.used_adapter].join(",")).join("\n");
      return new Blob([header, body], { type: "text/csv" });
    }
    const headers = new Headers();
    const token = getToken();
    if (token) headers.set("Authorization", "Bearer " + token);
    const response = await fetch(API_BASE + "/history/export.csv", { headers });
    if (!response.ok) throw new Error("Request failed (" + response.status + ")");
    return response.blob();
  },
  letterAdapters: async () => LOCAL_MODE ? localLetterAdapters() : request<any[]>("/users/me/adapters"),
  deleteAdapter: async (id: number) => {
    if (LOCAL_MODE) {
      saveLocalLetterAdapters(localLetterAdapters().filter((item) => item.id !== id));
      return { ok: true };
    }
    return request<any>("/users/me/adapters/" + id, { method: "DELETE" });
  },
  letterCalibrate: async (userId: number, samples: LetterSample[], calibrationSeconds: number): Promise<LetterCalibrationResult> => {
    if (LOCAL_MODE) {
      const fitted = localFit(samples);
      const items = localLetterAdapters();
      const adapter = {
        id: items.reduce((max, item) => Math.max(max, Number(item.id) || 0), 0) + 1,
        letters: fitted.letters,
        shots: fitted.shots,
        prototypes: fitted.prototypes,
        param_count: fitted.letters.length * 126,
        calibration_seconds: calibrationSeconds,
        created_at: new Date().toISOString(),
      };
      saveLocalLetterAdapters([...items, adapter]);
      return { adapter_id: adapter.id, letters: adapter.letters, shots: adapter.shots, param_count: adapter.param_count };
    }
    return request<LetterCalibrationResult>("/letter/calibrate", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, samples, calibration_seconds: calibrationSeconds }),
    });
  },

  letterBrowserModel: async (): Promise<BrowserLetterModel> => {
    if (LOCAL_MODE) throw new Error("Browser model is not required in local demo mode.");
    const payload = await request<BrowserModelPayload>("/letter/model");
    return new BrowserLetterModel(payload);
  },
  letterAdapterPayload: async (adapterId: number): Promise<BrowserAdapterPayload> => {
    if (LOCAL_MODE) {
      const adapter = localLetterAdapters().find((item) => item.id === adapterId);
      if (!adapter) throw new Error("Choose a calibrated signer adapter first.");
      return {
        version: 0,
        method: "local-raw-prototype",
        base_model_version: "local",
        base_model_sha256: "local",
        feature_dim: 126,
        embedding_dim: 126,
        prototypes: adapter.prototypes,
        shots: adapter.shots || {},
      };
    }
    return request<BrowserAdapterPayload>("/letter/adapters/" + adapterId);
  },
  logLetterEvent: async (event: LetterRecognitionEvent): Promise<void> => {
    if (LOCAL_MODE) {
      const history = localHistory();
      saveLocalHistory([...history, {
        id: Date.now(),
        predicted_text: event.predicted_letter,
        confidence: event.confidence,
        latency_ms: event.latency_ms,
        used_adapter: 1,
        created_at: new Date().toISOString(),
      }]);
      return;
    }
    await request<{ logged: boolean }>("/letter/event", {
      method: "POST",
      body: JSON.stringify(event),
    });
  },

  letterPredict: async (userId: number, adapterId: number, handKeypoints: number[]): Promise<LetterPredictionResult> => {
    const started = performance.now();
    if (LOCAL_MODE) {
      const adapter = localLetterAdapters().find((item) => item.id === adapterId);
      if (!adapter) throw new Error("Choose a calibrated signer adapter first.");
      const result = localPredict(adapter, handKeypoints);
      const latency = performance.now() - started;
      const history = localHistory();
      const row = {
        id: Date.now(),
        predicted_text: result.predicted_letter,
        confidence: result.confidence,
        latency_ms: latency,
        used_adapter: 1,
        created_at: new Date().toISOString(),
      };
      saveLocalHistory([...history, row]);
      return { ...result, latency_ms: latency, adapter_id: adapterId };
    }
    return request<LetterPredictionResult>("/letter/predict", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, adapter_id: adapterId, hand_keypoints: handKeypoints }),
    });
  },
};
