import { BrowserLetterModel, type BrowserAdapterPayload, type BrowserModelPayload } from "./browserModel";
import { normalizeHandPair } from "./landmarks";

export type ApiError = Error & { status?: number };

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");
const LOCAL_MODE = import.meta.env.VITE_LOCAL_MODE !== "false";
const LOCAL_USER_KEY = "visionbridge_user";
const LOCAL_AUTH_KEY = "visionbridge_local_auth";
const LOCAL_LETTER_ADAPTERS_KEY = "visionbridge_letter_adapters";
const LOCAL_HISTORY_KEY = "visionbridge_letter_history";
const LOCAL_CUSTOM_WORDS_KEY = "visionbridge_custom_words";
const LOCAL_QUICK_ACCESS_KEY = "visionbridge_quick_access";
const LOCAL_PROFILES_KEY = "visionbridge_personalization_profiles";
const LOCAL_ACTIVE_PROFILE_KEY = "visionbridge_active_profile_id";
const LOCAL_USAGE_KEY = "visionbridge_communication_usage";
const SESSION_HINT_KEY = "visionbridge_session_hint";
const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

let csrfTokenMemory: string | null = null;

export function hasSessionHint(): boolean { return localStorage.getItem(SESSION_HINT_KEY) === "1"; }
export function setSessionHint(): void { localStorage.setItem(SESSION_HINT_KEY, "1"); }
export function clearSessionHint(): void { localStorage.removeItem(SESSION_HINT_KEY); }

export function isLocalAuthenticated(): boolean {
  return LOCAL_MODE && localStorage.getItem(LOCAL_AUTH_KEY) === "1";
}

export function clearLocalAuth(): void {
  localStorage.removeItem(LOCAL_AUTH_KEY);
  clearSessionHint();
  localStorage.removeItem(LOCAL_USER_KEY);
  csrfTokenMemory = null;
}

async function ensureCsrfToken(): Promise<string> {
  if (csrfTokenMemory) return csrfTokenMemory;

  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 20_000);
  try {
    const response = await fetch(API_BASE + "/auth/csrf", {
      method: "GET",
      credentials: "include",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!response.ok) {
      const error = new Error("CSRF bootstrap failed (" + response.status + ")") as ApiError;
      error.status = response.status;
      throw error;
    }
    const payload = await response.json() as { csrf_token?: unknown };
    if (typeof payload.csrf_token !== "string" || !payload.csrf_token) {
      throw new Error("CSRF bootstrap returned an invalid token");
    }
    csrfTokenMemory = payload.csrf_token;
    return csrfTokenMemory;
  } finally {
    window.clearTimeout(timer);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type") && init.body) headers.set("Content-Type", "application/json");
  const method = (init.method || "GET").toUpperCase();
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 20_000);

  try {
    if (!SAFE_METHODS.has(method)) {
      headers.set("X-CSRF-Token", await ensureCsrfToken());
    }

    let response = await fetch(API_BASE + path, {
      ...init,
      headers,
      credentials: "include",
      signal: init.signal || controller.signal,
    });

    if (response.status === 403 && !SAFE_METHODS.has(method) && path !== "/auth/csrf") {
      csrfTokenMemory = null;
      headers.set("X-CSRF-Token", await ensureCsrfToken());
      response = await fetch(API_BASE + path, {
        ...init,
        headers,
        credentials: "include",
        signal: init.signal || controller.signal,
      });
    }

    if (!response.ok) {
      let detail = "Request failed (" + response.status + ")";
      try {
        const payload = await response.json();
        if (typeof payload?.detail === "string") detail = payload.detail;
      } catch (parseError) {
        void parseError;
      }
      const error = new Error(detail) as ApiError;
      error.status = response.status;
      throw error;
    }
    return (await response.json()) as T;
  } finally {
    window.clearTimeout(timer);
  }
}

export type User = { id: number; username: string; email?: string | null; created_at: string; is_verified: boolean };
export type Token = { access_token: string; token_type: string };
export type RegisterResponse = {
  message: string;
  email: string;
  verification_required: boolean;
  resend_after_seconds: number;
};
export type ResendOtpResponse = {
  message: string;
  resend_after_seconds: number;
};
export type LetterSample = { letter: string; hand_keypoints: number[] };
export type LetterCalibrationResult = { adapter_id: number; letters: string[]; shots: Record<string, number>; param_count: number };
export type LetterPredictionResult = { predicted_letter: string; confidence: number; latency_ms: number; adapter_id: number | null; mode: "base" | "adapter" };
export type LetterRecognitionEvent = { user_id: number; adapter_id: number | null; predicted_letter: string; confidence: number; latency_ms: number; };

function localUser(username?: string): User {
  const raw = localStorage.getItem(LOCAL_USER_KEY);
  if (raw) {
    try { return JSON.parse(raw) as User; } catch {}
  }
  const user: User = { id: 1, username: username || "Local User", email: null, created_at: new Date().toISOString(), is_verified: true };
  localStorage.setItem(LOCAL_USER_KEY, JSON.stringify(user));
  return user;
}

function localScopeKey(base: string): string {
  try {
    const raw = localStorage.getItem(LOCAL_USER_KEY);
    if (raw) {
      const user = JSON.parse(raw) as { username?: string };
      if (user.username) return base + ":" + encodeURIComponent(user.username);
    }
  } catch {}
  return base + ":anonymous";
}

function localLetterAdapters(): any[] {
  try { return JSON.parse(localStorage.getItem(localScopeKey(LOCAL_LETTER_ADAPTERS_KEY)) || "[]") as any[]; } catch { return []; }
}
function saveLocalLetterAdapters(items: any[]): void { localStorage.setItem(localScopeKey(LOCAL_LETTER_ADAPTERS_KEY), JSON.stringify(items)); }
function localHistory(): any[] {
  try { return JSON.parse(localStorage.getItem(localScopeKey(LOCAL_HISTORY_KEY)) || "[]") as any[]; } catch { return []; }
}
function saveLocalHistory(items: any[]): void { localStorage.setItem(localScopeKey(LOCAL_HISTORY_KEY), JSON.stringify(items.slice(-100))); }
function localCustomWords(): any[] {
  try { return JSON.parse(localStorage.getItem(localScopeKey(LOCAL_CUSTOM_WORDS_KEY)) || "[]") as any[]; } catch { return []; }
}
function saveLocalCustomWords(items: any[]): void { localStorage.setItem(localScopeKey(LOCAL_CUSTOM_WORDS_KEY), JSON.stringify(items)); }
function localQuickAccess(): Array<string | null> {
  try {
    const parsed = JSON.parse(localStorage.getItem(localScopeKey(LOCAL_QUICK_ACCESS_KEY)) || "[]");
    return Array.from({ length: 10 }, (_, index) => parsed[index] || null);
  } catch {
    return Array(10).fill(null);
  }
}
function saveLocalQuickAccess(items: Array<string | null>): void {
  localStorage.setItem(localScopeKey(LOCAL_QUICK_ACCESS_KEY), JSON.stringify(Array.from({ length: 10 }, (_, index) => items[index] || null)));
}
function localProfiles(): any[] {
  try { return JSON.parse(localStorage.getItem(localScopeKey(LOCAL_PROFILES_KEY)) || "[]") as any[]; } catch { return []; }
}
function saveLocalProfiles(items: any[]): void {
  localStorage.setItem(localScopeKey(LOCAL_PROFILES_KEY), JSON.stringify(items));
}
function localUsage(profileId?: number): Record<string, { count: number; lastUsedAt: string }> {
  const key = LOCAL_USAGE_KEY + ":" + (profileId || "account");
  try { return JSON.parse(localStorage.getItem(localScopeKey(key)) || "{}") as Record<string, { count: number; lastUsedAt: string }>; } catch { return {}; }
}
function saveLocalUsage(items: Record<string, { count: number; lastUsedAt: string }>, profileId?: number): void {
  const key = LOCAL_USAGE_KEY + ":" + (profileId || "account");
  localStorage.setItem(localScopeKey(key), JSON.stringify(items));
}

function isTransportError(error: unknown): boolean {
  return !error || typeof error !== "object" || typeof (error as ApiError).status !== "number";
}
let browserModelPromise: Promise<BrowserLetterModel> | null = null;


function localFit(samples: LetterSample[]) {
  const grouped: Record<string, number[][]> = {};
  for (const sample of samples) {
    const letter = sample.letter.toUpperCase();
    grouped[letter] = grouped[letter] || [];
    grouped[letter].push(normalizeHandPair(sample.hand_keypoints));
  }
  const entries = Object.entries(grouped);
  if (entries.length < 2) throw new Error("Calibrate at least two different letters.");
  const insufficient = entries.filter(([, values]) => values.length < 3).map(([letter]) => letter);
  if (insufficient.length) {
    throw new Error("Each calibrated letter requires at least 3 examples: " + insufficient.join(", "));
  }
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
  register: async (username: string, email: string, password: string): Promise<RegisterResponse> =>
    LOCAL_MODE
      ? { message: "Local demo account created.", email, verification_required: false, resend_after_seconds: 0 }
      : request<RegisterResponse>("/auth/register", { method: "POST", body: JSON.stringify({ username, email, password }) }),
  verifyOtp: async (email: string, otp: string): Promise<Token> => {
    if (LOCAL_MODE) {
      localUser();
      localStorage.setItem(LOCAL_AUTH_KEY, "1");
      setSessionHint();
      return { access_token: "visionbridge-local-token", token_type: "bearer" };
    }
    const token = await request<Token>("/auth/verify-otp", {
      method: "POST",
      body: JSON.stringify({ email, otp }),
    });
    setSessionHint();
    return token;
  },
  resendOtp: async (email: string): Promise<ResendOtpResponse> => {
    if (LOCAL_MODE) {
      return { message: "Local demo does not send email.", resend_after_seconds: 0 };
    }
    return request<ResendOtpResponse>("/auth/resend-otp", {
      method: "POST",
      body: JSON.stringify({ email }),
    });
  },
  login: async (identifier: string, password: string): Promise<Token> => {
    if (LOCAL_MODE) { localUser(identifier); localStorage.setItem(LOCAL_AUTH_KEY, "1"); setSessionHint(); return { access_token: "visionbridge-local-token", token_type: "bearer" }; }
    const token = await request<Token>("/auth/login", { method: "POST", body: JSON.stringify({ identifier, password }) });
    setSessionHint();
    return token;
  },
  logout: async (): Promise<void> => {
    if (LOCAL_MODE) {
      clearLocalAuth();
      return;
    }
    await request<{ logged_out: boolean }>("/auth/logout", { method: "POST" });
    csrfTokenMemory = null;
  },
  me: async (): Promise<User> => {
    if (LOCAL_MODE) return localUser();
    const user = await request<User>("/users/me");
    localStorage.setItem(LOCAL_USER_KEY, JSON.stringify(user));
    return user;
  },
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
  customWords: async (): Promise<any[]> => {
    if (LOCAL_MODE) return localCustomWords();
    try { return await request<any[]>("/communication/words"); } catch (error) { if (!isTransportError(error)) throw error; return localCustomWords(); }
  },
  createCustomWord: async (phrase: string, category: string): Promise<any> => {
    if (LOCAL_MODE) {
      const items = localCustomWords();
      const exists = items.some((item) => String(item.phrase).toLowerCase() === phrase.trim().toLowerCase());
      if (exists) throw new Error("That custom phrase already exists.");
      const item = { id: Date.now(), phrase: phrase.trim(), category: category.trim(), created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
      saveLocalCustomWords([...items, item]);
      return item;
    }
    try {
      return await request<any>("/communication/words", { method: "POST", body: JSON.stringify({ phrase, category }) });
    } catch (error) {
      if (!isTransportError(error)) throw error;
      const items = localCustomWords();
      const exists = items.some((item) => String(item.phrase).toLowerCase() === phrase.trim().toLowerCase());
      if (exists) throw error;
      const item = { id: Date.now(), phrase: phrase.trim(), category: category.trim(), created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
      saveLocalCustomWords([...items, item]);
      return item;
    }
  },
  updateCustomWord: async (id: number, phrase: string, category: string): Promise<any> => {
    if (LOCAL_MODE) {
      const items = localCustomWords();
      const next = items.map((item) => item.id === id ? { ...item, phrase: phrase.trim(), category: category.trim(), updated_at: new Date().toISOString() } : item);
      saveLocalCustomWords(next);
      return next.find((item) => item.id === id);
    }
    try {
      return await request<any>("/communication/words/" + id, { method: "PUT", body: JSON.stringify({ phrase, category }) });
    } catch (error) {
      if (!isTransportError(error)) throw error;
      const items = localCustomWords();
      const next = items.map((item) => item.id === id ? { ...item, phrase: phrase.trim(), category: category.trim(), updated_at: new Date().toISOString() } : item);
      saveLocalCustomWords(next);
      const updated = next.find((item) => item.id === id);
      if (!updated) throw error;
      return updated;
    }
  },
  deleteCustomWord: async (id: number): Promise<void> => {
    if (LOCAL_MODE) {
      saveLocalCustomWords(localCustomWords().filter((item) => item.id !== id));
      return;
    }
    try { await request<any>("/communication/words/" + id, { method: "DELETE" }); }
    catch (error) { if (!isTransportError(error)) throw error; saveLocalCustomWords(localCustomWords().filter((item) => item.id !== id)); }
  },
  profiles: async (): Promise<any[]> => {
    if (LOCAL_MODE) return localProfiles();
    try { return await request<any[]>("/communication/profiles"); }
    catch (error) { if (!isTransportError(error)) throw error; return localProfiles(); }
  },
  createProfile: async (name: string, config: any): Promise<any> => {
    if (LOCAL_MODE) {
      const items = localProfiles();
      const profile = { id: Date.now(), name: name.trim(), config, created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
      saveLocalProfiles([...items, profile]);
      return profile;
    }
    try { return await request<any>("/communication/profiles", { method: "POST", body: JSON.stringify({ name, config }) }); }
    catch (error) {
      if (!isTransportError(error)) throw error;
      const items = localProfiles();
      const profile = { id: Date.now(), name: name.trim(), config, created_at: new Date().toISOString(), updated_at: new Date().toISOString() };
      saveLocalProfiles([...items, profile]);
      return profile;
    }
  },
  updateProfile: async (id: number, name: string, config: any): Promise<any> => {
    if (LOCAL_MODE) {
      const items = localProfiles().map((item) => item.id === id ? { ...item, name: name.trim(), config, updated_at: new Date().toISOString() } : item);
      saveLocalProfiles(items);
      return items.find((item) => item.id === id);
    }
    try { return await request<any>("/communication/profiles/" + id, { method: "PUT", body: JSON.stringify({ name, config }) }); }
    catch (error) {
      if (!isTransportError(error)) throw error;
      const items = localProfiles().map((item) => item.id === id ? { ...item, name: name.trim(), config, updated_at: new Date().toISOString() } : item);
      saveLocalProfiles(items);
      const item = items.find((item) => item.id === id);
      if (!item) throw new Error("Profile not found.");
      return item;
    }
  },
  deleteProfile: async (id: number): Promise<void> => {
    if (LOCAL_MODE) {
      saveLocalProfiles(localProfiles().filter((item) => item.id !== id));
      return;
    }
    try { await request<any>("/communication/profiles/" + id, { method: "DELETE" }); }
    catch (error) { if (!isTransportError(error)) throw error; saveLocalProfiles(localProfiles().filter((item) => item.id !== id)); }
  },
  recordUsage: async (phrase: string, profileId?: number): Promise<void> => {
    const value = phrase.trim();
    if (!value) return;
    if (LOCAL_MODE) {
      const items = localUsage(profileId);
      const previous = items[value] || { count: 0, lastUsedAt: "" };
      items[value] = { count: previous.count + 1, lastUsedAt: new Date().toISOString() };
      saveLocalUsage(items, profileId);
      return;
    }
    try { await request<{ phrase: string; usage_count: number }>("/communication/usage", { method: "POST", body: JSON.stringify({ phrase: value, profileId: profileId || null }) }); }
    catch (error) {
      if (!isTransportError(error)) return;
      const items = localUsage(profileId);
      const previous = items[value] || { count: 0, lastUsedAt: "" };
      items[value] = { count: previous.count + 1, lastUsedAt: new Date().toISOString() };
      saveLocalUsage(items, profileId);
    }
  },
  mostUsed: async (profileId?: number): Promise<any[]> => {
    if (LOCAL_MODE) {
      const items = localUsage(profileId);
      return Object.entries(items)
        .sort((a, b) => b[1].count - a[1].count || b[1].lastUsedAt.localeCompare(a[1].lastUsedAt))
        .slice(0, 12)
        .map(([phrase, value]) => ({ phrase, usage_count: value.count, last_used_at: value.lastUsedAt }));
    }
    try { return await request<any[]>("/communication/most-used" + (profileId ? "?profileId=" + profileId : "")); } catch (error) {
      if (!isTransportError(error)) throw error;
      const items = localUsage(profileId);
      return Object.entries(items).sort((a, b) => b[1].count - a[1].count || b[1].lastUsedAt.localeCompare(a[1].lastUsedAt)).slice(0, 12).map(([phrase, value]) => ({ phrase, usage_count: value.count, last_used_at: value.lastUsedAt }));
    }
  },
  quickAccess: async (): Promise<{ slots: Array<string | null> }> => {
    if (LOCAL_MODE) return { slots: localQuickAccess() };
    try { return await request<{ slots: Array<string | null> }>("/communication/quick-access"); }
    catch (error) { if (!isTransportError(error)) throw error; return { slots: localQuickAccess() }; }
  },
  saveQuickAccess: async (slots: Array<string | null>): Promise<{ slots: Array<string | null> }> => {
    const normalized = Array.from({ length: 10 }, (_, index) => slots[index] || null);
    if (LOCAL_MODE) {
      saveLocalQuickAccess(normalized);
      return { slots: normalized };
    }
    try {
      return await request<{ slots: Array<string | null> }>("/communication/quick-access", {
        method: "PUT",
        body: JSON.stringify({ slots: normalized }),
      });
    } catch (error) {
      if (!isTransportError(error)) throw error;
      saveLocalQuickAccess(normalized);
      return { slots: normalized };
    }
  },
  clearHistory: async (): Promise<{ deleted: number; storage: "database" | "browser" }> => {
    if (LOCAL_MODE) {
      const count = localHistory().length;
      saveLocalHistory([]);
      return { deleted: count, storage: "browser" };
    }
    return request<{ deleted: number; storage: "database" }>("/history", { method: "DELETE" });
  },
  exportHistoryCsv: async (): Promise<Blob> => {
    if (LOCAL_MODE) {
      const rows = localHistory();
      const header = "id,created_at,predicted_text,confidence,latency_ms,used_adapter\n";
      const body = rows.map((row) => [row.id, row.created_at, row.predicted_text, row.confidence, row.latency_ms, row.used_adapter].join(",")).join("\n");
      return new Blob([header, body], { type: "text/csv" });
    }
    const response = await fetch(API_BASE + "/history/export.csv", {
      credentials: "include",
    });
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
    if (!browserModelPromise) {
      browserModelPromise = request<BrowserModelPayload>("/letter/model")
        .then((payload) => new BrowserLetterModel(payload))
        .catch((error) => {
          browserModelPromise = null;
          throw error;
        });
    }
    return browserModelPromise;
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
        preprocessing_version: "two-hand-wrist-scale-v1",
        landmark_runtime: "mediapipe-hand-landmarker-0.10.35",
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
        used_adapter: event.adapter_id == null ? 0 : 1,
        created_at: new Date().toISOString(),
      }]);
      return;
    }
    await request<{ logged: boolean }>("/letter/event", {
      method: "POST",
      body: JSON.stringify(event),
    });
  },

  letterPredict: async (userId: number, adapterId: number | undefined, handKeypoints: number[]): Promise<LetterPredictionResult> => {
    const started = performance.now();
    if (LOCAL_MODE) {
      if (adapterId == null) throw new Error("The local demo does not bundle the V3 base-model weights; use the deployed browser runtime for base-model inference.");
      const adapter = localLetterAdapters().find((item) => item.id === adapterId);
      if (!adapter) throw new Error("Choose a calibrated signer adapter first.");
      const result = localPredict(adapter, handKeypoints);
      const latency = performance.now() - started;
      return { ...result, latency_ms: latency, adapter_id: adapterId, mode: "adapter" };

    }
    return request<LetterPredictionResult>("/letter/predict", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, adapter_id: adapterId ?? null, hand_keypoints: handKeypoints }),
    });
  },
};
