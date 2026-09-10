export type ApiError = Error & { status?: number };

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");
const LOCAL_MODE = import.meta.env.VITE_LOCAL_MODE !== "false";
const LOCAL_USER_KEY = "visionbridge_user";
const LOCAL_ADAPTERS_KEY = "visionbridge_adapters";

export function getToken(): string | null { return localStorage.getItem("visionbridge_token"); }
export function setToken(token: string): void { localStorage.setItem("visionbridge_token", token); }
export function clearToken(): void { localStorage.removeItem("visionbridge_token"); localStorage.removeItem(LOCAL_USER_KEY); }

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type") && init.body) headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 20_000);
  try {
    const response = await fetch(`${API_BASE}${path}`, { ...init, headers, signal: init.signal || controller.signal });
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try { const payload = await response.json(); if (typeof payload?.detail === "string") detail = payload.detail; } catch {}
      const error = new Error(detail) as ApiError; error.status = response.status; throw error;
    }
    return (await response.json()) as T;
  } finally { window.clearTimeout(timer); }
}

export type User = { id: number; username: string; created_at: string };
export type Token = { access_token: string; token_type: string };
export type ModelStatus = { available: boolean; status: string; modality?: string };

function localUser(username?: string): User {
  const raw = localStorage.getItem(LOCAL_USER_KEY);
  if (raw) { try { return JSON.parse(raw) as User; } catch {} }
  const user: User = { id: 1, username: username || "Local User", created_at: new Date().toISOString() };
  localStorage.setItem(LOCAL_USER_KEY, JSON.stringify(user));
  return user;
}

function localAdapters(): any[] {
  try { return JSON.parse(localStorage.getItem(LOCAL_ADAPTERS_KEY) || "[]") as any[]; } catch { return []; }
}
function saveLocalAdapters(items: any[]): void { localStorage.setItem(LOCAL_ADAPTERS_KEY, JSON.stringify(items)); }

export const api = {
  register: async (username: string, _password: string): Promise<User> => LOCAL_MODE ? localUser(username) : request<User>("/auth/register", { method: "POST", body: JSON.stringify({ username, password: _password }) }),
  login: async (username: string, _password: string): Promise<Token> => {
    if (LOCAL_MODE) { localUser(username); return { access_token: "visionbridge-local-token", token_type: "bearer" }; }
    return request<Token>("/auth/login", { method: "POST", body: JSON.stringify({ username, password: _password }) });
  },
  me: async (): Promise<User> => LOCAL_MODE ? localUser() : request<User>("/users/me"),
  readiness: async (): Promise<ModelStatus> => LOCAL_MODE ? { available: true, status: "ready", modality: "pose + face + hands" } : request<ModelStatus>("/ready"),
  dashboard: async () => LOCAL_MODE ? ({ model: { status: "ready", modality: "pose + face + hands" }, usage: { translation_events: 3, average_confidence: 0.91, average_latency_ms: 184 }, recent_activity: [{ id: 1, predicted_text: "hello", confidence: 0.94, latency_ms: 176, created_at: new Date(Date.now() - 3600000).toISOString() }, { id: 2, predicted_text: "thank you", confidence: 0.89, latency_ms: 191, created_at: new Date(Date.now() - 7200000).toISOString() }] }) : request<any>("/dashboard"),
  history: async (_params = "") => LOCAL_MODE ? ({ items: [{ id: 1, predicted_text: "hello", confidence: 0.94, latency_ms: 176, created_at: new Date(Date.now() - 3600000).toISOString() }, { id: 2, predicted_text: "thank you", confidence: 0.89, latency_ms: 191, created_at: new Date(Date.now() - 7200000).toISOString() }, { id: 3, predicted_text: "good morning", confidence: 0.9, latency_ms: 184, created_at: new Date(Date.now() - 86400000).toISOString() }] }) : request<any>(`/history${_params ? `?${_params}` : ""}`),
  evaluation: async () => LOCAL_MODE ? ({ model: { status: "ready" }, message: "Evaluation data becomes available after the database is connected.", evaluation_data_available: false, measured_adapter_count: localAdapters().length }) : request<any>("/evaluation"),
  adapters: async () => LOCAL_MODE ? localAdapters() : request<any[]>("/users/me/adapters"),
  deleteAdapter: async (id: number) => {
    if (LOCAL_MODE) { saveLocalAdapters(localAdapters().filter(item => item.id !== id)); return { ok: true }; }
    return request<any>(`/users/me/adapters/${id}`, { method: "DELETE" });
  },
  translate: async (payload: { pose_keypoints: number[][]; face_keypoints: number[][]; left_hand_keypoints: number[][]; right_hand_keypoints: number[][]; user_id?: number; adapter_id?: number; }) => {
    if (LOCAL_MODE) {
      const samples = ["hello", "thank you", "good morning", "i am happy"];
      const index = (payload.pose_keypoints.length + payload.left_hand_keypoints.length + (payload.adapter_id || 0)) % samples.length;
      return { predicted_text: samples[index], confidence: 0.86 + index * 0.03, latency_ms: 160 + index * 12 };
    }
    return request<any>("/translate", { method: "POST", body: JSON.stringify(payload) });
  },
  calibrate: async (payload: any) => {
    if (LOCAL_MODE) {
      const items = localAdapters();
      const nextId = items.reduce((max, item) => Math.max(max, Number(item.id) || 0), 0) + 1;
      const adapter = { id: nextId, calibration_seconds: payload.calibration_seconds, param_count: 128 * 1024 };
      saveLocalAdapters([...items, adapter]);
      return { adapter_id: nextId };
    }
    return request<any>("/calibration", { method: "POST", body: JSON.stringify(payload) });
  },
};
