export type ApiError = Error & { status?: number };

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");

export function getToken(): string | null {
  return localStorage.getItem("visionbridge_token");
}

export function setToken(token: string): void {
  localStorage.setItem("visionbridge_token", token);
}

export function clearToken(): void {
  localStorage.removeItem("visionbridge_token");
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type") && init.body) headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 20_000);

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
      signal: init.signal || controller.signal,
    });
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const payload = await response.json();
        if (typeof payload?.detail === "string") detail = payload.detail;
      } catch {
        // Preserve the status-based message when the server did not return JSON.
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

export type User = { id: number; username: string; created_at: string };
export type Token = { access_token: string; token_type: string };
export type ModelStatus = { available: boolean; status: string; modality?: string };

export const api = {
  register: (username: string, password: string) =>
    request<User>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  login: (username: string, password: string) =>
    request<Token>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  me: () => request<User>("/users/me"),
  readiness: () => request<ModelStatus>("/ready"),
  dashboard: () => request<any>("/dashboard"),
  history: (params = "") => request<any>(`/history${params ? `?${params}` : ""}`),
  evaluation: () => request<any>("/evaluation"),
  adapters: () => request<any[]>("/users/me/adapters"),
  deleteAdapter: (id: number) => request<any>(`/users/me/adapters/${id}`, { method: "DELETE" }),
  translate: (payload: {
    pose_keypoints: number[][];
    face_keypoints: number[][];
    left_hand_keypoints: number[][];
    right_hand_keypoints: number[][];
    user_id?: number;
    adapter_id?: number;
  }) => request<any>("/translate", { method: "POST", body: JSON.stringify(payload) }),
  calibrate: (payload: any) => request<any>("/calibration", { method: "POST", body: JSON.stringify(payload) }),
};
