import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "../api";
import { DEFAULT_AVATAR, type AvatarPreferences } from "./AvatarCustomizer";

export const PROFILE_SPEEDS = [0.5, 0.75, 1] as const;
export const PROFILE_DEFAULTS = {
  quickAccess: Array<string | null>(10).fill(null),
  favorites: [] as string[],
  signingSpeed: 1,
  ttsVoice: null as string | null,
};

export type PersonalizationConfig = {
  avatar: AvatarPreferences;
  quickAccess: Array<string | null>;
  favorites: string[];
  signingSpeed: number;
  ttsVoice: string | null;
};

export type PersonalizationProfile = {
  id: number;
  name: string;
  config: PersonalizationConfig;
  created_at?: string;
  updated_at?: string;
};

type ProfileContextValue = {
  profiles: PersonalizationProfile[];
  activeProfile: PersonalizationProfile | null;
  loading: boolean;
  saving: boolean;
  voices: SpeechSynthesisVoice[];
  refresh: () => Promise<void>;
  switchProfile: (id: number) => Promise<void>;
  createProfile: (name: string, seed?: PersonalizationConfig) => Promise<void>;
  renameProfile: (name: string) => Promise<void>;
  deleteProfile: (id: number) => Promise<void>;
  updateConfig: (patch: Partial<PersonalizationConfig>) => Promise<void>;
  toggleFavorite: (phrase: string) => Promise<void>;
  isFavorite: (phrase: string) => boolean;
  refreshMostUsed: () => Promise<void>;
  mostUsed: Array<{ phrase: string; usage_count: number; last_used_at?: string }>;
};

const PersonalizationContext = createContext<ProfileContextValue | null>(null);
const ACTIVE_PROFILE_KEY = "visionbridge_active_profile_id";

function normalizeConfig(raw: Partial<PersonalizationConfig> | undefined): PersonalizationConfig {
  const avatar = { ...DEFAULT_AVATAR, ...(raw?.avatar || {}) };
  const quickAccess = Array.from({ length: 10 }, (_, index) => raw?.quickAccess?.[index] || null);
  const favorites = Array.from(new Set((raw?.favorites || []).map((item) => item.trim()).filter(Boolean))).slice(0, 50);
  const speed = PROFILE_SPEEDS.includes(Number(raw?.signingSpeed) as 0.5 | 0.75 | 1) ? Number(raw?.signingSpeed) : 1;
  return {
    avatar,
    quickAccess,
    favorites,
    signingSpeed: speed,
    ttsVoice: raw?.ttsVoice || null,
  };
}

function legacyProfileSeed(): { avatar: AvatarPreferences; quickAccess: Array<string | null> } {
  const avatar = (() => {
    try {
      const raw = localStorage.getItem("visionbridge_avatar_preferences");
      if (!raw) return DEFAULT_AVATAR;
      return { ...DEFAULT_AVATAR, ...(JSON.parse(raw) as Partial<AvatarPreferences>) };
    } catch {
      return DEFAULT_AVATAR;
    }
  })();

  const quickAccess = (() => {
    try {
      const raw = localStorage.getItem("visionbridge_user");
      let suffix = "anonymous";
      if (raw) {
        const user = JSON.parse(raw) as { username?: string };
        if (user.username) suffix = encodeURIComponent(user.username);
      }
      const stored = JSON.parse(localStorage.getItem("visionbridge_quick_access:" + suffix) || "[]");
      return Array.from({ length: 10 }, (_, index) => stored[index] || null);
    } catch {
      return Array<string | null>(10).fill(null);
    }
  })();

  return { avatar, quickAccess };
}

function localActiveKey(): string {
  const raw = localStorage.getItem("visionbridge_user");
  if (!raw) return ACTIVE_PROFILE_KEY + ":anonymous";
  try {
    const user = JSON.parse(raw) as { username?: string };
    return ACTIVE_PROFILE_KEY + ":" + encodeURIComponent(user.username || "anonymous");
  } catch {
    return ACTIVE_PROFILE_KEY + ":anonymous";
  }
}

export function PersonalizationProvider({ children }: { children: ReactNode }) {
  const [profiles, setProfiles] = useState<PersonalizationProfile[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [mostUsed, setMostUsed] = useState<Array<{ phrase: string; usage_count: number; last_used_at?: string }>>([]);
  const bootstrapRef = useRef(false);

  const loadVoices = () => {
    if (!("speechSynthesis" in window)) return;
    setVoices(window.speechSynthesis.getVoices());
  };

  useEffect(() => {
    loadVoices();
    window.speechSynthesis?.addEventListener("voiceschanged", loadVoices);
    return () => window.speechSynthesis?.removeEventListener("voiceschanged", loadVoices);
  }, []);

  const refresh = async () => {
    if (bootstrapRef.current) return;
    bootstrapRef.current = true;
    setLoading(true);
    try {
      let items = (await api.profiles()) as PersonalizationProfile[];
      if (!items.length) {
        const legacy = import.meta.env.VITE_LOCAL_MODE !== "false" ? legacyProfileSeed() : {
          avatar: DEFAULT_AVATAR,
          quickAccess: Array<string | null>(10).fill(null),
        };
        const created = await api.createProfile("Home", {
          avatar: legacy.avatar,
          ...PROFILE_DEFAULTS,
          quickAccess: legacy.quickAccess,
        });
        items = [created as PersonalizationProfile];
      }
      const normalized = items.map((item) => ({ ...item, config: normalizeConfig(item.config) }));
      setProfiles(normalized);

      const stored = Number(localStorage.getItem(localActiveKey()) || "");
      const nextId = normalized.some((item) => item.id === stored) ? stored : normalized[0].id;
      setActiveId(nextId);
      localStorage.setItem(localActiveKey(), String(nextId));
    } finally {
      setLoading(false);
      bootstrapRef.current = false;
    }
  };

  useEffect(() => { void refresh(); }, []);

  const activeProfile = useMemo(
    () => profiles.find((item) => item.id === activeId) || null,
    [profiles, activeId],
  );

  const switchProfile = async (id: number) => {
    if (!profiles.some((item) => item.id === id)) return;
    setActiveId(id);
    localStorage.setItem(localActiveKey(), String(id));
  };

  const createProfile = async (name: string, seed?: PersonalizationConfig) => {
    const trimmed = name.trim();
    if (!trimmed) throw new Error("Profile name is required.");
    if (profiles.some((item) => item.name.toLowerCase() === trimmed.toLowerCase())) {
      throw new Error("A profile with that name already exists.");
    }
    const created = await api.createProfile(trimmed, normalizeConfig(seed || activeProfile?.config || {
      avatar: DEFAULT_AVATAR,
      ...PROFILE_DEFAULTS,
    }));
    const profile = { ...created, config: normalizeConfig(created.config) } as PersonalizationProfile;
    setProfiles((items) => [...items, profile]);
    setActiveId(profile.id);
    localStorage.setItem(localActiveKey(), String(profile.id));
  };

  const renameProfile = async (name: string) => {
    if (!activeProfile) return;
    setSaving(true);
    try {
      const updated = await api.updateProfile(activeProfile.id, name, activeProfile.config);
      const profile = { ...updated, config: normalizeConfig(updated.config) } as PersonalizationProfile;
      setProfiles((items) => items.map((item) => item.id === profile.id ? profile : item));
    } finally {
      setSaving(false);
    }
  };

  const deleteProfile = async (id: number) => {
    if (profiles.length <= 1) throw new Error("Keep at least one profile.");
    setSaving(true);
    try {
      await api.deleteProfile(id);
      const remaining = profiles.filter((item) => item.id !== id);
      setProfiles(remaining);
      if (id === activeId) {
        const next = remaining[0];
        setActiveId(next.id);
        localStorage.setItem(localActiveKey(), String(next.id));
      }
    } finally {
      setSaving(false);
    }
  };

  const updateConfig = async (patch: Partial<PersonalizationConfig>) => {
    if (!activeProfile) return;
    const nextConfig = normalizeConfig({ ...activeProfile.config, ...patch });
    setSaving(true);
    setProfiles((items) => items.map((item) => item.id === activeProfile.id ? { ...item, config: nextConfig } : item));
    try {
      const updated = await api.updateProfile(activeProfile.id, activeProfile.name, nextConfig);
      const profile = { ...updated, config: normalizeConfig(updated.config) } as PersonalizationProfile;
      setProfiles((items) => items.map((item) => item.id === profile.id ? profile : item));
    } finally {
      setSaving(false);
    }
  };

  const toggleFavorite = async (phrase: string) => {
    if (!activeProfile) return;
    const current = activeProfile.config.favorites;
    const exists = current.some((item) => item.toLowerCase() === phrase.toLowerCase());
    const favorites = exists
      ? current.filter((item) => item.toLowerCase() !== phrase.toLowerCase())
      : [...current, phrase].slice(-50);
    await updateConfig({ favorites });
  };

  const isFavorite = (phrase: string) =>
    Boolean(activeProfile?.config.favorites.some((item) => item.toLowerCase() === phrase.toLowerCase()));

  const refreshMostUsed = async () => {
    if (!activeProfile) return;
    setMostUsed((await api.mostUsed(activeProfile.id)) as typeof mostUsed);
  };

  useEffect(() => { if (activeProfile) void refreshMostUsed(); }, [activeProfile?.id]);
  useEffect(() => {
    const refresh = () => { if (activeProfile) void refreshMostUsed(); };
    window.addEventListener("visionbridge:usage-changed", refresh);
    return () => window.removeEventListener("visionbridge:usage-changed", refresh);
  }, [activeProfile?.id]);

  const value = useMemo(() => ({
    profiles, activeProfile, loading, saving, voices, refresh, switchProfile, createProfile,
    renameProfile, deleteProfile, updateConfig, toggleFavorite, isFavorite, refreshMostUsed, mostUsed,
  }), [profiles, activeProfile, loading, saving, voices, mostUsed]);

  return <PersonalizationContext.Provider value={value}>{children}</PersonalizationContext.Provider>;
}

export function usePersonalization() {
  const value = useContext(PersonalizationContext);
  if (!value) throw new Error("usePersonalization must be used inside PersonalizationProvider");
  return value;
}
