export type ThemePreference = "light" | "dark" | "system";

export const THEME_STORAGE_KEY = "theme-preference";
const LEGACY_THEME_STORAGE_KEY = "visionbridge_theme";

const isThemePreference = (value: string | null): value is ThemePreference =>
  value === "light" || value === "dark" || value === "system";

export function getThemePreference(): ThemePreference {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (isThemePreference(stored)) return stored;
    const legacy = localStorage.getItem(LEGACY_THEME_STORAGE_KEY);
    if (isThemePreference(legacy)) {
      localStorage.setItem(THEME_STORAGE_KEY, legacy);
      return legacy;
    }
  } catch {}
  return "system";
}

function prefersDark(): boolean {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

function syncThemeChrome(preference: ThemePreference): void {
  const resolved = preference === "system" ? (prefersDark() ? "dark" : "light") : preference;
  document.documentElement.style.colorScheme = resolved;
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (meta) {
    const background = getComputedStyle(document.documentElement).getPropertyValue("--vb-bg").trim();
    if (background) meta.content = background;
  }
}

export function applyThemePreference(preference: ThemePreference, persist = true): void {
  if (persist) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, preference);
      localStorage.setItem(LEGACY_THEME_STORAGE_KEY, preference);
    } catch {}
  }
  const root = document.documentElement;
  root.dataset.themePreference = preference;
  if (preference === "system") root.removeAttribute("data-theme");
  else root.dataset.theme = preference;
  syncThemeChrome(preference);
}

export function initializeThemeSystem(): () => void {
  const preference = getThemePreference();
  applyThemePreference(preference, false);
  const media = window.matchMedia?.("(prefers-color-scheme: dark)");
  if (!media) return () => {};
  const handleChange = () => {
    if (getThemePreference() === "system") syncThemeChrome("system");
  };
  media.addEventListener?.("change", handleChange);
  return () => media.removeEventListener?.("change", handleChange);
}
