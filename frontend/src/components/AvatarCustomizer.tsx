import { useEffect, useState } from "react";

export type AvatarPreferences = {
  skinTone: string;
  hair: "short" | "curly" | "long";
  hairColor: string;
  shirtColor: string;
  bodyShape: "slim" | "average" | "athletic";
  apparel: "tee" | "vest" | "button-down";
  highContrast: boolean;
};

export const DEFAULT_AVATAR: AvatarPreferences = {
  skinTone: "#b97850",
  hair: "short",
  hairColor: "#1b1a18",
  shirtColor: "#244f7a",
  bodyShape: "average",
  apparel: "tee",
  highContrast: false,
};

const STORAGE_KEY = "visionbridge_avatar_preferences";
// Retained for compatibility with older callers. Controlled profile state is the source of truth.

export function loadAvatarPreferences(): AvatarPreferences {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") as Partial<AvatarPreferences> | null;
    return { ...DEFAULT_AVATAR, ...(saved || {}) };
  } catch {
    return DEFAULT_AVATAR;
  }
}

export default function AvatarCustomizer({
  value,
  onChange,
  onResetView,
}: {
  value: AvatarPreferences;
  onChange: (value: AvatarPreferences) => void;
  onResetView?: () => void;
}) {
  const [draft, setDraft] = useState(value);

  useEffect(() => setDraft(value), [value]);

  const update = (patch: Partial<AvatarPreferences>) => {
    const next = { ...draft, ...patch };
    setDraft(next);
    onChange(next);
  };

  return (
    <section className="panel avatar-customizer">
      <div className="panel-head">
        <div>
          <div className="eyebrow">2D CUSTOMIZATION</div>
          <h2>Avatar profile</h2>
        </div>
        <span className="status-chip">{draft.highContrast ? "HIGH CONTRAST" : "STANDARD"}</span>
      </div>

      <div className="customizer-grid avatar-customizer-grid">
        <label>BODY SHAPE
          <select value={draft.bodyShape} onChange={(event) => update({ bodyShape: event.target.value as AvatarPreferences["bodyShape"] })}>
            <option value="slim">Slim</option>
            <option value="average">Average</option>
            <option value="athletic">Athletic</option>
          </select>
        </label>

        <label>APPAREL
          <select value={draft.apparel} onChange={(event) => update({ apparel: event.target.value as AvatarPreferences["apparel"] })}>
            <option value="tee">Fitted Tee</option>
            <option value="vest">Vest</option>
            <option value="button-down">Button-down</option>
          </select>
        </label>

        <label>SKIN TONE
          <span className="swatch-row">
            {["#f3e0d8", "#d7a17e", "#b97850", "#8c5539", "#5f3728", "#2c1b18"].map((tone) => (
              <button type="button" key={tone} className={draft.skinTone === tone ? "swatch selected" : "swatch"} style={{ background: tone }} onClick={() => update({ skinTone: tone })} aria-label={"Skin tone " + tone} />
            ))}
          </span>
        </label>

        <label>HAIR STYLE
          <select value={draft.hair} onChange={(event) => update({ hair: event.target.value as AvatarPreferences["hair"] })}>
            <option value="short">Short</option>
            <option value="curly">Curly</option>
            <option value="long">Long</option>
          </select>
        </label>

        <label>HAIR COLOR
          <span className="swatch-row">
            {["#161616", "#513527", "#8b442c", "#c4b39c"].map((tone) => (
              <button type="button" key={tone} className={draft.hairColor === tone ? "swatch selected" : "swatch"} style={{ background: tone }} onClick={() => update({ hairColor: tone })} aria-label={"Hair color " + tone} />
            ))}
          </span>
        </label>

        <label>HAND OUTLINES
          <button type="button" className={draft.highContrast ? "contrast-toggle on" : "contrast-toggle"} aria-pressed={draft.highContrast} onClick={() => update({ highContrast: !draft.highContrast })}>
            <span className="toggle-track"><i /></span>
            {draft.highContrast ? "High contrast on" : "High contrast off"}
          </button>
        </label>

        <label>CLOTHING COLOR
          <span className="swatch-row">
            {["#1e1e1e", "#1a2238", "#244f7a", "#226b68", "#6d314a", "#6a5639"].map((tone) => (
              <button type="button" key={tone} className={draft.shirtColor === tone ? "swatch selected" : "swatch"} style={{ background: tone }} onClick={() => update({ shirtColor: tone })} aria-label={"Clothing color " + tone} />
            ))}
          </span>
        </label>

        {onResetView && (
          <button type="button" className="ghost-btn avatar-reset-view" onClick={onResetView}>
            RESET VIEW
          </button>
        )}
      </div>
    </section>
  );
}
