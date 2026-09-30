import { useEffect, useState } from "react";

export type AvatarPreferences = {
  skinTone: string;
  hair: "short" | "curly" | "long";
  hairColor: string;
  shirtColor: string;
};

export const DEFAULT_AVATAR: AvatarPreferences = {
  skinTone: "#b97850",
  hair: "short",
  hairColor: "#1b1a18",
  shirtColor: "#244f7a",
};

const STORAGE_KEY = "visionbridge_avatar_preferences";

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
}: {
  value: AvatarPreferences;
  onChange: (value: AvatarPreferences) => void;
}) {
  const [draft, setDraft] = useState(value);

  useEffect(() => setDraft(value), [value]);

  const update = (patch: Partial<AvatarPreferences>) => {
    const next = { ...draft, ...patch };
    setDraft(next);
    onChange(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  return (
    <section className="panel avatar-customizer">
      <div className="panel-head">
        <div>
          <div className="eyebrow">AVATAR</div>
          <h2>Customize</h2>
        </div>
        <span className="status-chip">LOCAL PROFILE</span>
      </div>

      <div className="customizer-grid">
        <label>SKIN TONE
          <span className="swatch-row">
            {["#f0c39d", "#d79a72", "#b97850", "#8c5539", "#5f3728"].map((tone) => (
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

        <label>CLOTHING
          <span className="swatch-row">
            {["#244f7a", "#226b68", "#6d314a", "#4a4b45", "#6a5639"].map((tone) => (
              <button type="button" key={tone} className={draft.shirtColor === tone ? "swatch selected" : "swatch"} style={{ background: tone }} onClick={() => update({ shirtColor: tone })} aria-label={"Clothing color " + tone} />
            ))}
          </span>
        </label>
      </div>
    </section>
  );
}
