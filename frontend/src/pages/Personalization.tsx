import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import AvatarCustomizer from "../components/AvatarCustomizer";
import Speakable from "../components/Speakable";
import { QUICK_ACCESS_SLOTS, useQuickAccess } from "../components/QuickAccessContext";
import { PROFILE_DEFAULTS, PROFILE_SPEEDS, usePersonalization, type PersonalizationConfig } from "../components/PersonalizationContext";
import { Empty, Page } from "../components/Page";

const PRESETS: Record<string, Partial<PersonalizationConfig>> = {
  Home: {
    quickAccess: ["Hello", "I am hungry", "I need water", "I need help", "Bathroom", "Medicine", "I am tired", "Thank you", "Please", "Good night"],
    favorites: ["I need water", "I need help", "Bathroom", "Thank you"],
  },
  Work: {
    quickAccess: ["Good morning", "Please wait", "Please write it down", "I understand", "I do not understand", "Meeting", "Call me", "Thank you", "One moment", "Goodbye"],
    favorites: ["Please write it down", "I understand", "Please wait", "Thank you"],
  },
  School: {
    quickAccess: ["Good morning", "Teacher", "I need help", "Please repeat", "I do not understand", "Where", "Lunch", "Bathroom", "Thank you", "See you"],
    favorites: ["I need help", "Please repeat", "I do not understand", "Thank you"],
  },
};

export default function Personalization() {
  const {
    profiles, activeProfile, loading, saving, voices,
    createProfile, renameProfile, deleteProfile, updateConfig,
    switchProfile, isFavorite, toggleFavorite, mostUsed,
  } = usePersonalization();
  const { slots, assign, move, clear } = useQuickAccess();

  const [newName, setNewName] = useState("");
  const [nameDraft, setNameDraft] = useState(activeProfile?.name || "");
  const [message, setMessage] = useState("");
  const [slotDrafts, setSlotDrafts] = useState<Array<string | null>>(slots);

  useEffect(() => {
    if (activeProfile) setNameDraft(activeProfile.name);
  }, [activeProfile?.id, activeProfile?.name]);

  useEffect(() => {
    setSlotDrafts(slots);
  }, [activeProfile?.id, slots.join("|")]);

  if (loading || !activeProfile) {
    return <Page title="My Profile" subtitle="Loading your personal communication workspace."><div className="loading-page">Loading profile…</div></Page>;
  }

  async function createNamedProfile(name: string, preset?: Partial<PersonalizationConfig>) {
    const seed: PersonalizationConfig = {
      ...activeProfile.config,
      ...preset,
      avatar: { ...activeProfile.config.avatar },
      quickAccess: preset?.quickAccess ? Array.from({ length: 10 }, (_, i) => preset.quickAccess?.[i] || null) : [...activeProfile.config.quickAccess],
      favorites: preset?.favorites ? [...preset.favorites] : [...activeProfile.config.favorites],
      signingSpeed: preset?.signingSpeed || activeProfile.config.signingSpeed,
      ttsVoice: preset?.ttsVoice ?? activeProfile.config.ttsVoice,
    };
    try {
      await createProfile(name, seed);
      setNewName("");
      setMessage("Profile created and activated.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Profile could not be created.");
    }
  }

  async function saveName() {
    if (!nameDraft.trim() || nameDraft.trim() === activeProfile.name) return;
    try {
      await renameProfile(nameDraft);
      setMessage("Profile name saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Profile name could not be saved.");
    }
  }

  async function removeProfile() {
    if (!window.confirm("Delete the profile " + activeProfile.name + "?")) return;
    try {
      await deleteProfile(activeProfile.id);
      setMessage("Profile deleted.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Profile could not be deleted.");
    }
  }

  async function saveSlot(slot: number) {
    try {
      await assign(slot, slotDrafts[slot] || null);
      setMessage("Quick Access saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Quick Access could not be saved.");
    }
  }

  return (
    <Page title="My Profile" subtitle="Make VisionBridge feel like your communication tool. Each profile carries its own avatar, quick access, favorites, signing speed, and preferred voice.">
      <div className="profile-workspace">
        <section className="panel profile-switch-panel">
          <div className="panel-head">
            <div>
              <div className="eyebrow">PERSONAL PROFILES</div>
              <h2>Switch your communication context</h2>
            </div>
            <span className="status-chip">{profiles.length} PROFILE{profiles.length === 1 ? "" : "S"}</span>
          </div>

          <div className="profile-cards">
            {profiles.map((profile) => (
              <button
                type="button"
                key={profile.id}
                className={profile.id === activeProfile.id ? "profile-card active" : "profile-card"}
                onClick={() => void switchProfile(profile.id)}
              >
                <span className="profile-card-index">{String(profiles.indexOf(profile) + 1).padStart(2, "0")}</span>
                <strong>{profile.name}</strong>
                <small>{profile.config.favorites.length} favorites · {profile.config.quickAccess.filter(Boolean).length}/10 pinned</small>
              </button>
            ))}
          </div>

          <div className="profile-create-row">
            <input value={newName} onChange={(event) => setNewName(event.target.value)} maxLength={80} placeholder="New profile name, e.g. Clinic" />
            <button type="button" className="primary-btn" onClick={() => void createNamedProfile(newName)} disabled={!newName.trim() || saving}>Create profile</button>
          </div>

          <div className="profile-presets">
            <span className="eyebrow">QUICK PRESETS</span>
            {Object.keys(PRESETS).map((name) => (
              <button type="button" key={name} className="ghost-btn profile-preset" onClick={() => void createNamedProfile(name + " " + (profiles.length + 1), PRESETS[name])} disabled={saving}>
                + {name}
              </button>
            ))}
          </div>
        </section>

        <div className="profile-two-column">
          <section className="panel profile-identity-panel">
            <div className="eyebrow">ACTIVE PROFILE</div>
            <h2>{activeProfile.name}</h2>
            <div className="profile-name-editor">
              <label>PROFILE NAME
                <input value={nameDraft} onChange={(event) => setNameDraft(event.target.value)} maxLength={80} />
              </label>
              <button type="button" className="primary-btn" onClick={() => void saveName()} disabled={saving || !nameDraft.trim() || nameDraft.trim() === activeProfile.name}>Save name</button>
            </div>
            <button type="button" className="danger-btn profile-delete" onClick={() => void removeProfile()} disabled={saving || profiles.length <= 1}>Delete profile</button>
            <p className="muted profile-tip">Keep one profile for everyday communication. Home, Work, School, Clinic, Family, or any context you use repeatedly can each have their own vocabulary rhythm.</p>
          </section>

          <section className="panel">
            <div className="eyebrow">SIGNING PREFERENCES</div>
            <h2>Speed & voice</h2>
            <div className="profile-preference-form">
              <label>SIGNING SPEED
                <select value={activeProfile.config.signingSpeed} onChange={(event) => void updateConfig({ signingSpeed: Number(event.target.value) })}>
                  {PROFILE_SPEEDS.map((speed) => <option key={speed} value={speed}>{speed}×</option>)}
                </select>
              </label>
              <label>PREFERRED VOICE
                <select value={activeProfile.config.ttsVoice || ""} onChange={(event) => void updateConfig({ ttsVoice: event.target.value || null })}>
                  <option value="">Browser default</option>
                  {voices.map((voice) => <option key={voice.voiceURI || voice.name} value={voice.name}>{voice.name} · {voice.lang}</option>)}
                </select>
              </label>
            </div>
            <p className="muted profile-tip">{voices.length ? voices.length + " browser voices available on this device." : "Voice choices will appear when the browser exposes its speech voices."}</p>
          </section>
        </div>

        <AvatarCustomizer value={activeProfile.config.avatar} onChange={(avatar) => void updateConfig({ avatar })} />

        <section className="panel quick-profile-panel">
          <div className="panel-head">
            <div>
              <div className="eyebrow">PROFILE QUICK ACCESS</div>
              <h2>Ten slots for this context</h2>
            </div>
            <span className="status-chip">10 / 10</span>
          </div>

          <div className="quick-profile-grid">
            {Array.from({ length: QUICK_ACCESS_SLOTS }, (_, index) => (
              <div className="quick-profile-row" key={index}>
                <strong>{String(index + 1).padStart(2, "0")}</strong>
                <input
                  value={slotDrafts[index] || ""}
                  onChange={(event) => setSlotDrafts((current) => { const next = [...current]; next[index] = event.target.value || null; return next; })}
                  onBlur={() => void saveSlot(index)}
                  onKeyDown={(event) => { if (event.key === "Enter") { event.currentTarget.blur(); } }}
                  maxLength={200}
                  placeholder="Assign a phrase"
                  list="visionbridge-wordbank-options"
                />
                <Speakable text={slotDrafts[index] || "Empty slot"} className="quick-profile-speak" />
                <button type="button" className="small-icon-btn" onClick={() => void move(index, index - 1)} disabled={index === 0}>↑</button>
                <button type="button" className="small-icon-btn" onClick={() => void move(index, index + 1)} disabled={index === 9}>↓</button>
                <button type="button" className="small-icon-btn danger-icon" onClick={() => void clear(index)} disabled={!slots[index]}>×</button>
              </div>
            ))}
          </div>
          <datalist id="visionbridge-wordbank-options">
            {activeProfile.config.favorites.map((item) => <option key={item} value={item} />)}
          </datalist>
        </section>

        <div className="profile-two-column">
          <section className="panel">
            <div className="panel-head">
              <div>
                <div className="eyebrow">FAVORITES</div>
                <h2>My saved phrases</h2>
              </div>
              <Link to="/word-bank" className="text-btn">Browse Word Bank →</Link>
            </div>
            {activeProfile.config.favorites.length ? (
              <div className="favorite-list">
                {activeProfile.config.favorites.map((phrase) => (
                  <div className="favorite-row" key={phrase}>
                    <Speakable text={phrase} className="favorite-speak" />
                    <button type="button" className="text-btn danger-text" onClick={() => void toggleFavorite(phrase)}>Remove</button>
                  </div>
                ))}
              </div>
            ) : <Empty text="Favorite a phrase in Word Bank and it will appear here." />}
          </section>

          <section className="panel">
            <div className="panel-head">
              <div>
                <div className="eyebrow">MOST USED</div>
                <h2>What you actually say</h2>
              </div>
              <span className="status-chip">LIVE COUNTS</span>
            </div>
            {mostUsed.length ? (
              <div className="most-used-list">
                {mostUsed.map((item) => (
                  <Speakable key={item.phrase} text={item.phrase} className="most-used-row" />
                ))}
              </div>
            ) : <Empty text="Speak or click phrases to start building your personal usage list." />}
          </section>
        </div>

        {message && <div className="alert history-message" role="status">{message}</div>}
      </div>
    </Page>
  );
}
