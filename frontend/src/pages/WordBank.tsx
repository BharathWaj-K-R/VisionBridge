import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { DEFAULT_VOCABULARY, VOCABULARY_CATEGORIES, type VocabularyCategory, type VocabularyItem } from "../data/vocabulary";
import Speakable from "../components/Speakable";
import { QUICK_ACCESS_SLOTS, useQuickAccess } from "../components/QuickAccessContext";
import { usePersonalization } from "../components/PersonalizationContext";
import { Empty, Loading, Page } from "../components/Page";
import { EmptyState, ErrorState } from "../components/SystemStates";

type CustomWord = VocabularyItem & { custom: true; customId: number };

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export default function WordBank() {
  const [customWords, setCustomWords] = useState<CustomWord[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<"All" | VocabularyCategory>("All");
  const [phrase, setPhrase] = useState("");
  const [customCategory, setCustomCategory] = useState<VocabularyCategory>("Greetings & Social");
  const [saving, setSaving] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editPhrase, setEditPhrase] = useState("");
  const [editCategory, setEditCategory] = useState<VocabularyCategory>("Greetings & Social");
  const [message, setMessage] = useState("");
  const [loadError, setLoadError] = useState("");

  const { slots, assign, move, clear } = useQuickAccess();
  const { activeProfile, mostUsed, isFavorite, toggleFavorite } = usePersonalization();

  const loadCustomWords = async () => {
    setLoading(true);
    try {
      const items = await api.customWords();
      setCustomWords(items.map((item: any) => ({
        id: "custom-" + Number(item.id),
        customId: Number(item.id),
        phrase: String(item.phrase),
        category: item.category as VocabularyCategory,
        custom: true,
      })));
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Custom phrases could not be loaded.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void loadCustomWords(); }, []);

  const allWords = useMemo<VocabularyItem[]>(() => [...DEFAULT_VOCABULARY, ...customWords], [customWords]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return allWords.filter((item) => {
      const matchesCategory = category === "All" || item.category === category;
      const matchesQuery = !needle || item.phrase.toLowerCase().includes(needle);
      return matchesCategory && matchesQuery;
    });
  }, [allWords, category, query]);

  const grouped = useMemo(() => {
    const map = new Map<string, VocabularyItem[]>();
    for (const item of filtered) {
      const list = map.get(item.category) || [];
      list.push(item);
      map.set(item.category, list);
    }
    return Array.from(map.entries());
  }, [filtered]);

  async function pinPhrase(value: string) {
    const emptyIndex = slots.findIndex((slot) => !slot);
    if (emptyIndex === -1) {
      setMessage("All 10 Quick Access slots are full. Use the profile manager to replace one.");
      return;
    }
    await assign(emptyIndex, value);
    setMessage("Pinned to Quick Access slot " + (emptyIndex + 1) + ".");
  }

  async function addCustomWord() {
    const next = phrase.trim();
    if (!next) return;
    if (allWords.some((item) => item.phrase.toLowerCase() === next.toLowerCase())) {
      setMessage("That phrase is already in the Word Bank.");
      return;
    }
    setSaving(true);
    setMessage("");
    try {
      const item = await api.createCustomWord(next, customCategory);
      setCustomWords((items) => [...items, {
        id: "custom-" + Number(item.id),
        customId: Number(item.id),
        phrase: String(item.phrase),
        category: item.category as VocabularyCategory,
        custom: true,
      }]);
      setPhrase("");
      setMessage("Phrase saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Phrase could not be saved.");
    } finally {
      setSaving(false);
    }
  }

  function beginEdit(item: CustomWord) {
    setEditingId(item.customId);
    setEditPhrase(item.phrase);
    setEditCategory(item.category);
    setMessage("");
  }

  async function saveEdit() {
    if (editingId == null || !editPhrase.trim()) return;
    setSaving(true);
    setMessage("");
    try {
      const item = await api.updateCustomWord(editingId, editPhrase.trim(), editCategory);
      setCustomWords((items) => items.map((existing) => existing.customId === editingId ? {
        ...existing,
        phrase: String(item.phrase),
        category: item.category as VocabularyCategory,
      } : existing));
      setEditingId(null);
      setMessage("Phrase updated.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Phrase could not be updated.");
    } finally {
      setSaving(false);
    }
  }

  async function removeCustomWord(id: number) {
    if (!window.confirm("Delete this phrase?")) return;
    setSaving(true);
    setMessage("");
    try {
      const target = customWords.find((item) => item.customId === id);
      await api.deleteCustomWord(id);
      setCustomWords((items) => items.filter((item) => item.customId !== id));
      if (target) {
        for (let index = 0; index < slots.length; index += 1) {
          if (slots[index] === target.phrase) await clear(index);
        }
      }
      setMessage("Phrase deleted.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Custom phrase could not be deleted.");
    } finally {
      setSaving(false);
    }
  }

  async function assignWord(phraseValue: string, slotIndex: number) {
    if (!phraseValue.trim()) return;
    setMessage("");
    try {
      await assign(slotIndex, phraseValue);
      setMessage("Assigned to quick slot " + (slotIndex + 1) + ".");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Quick Access could not be updated.");
    }
  }

  return (
    <Page title="Word Bank" subtitle={"Browse communication phrases for " + (activeProfile?.name || "this profile") + ", speak or pin frequently used phrases, and manage ten Quick Access slots."}>
      <section className="panel alphabet-panel">
        <div className="panel-head">
          <div>
            <div className="eyebrow">A–Z voice output</div>
            <h2>Speak any letter</h2>
          </div>
          <span className="status-chip">Click to speak</span>
        </div>
        <div className="alphabet-grid">
          {LETTERS.map((letter) => <Speakable key={letter} text={letter} className="alphabet-tile" />)}
        </div>
      </section>

      <div className="wordbank-priority-grid">
        <section className="panel">
          <div className="panel-head">
            <div>
              <div className="eyebrow">FAVORITES</div>
              <h2>Frequent phrases</h2>
            </div>
            <span className="status-chip">{activeProfile?.config.favorites.length || 0}</span>
          </div>
          {activeProfile?.config.favorites.length ? (
            <div className="priority-chip-grid">
              {activeProfile.config.favorites.map((item) => <Speakable key={item} text={item} className="priority-speak" />)}
            </div>
          ) : <Empty text="Favorite a phrase below to add it to this profile." />}
        </section>

        <section className="panel">
          <div className="panel-head">
            <div>
              <div className="eyebrow">MOST USED</div>
              <h2>Most used phrases</h2>
            </div>
            <span className="status-chip">AUTO</span>
          </div>
          {mostUsed.length ? (
            <div className="priority-chip-grid">
              {mostUsed.slice(0, 6).map((item) => <Speakable key={item.phrase} text={item.phrase} className="priority-speak"><span>{item.phrase}</span><small>{item.usage_count}×</small></Speakable>)}
            </div>
          ) : <Empty text="Use or speak phrases to build usage counts." />}
        </section>
      </div>

      <section className="panel">
        <div className="panel-head">
          <div>
            <div className="eyebrow">Daily vocabulary</div>
            <h2>Communication phrases</h2>
          </div>
          <span className="mono">{filtered.length} ITEMS</span>
        </div>

        <div className="wordbank-filters">
          <input className="compact-input wide-input" placeholder="Search phrases and words" value={query} onChange={(event) => setQuery(event.target.value)} />
          <select value={category} onChange={(event) => setCategory(event.target.value as "All" | VocabularyCategory)}>
            <option value="All">All categories</option>
            {VOCABULARY_CATEGORIES.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </div>

        <div className="vocabulary-groups">
          {grouped.map(([group, items]) => (
            <section key={group} className="vocabulary-group">
              <div className="vocabulary-group-head">
                <h3>{group}</h3>
                <span>{items.length}</span>
              </div>
              <div className="vocabulary-grid">
                {items.map((item) => {
                  const custom = Boolean(item.custom);
                  const favorite = isFavorite(item.phrase);
                  return (
                    <article key={item.id} className="word-card">
                      <Speakable text={item.phrase} className="word-speakable" />
                      <div className="word-card-foot">
                        <span className="word-source">{custom ? "Custom" : item.category.toUpperCase()}</span>
                        <div className="word-card-actions">
                          <button type="button" className={favorite ? "word-action favorite active" : "word-action favorite"} onClick={() => void toggleFavorite(item.phrase)} aria-pressed={favorite} aria-label={(favorite ? "Remove " : "Add ") + item.phrase + " from favorites"}>{favorite ? "★" : "☆"}</button>
                          <button type="button" className="word-action" onClick={() => void pinPhrase(item.phrase)} aria-label={"Add " + item.phrase + " to next empty Quick Access slot"}>PIN</button>
                          <label className="assign-control">SLOT
                            <select defaultValue="" onChange={(event) => {
                              const selected = event.target.value;
                              if (selected !== "") void assignWord(item.phrase, Number(selected));
                              event.currentTarget.value = "";
                            }}>
                              <option value="">Assign…</option>
                              {Array.from({ length: QUICK_ACCESS_SLOTS }, (_, index) => <option key={index} value={index}>Slot {index + 1}{slots[index] ? " · " + slots[index] : ""}</option>)}
                            </select>
                          </label>
                        </div>
                      </div>

                      {custom && (
                        <div className="custom-word-actions">
                          <button type="button" className="text-btn" onClick={() => beginEdit(item as CustomWord)}>Edit</button>
                          <button type="button" className="text-btn danger-text" onClick={() => void removeCustomWord((item as CustomWord).customId)} disabled={saving}>Delete</button>
                        </div>
                      )}
                    </article>
                  );
                })}
              </div>
            </section>
          ))}
        </div>

        {!loading && !grouped.length && <Empty text="No vocabulary items match your search." />}
        {loading && <Loading />}
      </section>

      <section className="panel custom-word-panel">
        {loadError && <ErrorState title="Your custom phrases could not be loaded" message="Check your connection and try again. Your saved phrases have not been changed." onRetry={() => void loadCustomWords()} />}
        {!loading && !loadError && !customWords.length && <EmptyState title="No custom phrases yet" message="Add a phrase you use often and it will appear in your phrase list." />}
        <div className="panel-head">
          <div>
            <div className="eyebrow">Your phrases</div>
            <h2>Add a phrase</h2>
          </div>
          <span className="status-chip">Profile ready</span>
        </div>
        <div className="custom-word-form">
          <label>PHRASE<input value={phrase} onChange={(event) => setPhrase(event.target.value)} maxLength={200} placeholder="e.g. Please call my sister" /></label>
          <label>CATEGORY<select value={customCategory} onChange={(event) => setCustomCategory(event.target.value as VocabularyCategory)}>{VOCABULARY_CATEGORIES.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          <button type="button" className="primary-btn" onClick={() => void addCustomWord()} disabled={saving || !phrase.trim()}>{saving ? "Saving…" : "Add phrase"}</button>
        </div>
      </section>

      <section className="panel quick-manager">
        <div className="panel-head">
          <div>
            <div className="eyebrow">Quick Access settings</div>
            <h2>{activeProfile?.name || "Profile"} · 10 slots</h2>
          </div>
          <Link to="/personalization" className="text-btn">Open full profile →</Link>
        </div>
        <div className="quick-manager-list">
          {Array.from({ length: QUICK_ACCESS_SLOTS }, (_, index) => {
            const value = slots[index] || "";
            return (
              <div className="quick-manager-row" key={index}>
                <strong className="slot-number">{String(index + 1).padStart(2, "0")}</strong>
                <select value={value} onChange={(event) => void assignWord(event.target.value, index)}>
                  <option value="">Empty slot</option>
                  {allWords.map((item) => <option key={item.id} value={item.phrase}>{item.phrase}</option>)}
                </select>
                <Speakable text={value || "Empty slot"} className="quick-manager-speak" label={value ? "Speak slot " + (index + 1) : "Empty slot " + (index + 1)} />
                <button type="button" className="small-icon-btn" onClick={() => void move(index, index - 1)} disabled={index === 0} aria-label={"Move slot " + (index + 1) + " up"}>↑</button>
                <button type="button" className="small-icon-btn" onClick={() => void move(index, index + 1)} disabled={index === QUICK_ACCESS_SLOTS - 1} aria-label={"Move slot " + (index + 1) + " down"}>↓</button>
                <button type="button" className="small-icon-btn danger-icon" onClick={() => void clear(index)} disabled={!value} aria-label={"Clear slot " + (index + 1)}>×</button>
              </div>
            );
          })}
        </div>
      </section>

      {editingId != null && (
        <div className="modal-backdrop">
          <section className="modal-card" role="dialog" aria-modal="true" aria-label="Edit custom phrase">
            <div className="panel-head"><div><div className="eyebrow">Custom PHRASE</div><h2>Edit phrase</h2></div></div>
            <div className="custom-word-form">
              <label>PHRASE<input value={editPhrase} onChange={(event) => setEditPhrase(event.target.value)} maxLength={200} /></label>
              <label>CATEGORY<select value={editCategory} onChange={(event) => setEditCategory(event.target.value as VocabularyCategory)}>{VOCABULARY_CATEGORIES.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
              <div className="button-row">
                <button type="button" className="primary-btn" onClick={() => void saveEdit()} disabled={saving || !editPhrase.trim()}>Save changes</button>
                <button type="button" className="ghost-btn" onClick={() => setEditingId(null)}>Cancel</button>
              </div>
            </div>
          </section>
        </div>
      )}

      {message && <div className="alert history-message" role="status">{message}</div>}
    </Page>
  );
}
