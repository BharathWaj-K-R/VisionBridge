import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { DEFAULT_VOCABULARY, VOCABULARY_CATEGORIES, type VocabularyCategory, type VocabularyItem } from "../data/vocabulary";
import Speakable from "../components/Speakable";
import { QUICK_ACCESS_SLOTS, useQuickAccess } from "../components/QuickAccessContext";
import { Empty, Loading, Page } from "../components/Page";

type CustomWord = VocabularyItem & { id: number; custom: true };

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
  const [editCategory, setEditCategory] = useState<VocabularyCategory>("Custom");
  const [message, setMessage] = useState("");

  const { slots, loading: quickLoading, assign, move, clear } = useQuickAccess();

  const loadCustomWords = async () => {
    setLoading(true);
    try {
      const items = await api.customWords();
      setCustomWords(items.map((item: any) => ({
        id: Number(item.id),
        phrase: String(item.phrase),
        category: item.category as VocabularyCategory,
        custom: true,
      })));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Custom phrases could not be loaded.");
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
        id: Number(item.id),
        phrase: String(item.phrase),
        category: item.category as VocabularyCategory,
        custom: true,
      }]);
      setPhrase("");
      setMessage("Custom phrase saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Custom phrase could not be saved.");
    } finally {
      setSaving(false);
    }
  }

  function beginEdit(item: CustomWord) {
    setEditingId(item.id);
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
      setCustomWords((items) => items.map((existing) => existing.id === editingId ? {
        ...existing,
        phrase: String(item.phrase),
        category: item.category as VocabularyCategory,
      } : existing));
      setEditingId(null);
      setMessage("Custom phrase updated.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Custom phrase could not be updated.");
    } finally {
      setSaving(false);
    }
  }

  async function removeCustomWord(id: number) {
    if (!window.confirm("Delete this custom phrase?")) return;
    setSaving(true);
    setMessage("");
    try {
      const target = customWords.find((item) => item.id === id);
      await api.deleteCustomWord(id);
      setCustomWords((items) => items.filter((item) => item.id !== id));
      if (target) {
        for (let index = 0; index < slots.length; index += 1) {
          if (slots[index] === target.phrase) await clear(index);
        }
      }
      setMessage("Custom phrase deleted.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Custom phrase could not be deleted.");
    } finally {
      setSaving(false);
    }
  }

  async function assignWord(phraseValue: string, slotIndex: number) {
    setMessage("");
    try {
      await assign(slotIndex, phraseValue);
      setMessage("Assigned to quick slot " + (slotIndex + 1) + ".");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Quick Access could not be updated.");
    }
  }

  return (
    <Page title="Word Bank" subtitle="Browse practical daily phrases, speak them aloud, add your own, and build a ten-slot communication bar.">
      <section className="panel alphabet-panel">
        <div className="panel-head">
          <div>
            <div className="eyebrow">A–Z VOICE OUTPUT</div>
            <h2>Every letter is speakable</h2>
          </div>
          <span className="status-chip">CLICK TO SPEAK</span>
        </div>
        <div className="alphabet-grid">
          {LETTERS.map((letter) => <Speakable key={letter} text={letter} className="alphabet-tile" />)}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <div>
            <div className="eyebrow">DAILY VOCABULARY</div>
            <h2>Useful communication phrases</h2>
          </div>
          <span className="mono">{filtered.length} ITEMS</span>
        </div>

        <div className="wordbank-filters">
          <input className="compact-input wide-input" placeholder="Search words and phrases" value={query} onChange={(event) => setQuery(event.target.value)} />
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
                  return (
                    <article key={item.id} className="word-card">
                      <Speakable text={item.phrase} className="word-speakable" />
                      <div className="word-card-foot">
                        <span className="word-source">{custom ? "CUSTOM" : item.category.toUpperCase()}</span>
                        <label className="assign-control">QUICK SLOT
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

                      {custom && (
                        <div className="custom-word-actions">
                          <button type="button" className="text-btn" onClick={() => beginEdit(item as CustomWord)}>Edit</button>
                          <button type="button" className="text-btn danger-text" onClick={() => void removeCustomWord((item as CustomWord).id)} disabled={saving}>Delete</button>
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
        <div className="panel-head">
          <div>
            <div className="eyebrow">YOUR WORDS</div>
            <h2>Add a custom phrase</h2>
          </div>
          <span className="status-chip">USER SAVED</span>
        </div>
        <div className="custom-word-form">
          <label>PHRASE
            <input value={phrase} onChange={(event) => setPhrase(event.target.value)} maxLength={200} placeholder="e.g. Please call my sister" />
          </label>
          <label>CATEGORY
            <select value={customCategory} onChange={(event) => setCustomCategory(event.target.value as VocabularyCategory)}>
              {VOCABULARY_CATEGORIES.map((item) => <option key={item} value={item}>{item}</option>)}
              <option value="Accessibility">Accessibility</option>
            </select>
          </label>
          <button type="button" className="primary-btn" onClick={() => void addCustomWord()} disabled={saving || !phrase.trim()}>
            {saving ? "Saving…" : "Add phrase"}
          </button>
        </div>
      </section>

      <section className="panel quick-manager">
        <div className="panel-head">
          <div>
            <div className="eyebrow">QUICK ACCESS CONFIG</div>
            <h2>Your 10 permanent slots</h2>
          </div>
          <span className="status-chip">{quickLoading ? "LOADING" : "10 SLOTS"}</span>
        </div>

        <div className="quick-manager-list">
          {Array.from({ length: QUICK_ACCESS_SLOTS }, (_, index) => {
            const value = slots[index] || "";
            return (
              <div className="quick-manager-row" key={index}>
                <strong className="slot-number">{String(index + 1).padStart(2, "0")}</strong>
                <select value={value} onChange={(event) => void assignWord(event.target.value || "", index)}>
                  <option value="">Empty slot</option>
                  {allWords.map((item) => <option key={item.id} value={item.phrase}>{item.phrase}</option>)}
                </select>
                <Speakable text={value || "Empty slot"} className="quick-manager-speak" label={value ? "Speak slot " + (index + 1) : "Empty slot " + (index + 1)} />
                <button type="button" className="small-icon-btn" onClick={() => void move(index, index - 1)} disabled={index === 0} aria-label={"Move slot " + (index + 1) + " left"}>↑</button>
                <button type="button" className="small-icon-btn" onClick={() => void move(index, index + 1)} disabled={index === QUICK_ACCESS_SLOTS - 1} aria-label={"Move slot " + (index + 1) + " right"}>↓</button>
                <button type="button" className="small-icon-btn danger-icon" onClick={() => void clear(index)} disabled={!value} aria-label={"Clear slot " + (index + 1)}>×</button>
              </div>
            );
          })}
        </div>
      </section>

      {editingId != null && (
        <div className="modal-backdrop" role="presentation">
          <section className="modal-card" role="dialog" aria-modal="true" aria-label="Edit custom phrase">
            <div className="panel-head">
              <div><div className="eyebrow">CUSTOM PHRASE</div><h2>Edit phrase</h2></div>
            </div>
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
