import { Link } from "react-router-dom";
import Speakable from "./Speakable";
import { useQuickAccess } from "./QuickAccessContext";

export default function QuickAccessBar() {
  const { slots, loading } = useQuickAccess();
  const phrases = slots
    .map((phrase, index) => ({ phrase, index }))
    .filter((item): item is { phrase: string; index: number } => Boolean(item.phrase));

  return (
    <section className="quick-access-bar" aria-label="Quick access">
      <div className="quick-access-label">
        <span className="eyebrow">Everyday phrases</span>
        <strong>Quick access</strong>
      </div>

      {!loading && phrases.length === 0 && (
        <p className="quick-access-empty-copy">Keep frequently used phrases within one tap.</p>
      )}

      <div className="quick-access-slots">
        {phrases.map(({ phrase, index }) => (
          <Speakable
            key={index}
            text={phrase}
            className="quick-access-item"
            label={"Speak quick access slot " + (index + 1) + ": " + phrase}
          />
        ))}
        {!loading && phrases.length < 10 && (
          <Link
            to="/word-bank"
            className="quick-access-item empty"
            aria-label="+ Add phrase"
          >
            + Add phrase
          </Link>
        )}
      </div>

      <Link className="quick-access-config" to="/word-bank">Manage</Link>
    </section>
  );
}
