import { Link } from "react-router-dom";
import Speakable from "./Speakable";
import { QUICK_ACCESS_SLOTS, useQuickAccess } from "./QuickAccessContext";

export default function QuickAccessBar() {
  const { slots, loading } = useQuickAccess();

  return (
    <section className="quick-access-bar" aria-label="Quick access">
      <div className="quick-access-label">
        <span className="eyebrow">EVERYDAY PHRASES</span>
        <strong>Quick access</strong>
      </div>
      {!loading && slots.every((slot) => !slot) && <p className="quick-access-empty-copy">Add the phrases you use most. They will stay available from any page.</p>}
      <div className="quick-access-slots">
        {Array.from({ length: QUICK_ACCESS_SLOTS }, (_, index) => {
          const phrase = slots[index];
          return phrase
            ? <Speakable key={index} text={phrase} className="quick-access-item" label={"Speak quick access slot " + (index + 1) + ": " + phrase} />
            : <Link key={index} to="/word-bank" className="quick-access-item empty" aria-label={"Configure quick access slot " + (index + 1)} title={"Add a phrase to slot " + (index + 1)}>Add phrase</Link>;
        })}
      </div>
      <Link className="quick-access-config" to="/word-bank">Manage phrases</Link>
    </section>
  );
}
