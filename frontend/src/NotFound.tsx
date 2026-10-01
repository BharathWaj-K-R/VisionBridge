import { Link } from "react-router-dom";
import Seo from "./components/Seo";

export default function NotFound() {
  return <main className="not-found-page"><Seo /><section className="not-found-card" aria-labelledby="not-found-title">
    <div className="eyebrow">VISIONBRIDGE</div><div className="not-found-mark" aria-hidden="true">404</div>
    <h1 id="not-found-title">We can't find that page</h1><p className="muted">It may have moved, or the address may be incorrect. You can return to a familiar part of VisionBridge.</p>
    <nav className="not-found-nav" aria-label="Application sections">
      <Link className="primary-btn" to="/dashboard">Go home</Link><Link className="ghost-btn" to="/translate">Translate</Link><Link className="ghost-btn" to="/voice-to-sign">Speak</Link><Link className="ghost-btn" to="/word-bank">Words</Link><Link className="ghost-btn" to="/history">History</Link>
    </nav>
  </section></main>;
}