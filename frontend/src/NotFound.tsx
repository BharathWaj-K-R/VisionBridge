import { Link } from "react-router-dom";
import Seo from "./components/Seo";

export default function NotFound() {
  return <main className="not-found-page"><Seo /><section className="not-found-card" aria-labelledby="not-found-title">
    <div className="eyebrow">VISIONBRIDGE</div><div className="not-found-mark" aria-hidden="true">404</div>
    <h1 id="not-found-title">Page not found</h1><p className="muted">The requested address does not match a VisionBridge page. Use one of the available sections below.</p>
    <nav className="not-found-nav" aria-label="Application sections">
      <Link className="primary-btn" to="/dashboard">Home</Link><Link className="ghost-btn" to="/translate">Translate</Link><Link className="ghost-btn" to="/voice-to-sign">Speak</Link><Link className="ghost-btn" to="/word-bank">Words</Link><Link className="ghost-btn" to="/history">History</Link>
    </nav>
  </section></main>;
}