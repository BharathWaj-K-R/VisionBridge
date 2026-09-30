import { Link } from "react-router-dom";
import Seo from "./components/Seo";

export default function NotFound() {
  return <main className="not-found-page"><Seo /><section className="not-found-card" aria-labelledby="not-found-title">
    <div className="eyebrow">VISIONBRIDGE / 404</div><div className="not-found-mark" aria-hidden="true">404</div>
    <h1 id="not-found-title">Page not found</h1><p className="muted">The page you requested does not exist or has been moved.</p>
    <nav className="not-found-nav" aria-label="Application sections">
      <Link className="primary-btn" to="/dashboard">Dashboard</Link><Link className="ghost-btn" to="/translate">Live Translate</Link><Link className="ghost-btn" to="/calibration">Calibration</Link><Link className="ghost-btn" to="/history">History</Link><Link className="ghost-btn" to="/settings">Signer Profiles</Link>
    </nav>
  </section></main>;
}