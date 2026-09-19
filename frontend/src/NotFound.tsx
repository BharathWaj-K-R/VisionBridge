import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <main className="auth-page">
      <section className="auth-card">
        <div className="eyebrow">VISIONBRIDGE</div>
        <div className="not-found-code" aria-hidden="true">404</div>
        <h1>Page not found</h1>
        <p className="muted">The page you requested does not exist or has been moved.</p>
        <div className="button-row">
          <Link className="primary-btn" to="/">Go home</Link>
          <button className="ghost-btn" onClick={() => window.history.back()}>Go back</button>
        </div>
      </section>
    </main>
  );
}
