import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <main className="not-found-page">
      <section className="not-found-card">
        <div className="not-found-mark" aria-hidden="true">404</div>
        <div className="eyebrow">VISIONBRIDGE</div>
        <h1>Page not found</h1>
        <p className="muted">The page you requested does not exist, was moved, or took a wrong turn through the routing system.</p>
        <div className="button-row">
          <Link className="primary-btn" to="/">Go home</Link>
          <button className="ghost-btn" onClick={() => window.history.back()}>Go back</button>
        </div>
      </section>
    </main>
  );
}
