import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import Seo from "./Seo";
import { EmptyState, LoadingState } from "./SystemStates";

const labels: Record<string, string> = {
  "/dashboard": "Home",
  "/translate": "Translate",
  "/calibration": "Personalize recognition",
  "/history": "History",
  "/settings": "Settings",
  "/voice-to-sign": "Speak",
  "/word-bank": "Words",
  "/personalization": "Profile",
};

export function Breadcrumbs() {
  const location = useLocation();
  const current = labels[location.pathname] || "VisionBridge";
  if (location.pathname === "/" || location.pathname === "/dashboard") return null;
  return <nav className="breadcrumbs" aria-label="Breadcrumb"><ol>
    {current === "Dashboard" ? <li aria-current="page">Dashboard</li> : <><li><Link to="/dashboard">Dashboard</Link></li><li aria-current="page">{current}</li></>}
  </ol></nav>;
}

export function Page({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return <div className="page"><Seo /><Breadcrumbs /><header className="page-header"><div>
    <div className="eyebrow">VISIONBRIDGE</div><h1>{title}</h1><p className="muted">{subtitle}</p>
  </div></header>{children}</div>;
}
export function Loading() { return <LoadingState label="Loading…" />; }
export function Empty({ text }: { text: string }) { return <EmptyState message={text} />; }
export function Metric({ label, value, detail }: { label: string; value: string | number; detail: string }) {
  return <div className="metric"><span className="eyebrow">{label}</span><strong>{value}</strong><span className="muted">{detail}</span></div>;
}