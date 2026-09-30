import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import Seo from "./Seo";

const labels: Record<string, string> = {
  "/dashboard": "Dashboard",
  "/translate": "Live Translate",
  "/calibration": "Few-Shot Calibration",
  "/history": "Letter History",
  "/settings": "Signer Profiles",
};

export function Breadcrumbs() {
  const location = useLocation();
  const current = labels[location.pathname] || "VisionBridge";
  if (location.pathname === "/") return null;
  return <nav className="breadcrumbs" aria-label="Breadcrumb"><ol>
    <li><Link to="/dashboard">Dashboard</Link></li>
    {current !== "Dashboard" && <li aria-current="page">{current}</li>}
  </ol></nav>;
}

export function Page({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return <div className="page"><Seo /><Breadcrumbs /><header className="page-header"><div>
    <div className="eyebrow">VISIONBRIDGE / WORKSTATION</div><h1>{title}</h1><p className="muted">{subtitle}</p>
  </div></header>{children}</div>;
}
export function Loading() { return <div className="loading" role="status" aria-live="polite">Loading workspace…</div>; }
export function Empty({ text }: { text: string }) { return <div className="empty">{text}</div>; }
export function Metric({ label, value, detail }: { label: string; value: string | number; detail: string }) {
  return <div className="metric"><span className="eyebrow">{label}</span><strong>{value}</strong><span className="muted">{detail}</span></div>;
}