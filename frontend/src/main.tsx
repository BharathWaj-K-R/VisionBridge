import React, { Component, type ErrorInfo, type ReactNode } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, useLocation } from "react-router-dom";
import App from "./App";
import NotFound from "./NotFound";
import "./styles.css";

type ErrorBoundaryProps = { children: ReactNode };
type ErrorBoundaryState = { error: Error | null };

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };
  static getDerivedStateFromError(error: Error): ErrorBoundaryState { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error("VisionBridge frontend runtime error", error, info); }
  render() {
    if (this.state.error) {
      const message = this.state.error.message || "Unknown frontend error";
      return <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24, fontFamily: "Inter, system-ui, sans-serif", background: "#f6f6f4" }}><section style={{ width: "min(720px, 100%)", background: "#fff", border: "1px solid #d9d9d5", borderRadius: 14, padding: 24 }}><div style={{ fontSize: 11, fontWeight: 700, letterSpacing: ".14em", color: "#777" }}>VISIONBRIDGE</div><h1 style={{ margin: "8px 0", fontSize: 28 }}>Frontend failed to start</h1><p style={{ color: "#555", lineHeight: 1.6 }}>The application hit a runtime error before it could render.</p><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", background: "#f7f7f4", border: "1px solid #e5e5e0", borderRadius: 8, padding: 12 }}>{message}</pre><button onClick={() => window.location.reload()} style={{ border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 8, padding: "10px 16px", fontWeight: 600 }}>Reload</button></section></main>;
    }
    return this.props.children;
  }
}

const knownPaths = new Set(["/", "/login", "/dashboard", "/translate", "/calibration", "/history", "/evaluation", "/settings"]);
function RouteGuard() {
  const { pathname } = useLocation();
  return knownPaths.has(pathname) ? <App /> : <NotFound />;
}

const root = document.getElementById("root");
if (!root) {
  document.body.innerHTML = "<main style=\"padding:24px;font-family:system-ui\">VisionBridge could not find the application root.</main>";
} else {
  ReactDOM.createRoot(root).render(<ErrorBoundary><React.StrictMode><BrowserRouter><RouteGuard /></BrowserRouter></React.StrictMode></ErrorBoundary>);
}
