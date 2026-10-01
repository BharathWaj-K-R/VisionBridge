import React, { Component, type ErrorInfo, type ReactNode } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, useLocation } from "react-router-dom";
import App from "./App";
import NotFound from "./NotFound";
import { AppBootstrap, ErrorState } from "./components/SystemStates";
import "./styles.css";
import "./design-system.css";

type ErrorBoundaryProps = { children: ReactNode };
type ErrorBoundaryState = { error: Error | null };

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };
  static getDerivedStateFromError(error: Error): ErrorBoundaryState { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { if (import.meta.env.DEV) console.error("VisionBridge frontend runtime error", error, info); }
  render() {
    if (this.state.error) {
      return <main className="system-full-page" role="alert">
        <section className="system-state-card system-runtime-error">
          <ErrorState
            title="VisionBridge needs to reload"
            message="Something unexpected stopped this screen. Reload the app and try again. Your saved account data is not changed by this message."
            actionLabel="Reload VisionBridge"
            onRetry={() => window.location.reload()}
            home
          />
        </section>
      </main>;
    }
    return this.props.children;
  }
}

const knownPaths = new Set(["/", "/login", "/dashboard", "/translate", "/calibration", "/history", "/settings", "/voice-to-sign", "/word-bank", "/personalization"]);
function RouteGuard() {
  const { pathname } = useLocation();
  return knownPaths.has(pathname) ? <App /> : <NotFound />;
}

const savedTheme = localStorage.getItem("visionbridge_theme");
const savedContrast = localStorage.getItem("visionbridge_contrast") === "high" ? "high" : "normal";
const initialTheme = savedTheme === "dark" || savedTheme === "light" || savedTheme === "system" ? savedTheme : "system";
document.documentElement.dataset.theme = initialTheme;
document.documentElement.dataset.contrast = savedContrast;

const root = document.getElementById("root");
if (!root) {
  document.body.innerHTML = "<main style=\"padding:24px;font-family:system-ui\">VisionBridge could not find the application root.</main>";
} else {
  ReactDOM.createRoot(root).render(<ErrorBoundary><React.StrictMode><BrowserRouter><RouteGuard /></BrowserRouter></React.StrictMode></ErrorBoundary>);
}
