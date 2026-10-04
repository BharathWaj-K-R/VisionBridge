import React, { type ReactNode } from "react";
import { Link } from "react-router-dom";

type ButtonProps = {
  onClick?: () => void;
  children: ReactNode;
};

function ActionButton({ onClick, children }: ButtonProps) {
  return <button type="button" className="primary-btn system-state-action" onClick={onClick}>{children}</button>;
}

export function AppBootstrap({ text = "Starting VisionBridge…" }: { text?: string }) {
  return (
    <main className="system-full-page bootstrap-page" role="status" aria-live="polite">
      <section className="bootstrap-card">
        <div className="system-logo" aria-hidden="true">V</div>
        <div className="eyebrow">VISIONBRIDGE</div>
        <h1>Preparing VisionBridge</h1>
        <p>{text}</p>
        <span className="loading-dots" aria-hidden="true"><i /><i /><i /></span>
      </section>
    </main>
  );
}

export function FullPageLoading({ title = "Loading", message = "Loading the requested view…" }: { title?: string; message?: string }) {
  return (
    <main className="system-full-page" role="status" aria-live="polite">
      <section className="system-state-card">
        <span className="system-spinner" aria-hidden="true" />
        <div>
          <div className="eyebrow">VISIONBRIDGE</div>
          <h1>{title}</h1>
          <p>{message}</p>
        </div>
      </section>
    </main>
  );
}

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="system-loading-state" role="status" aria-live="polite">
      <span className="system-spinner small" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

type EmptyStateProps = {
  title?: string;
  message: string;
  actionLabel?: string;
  to?: string;
  onAction?: () => void;
};

export function EmptyState({ title = "Nothing here yet", message, actionLabel, to, onAction }: EmptyStateProps) {
  return (
    <div className="system-state-inline empty-state" role="status">
      <div className="system-state-icon" aria-hidden="true">+</div>
      <div className="system-state-copy">
        <h3>{title}</h3>
        <p>{message}</p>
        {actionLabel && to ? <Link className="primary-btn system-state-action" to={to}>{actionLabel}</Link> : null}
        {actionLabel && !to && onAction ? <ActionButton onClick={onAction}>{actionLabel}</ActionButton> : null}
      </div>
    </div>
  );
}

type ErrorStateProps = {
  title?: string;
  message: string;
  actionLabel?: string;
  onRetry?: () => void;
  home?: boolean;
};

export function ErrorState({ title = "Something went wrong", message, actionLabel = "Try again", onRetry, home = false }: ErrorStateProps) {
  return (
    <div className="system-state-inline error-state" role="alert">
      <div className="system-state-icon error" aria-hidden="true">!</div>
      <div className="system-state-copy">
        <h3>{title}</h3>
        <p>{message}</p>
        <div className="system-state-actions">
          {onRetry ? <ActionButton onClick={onRetry}>{actionLabel}</ActionButton> : null}
          {home ? <Link className="ghost-btn system-state-action" to="/dashboard">Home</Link> : null}
        </div>
      </div>
    </div>
  );
}

type PermissionKind = "camera" | "microphone";

export function PermissionGuidance({
  kind,
  onRetry,
  compact = false,
}: {
  kind: PermissionKind;
  onRetry: () => void;
  compact?: boolean;
}) {
  const camera = kind === "camera";
  const device = camera ? "camera" : "microphone";
  return (
    <section className={compact ? "permission-card compact" : "permission-card"} aria-labelledby={"permission-" + kind}>
      <div className="system-state-icon" aria-hidden="true">{camera ? "⌾" : "◉"}</div>
      <div className="system-state-copy">
        <div className="eyebrow">{camera ? "CAMERA ACCESS" : "MICROPHONE ACCESS"}</div>
        <h3 id={"permission-" + kind}>{camera ? "Camera access required" : "Microphone access required"}</h3>
        <p>Allow browser access to the {device}. If access was denied, open this site's browser permissions, choose Allow, then try again.</p>
        <div className="system-state-actions">
          <ActionButton onClick={onRetry}>Try again</ActionButton>
          <span className="permission-hint">Browser site permissions → {device} → Allow</span>
        </div>
      </div>
    </section>
  );
}

export function OfflineBanner() {
  const [online, setOnline] = React.useState(() => navigator.onLine);
  React.useEffect(() => {
    const on = () => setOnline(true);
    const off = () => setOnline(false);
    window.addEventListener("online", on);
    window.addEventListener("offline", off);
    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", off);
    };
  }, []);
  if (online) return null;
  return (
    <div className="offline-banner" role="status" aria-live="polite">
      <span className="offline-dot" aria-hidden="true" />
      <div><strong>You're offline.</strong><span>Account data and server actions require a connection. The current page can remain open.</span></div>
    </div>
  );
}

export function SuccessMessage({ children }: { children: ReactNode }) {
  return <div className="system-success" role="status" aria-live="polite"><span aria-hidden="true">✓</span><span>{children}</span></div>;
}
