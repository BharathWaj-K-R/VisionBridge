# VisionBridge Essential System States

Implemented as reusable frontend components in `frontend/src/components/SystemStates.tsx`.

## States
- App bootstrap: branded full-page "Starting VisionBridge…" state.
- Full-page loading: reusable branded spinner state.
- Inline loading: reusable spinner for data-fetching pages.
- Empty states: reusable title, guidance, and optional recovery action.
- Generic runtime error: human-readable recovery with reload.
- Data/network error: retry and optional Home actions on Home, History, Settings, and Word Bank.
- Camera permission guidance: actionable retry plus browser site-control instructions.
- Microphone permission guidance: actionable retry plus browser site-control instructions.
- Recognition model unavailable: retry state on Translate.
- Session expired: returns to sign-in with a friendly session-expired message.
- 404: friendly navigation to Home, Translate, Speak, Words, and History.
- Offline: reactive banner when the browser reports no network connection.
- Success feedback: lightweight status/confirmation treatment remains inline for existing save/update flows.

## Core UX rule
Every state explains what happened in plain language and either provides a direct recovery action or tells the user the next action.

## Accessibility
States use semantic status/alert roles, visible keyboard focus, large interaction targets, readable contrast, and mobile layouts. High-contrast mode continues to use the existing VisionBridge accessibility setting.
