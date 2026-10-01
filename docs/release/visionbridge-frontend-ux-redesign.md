# VisionBridge Frontend UX Redesign

## 1. Main UX problems identified
- The primary shell exposed implementation details such as runtime, model, vector, and pipeline status on every page.
- Everyday actions competed with calibration and profile-management tasks in the main navigation.
- First-time guidance was not prominent enough before camera or microphone permission prompts.
- The camera screen looked like a diagnostic console instead of a communication tool.
- Speak-to-Sentence exposed implementation terminology alongside the main speaking and playback flow.
- Quick Access worked, but its presentation emphasized slot configuration more than immediate communication.
- Dense monospace labels, hard edges, and persistent telemetry increased cognitive load.

## 2. Information architecture
Primary navigation is now **Translate · Speak · Words · History · Settings**.
The VisionBridge logo returns to Home. Calibration and Profile remain available as secondary setup tools from Settings and Words. Quick Access remains globally available.

## 3. Concrete UI changes
- Added a calm accessible design layer without replacing the existing functional CSS foundation.
- Simplified the top bar and removed the persistent runtime strip.
- Added mobile bottom navigation with large touch targets.
- Added an action-first Home screen for Camera Translate and Speak.
- Added first-use guidance for permissions, lighting, framing, and optional personalization.
- Moved camera tracker/model details into expandable Advanced sections.
- Added a friendly inactive-camera state and visible permission status.
- Simplified Speak controls, transcript wording, sentence queue, and playback controls.
- Moved Speak implementation/scope information into an Advanced disclosure.
- Added persistent high-contrast support.
- Strengthened keyboard focus states and touch target sizing.
- Restyled Quick Access around everyday phrases.
- Kept all existing routes, backend calls, authentication, recognition logic, speech logic, sign assets, calibration, history, profiles, and Quick Access contracts.

## 4. Before / after
### Home
**Before:** Metrics and workstation status dominated the opening view.
**After:** The first decision is simply Camera or Speak, followed by three practical setup tips. Usage metrics remain available under Advanced.

### Translate
**Before:** Vector dimensions, model state, MediaPipe version, and other runtime telemetry appeared directly on the camera surface.
**After:** The camera tells the user where to place their hands, what to do when it is off, and how personalization is configured. Technical telemetry is expandable.

### Speak
**Before:** Speech input, sentence queue, implementation pipeline, and scope notes competed for attention.
**After:** Start speaking, review the sentence, and play/pause/repeat/navigate are the core flow. Implementation notes are secondary.

### Words
**Before:** The Word Bank read like a configuration console.
**After:** The same vocabulary, favorites, custom phrases, and ten Quick Access slots are framed as everyday communication tools.

### Settings
**Before:** Account IDs and implementation language competed with appearance and account controls.
**After:** Comfort, privacy, personalization, and browser permissions are the main concepts. Technical security details are disclosed only when requested.

## 5. Core functionality
The redesign intentionally leaves the existing feature implementations and backend contracts in place:
- Live Translate camera recognition
- Speak-to-Sentence
- Word Bank and custom words
- Ten-slot Quick Access
- Few-shot calibration
- Authentication
- History
- Sign reference and avatar systems
- Existing API/data contracts

Automated build/deploy validation is required before release. Real camera and microphone interaction still requires a browser with working device permissions.
