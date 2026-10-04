import { PermissionGuidance } from "./SystemStates";

type Props = {
  transcript: string;
  interimTranscript: string;
  listening: boolean;
  supported: boolean;
  error: string;
  language: string;
  onLanguageChange: (language: string) => void;
  onStart: () => void;
  onStop: () => void;
  onEndSentence: () => void;
};

export default function SpeechRecognizer({
  transcript,
  interimTranscript,
  listening,
  supported,
  error,
  language,
  onLanguageChange,
  onStart,
  onStop,
  onEndSentence,
}: Props) {
  return (
    <section className="panel voice-input-panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">Speak</div>
          <h2>Speak a sentence</h2>
          <p className="assistive-copy">Start listening, allow microphone access, and speak at your normal pace. The transcript appears below.</p>
        </div>
        <span className={listening ? "status-chip dark" : "status-chip"}><i />{listening ? "Listening" : "Ready"}</span>
      </div>

      <div className="voice-control-row sentence-speech-controls">
        <button
          type="button"
          className={listening ? "voice-mic-btn active" : "voice-mic-btn"}
          onClick={listening ? onStop : onStart}
          disabled={!supported}
          aria-pressed={listening}
        >
          <span aria-hidden="true">{listening ? "■" : "●"}</span>
          {listening ? "Stop listening" : "Start listening"}
        </button>

        <button
          type="button"
          className="ghost-btn"
          onClick={onEndSentence}
          disabled={!transcript && !interimTranscript}
        >
          Finish sentence
        </button>

        <label className="voice-language">
          Language
          <select value={language} onChange={(event) => onLanguageChange(event.target.value)}>
            <option value="en-IN">English · India</option>
            <option value="en-US">English · US</option>
          </select>
        </label>
      </div>

      {!supported && (
        <div className="alert error" role="alert">
          Voice input is not available in this browser. Use a supported browser with microphone access.
        </div>
      )}

      {error && (/microphone.*denied|permission was denied|permission.*denied/i.test(error)
        ? <PermissionGuidance kind="microphone" onRetry={onStart} compact />
        : <div className="alert error" role="alert">{error}</div>)}

      <div className="transcript-box" aria-live="polite">
        <span className="eyebrow">Transcript</span>
        <p>{transcript || "Your transcript will appear here."}</p>
        {interimTranscript && <span className="transcript-interim">{interimTranscript}</span>}
      </div>

      <div className="sentence-input-hint">
        <span>A sentence ends when you pause or use punctuation.</span>
        <strong>Use “Finish sentence” to end it manually.</strong>
      </div>
    </section>
  );
}
