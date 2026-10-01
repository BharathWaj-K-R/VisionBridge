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
          <div className="eyebrow">SPEECH INPUT</div>
          <h2>Speak to sentence</h2>
        </div>
        <span className={listening ? "status-chip dark" : "status-chip"}><i />{listening ? "LISTENING" : "IDLE"}</span>
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
          {listening ? "STOP LISTENING" : "START LISTENING"}
        </button>

        <button
          type="button"
          className="ghost-btn"
          onClick={onEndSentence}
          disabled={!transcript && !interimTranscript}
        >
          END SENTENCE
        </button>

        <label className="voice-language">
          LANGUAGE
          <select value={language} onChange={(event) => onLanguageChange(event.target.value)}>
            <option value="en-IN">English · India</option>
            <option value="en-US">English · US</option>
          </select>
        </label>
      </div>

      {!supported && (
        <div className="alert error" role="alert">
          Speech recognition is not available in this browser. Use a browser with Web Speech API support and microphone access.
        </div>
      )}

      {error && <div className="alert error" role="alert">{error}</div>}

      <div className="transcript-box" aria-live="polite">
        <span className="eyebrow">LIVE TRANSCRIPT</span>
        <p>{transcript || "Start speaking to build sentences."}</p>
        {interimTranscript && <span className="transcript-interim">{interimTranscript}</span>}
      </div>

      <div className="sentence-input-hint">
        <span>Automatic boundary</span>
        <strong>punctuation or ~1.4s pause</strong>
        <span>Manual boundary</span>
        <strong>END SENTENCE</strong>
      </div>
    </section>
  );
}
