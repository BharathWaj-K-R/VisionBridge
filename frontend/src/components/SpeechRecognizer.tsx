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
          <div className="eyebrow">SPEAK</div>
          <h2>Speak naturally</h2>
          <p className="assistive-copy">Press start, allow microphone access, then speak at your normal pace. Your sentence will appear below.</p>
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
          Voice input is not available in this browser. Try a supported browser and make sure microphone access is allowed.
        </div>
      )}

      {error && <div className="alert error" role="alert">{error}</div>}

      <div className="transcript-box" aria-live="polite">
        <span className="eyebrow">YOUR WORDS</span>
        <p>{transcript || "Your words will appear here as you speak."}</p>
        {interimTranscript && <span className="transcript-interim">{interimTranscript}</span>}
      </div>

      <div className="sentence-input-hint">
        <span>Sentence ends when you pause or finish with punctuation.</span>
        <strong>Use “Finish sentence” any time.</strong>
      </div>
    </section>
  );
}
