import { useEffect, useRef, useState } from "react";

type SpeechLikeEvent = Event & {
  resultIndex: number;
  results: {
    length: number;
    [index: number]: {
      isFinal: boolean;
      length: number;
      [index: number]: { transcript: string };
    };
  };
};

type SpeechErrorEvent = Event & { error?: string };

type SpeechRecognizerInstance = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onresult: ((event: SpeechLikeEvent) => void) | null;
  onerror: ((event: SpeechErrorEvent) => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type SpeechRecognizerConstructor = new () => SpeechRecognizerInstance;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognizerConstructor;
    webkitSpeechRecognition?: SpeechRecognizerConstructor;
  }
}

type Props = {
  transcript: string;
  interimTranscript: string;
  listening: boolean;
  supported: boolean;
  error: string;
  onTranscriptChange: (value: string) => void;
  onInterimChange: (value: string) => void;
  onListeningChange: (value: boolean) => void;
  onError: (value: string) => void;
};

export default function SpeechRecognizer({
  transcript,
  interimTranscript,
  listening,
  supported,
  error,
  onTranscriptChange,
  onInterimChange,
  onListeningChange,
  onError,
}: Props) {
  const recognitionRef = useRef<SpeechRecognizerInstance | null>(null);
  const requestedStopRef = useRef(false);
  const restartTimerRef = useRef<number | null>(null);
  const [language, setLanguage] = useState("en-IN");

  useEffect(() => {
    const Constructor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Constructor) return;

    const recognition = new Constructor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = language;

    recognition.onstart = () => onListeningChange(true);

    recognition.onresult = (event) => {
      let nextFinal = transcript;
      let nextInterim = "";

      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const value = result[0]?.transcript || "";
        if (result.isFinal) {
          nextFinal += value + " ";
        } else {
          nextInterim += value;
        }
      }

      onTranscriptChange(nextFinal.replace(/\s+/g, " ").trim());
      onInterimChange(nextInterim.trim());
    };

    recognition.onerror = (event) => {
      const code = event.error || "unknown";
      if (code === "not-allowed" || code === "service-not-allowed") {
        requestedStopRef.current = true;
        onListeningChange(false);
        onError("Microphone or speech recognition permission was denied.");
        return;
      }
      if (code !== "aborted") onError("Speech recognition error: " + code + ".");
    };

    recognition.onend = () => {
      onListeningChange(false);
      if (!requestedStopRef.current) {
        restartTimerRef.current = window.setTimeout(() => {
          try {
            recognition.start();
          } catch {
            // Browser speech services can reject an immediate restart.
          }
        }, 250);
      }
    };

    recognitionRef.current = recognition;

    return () => {
      requestedStopRef.current = true;
      if (restartTimerRef.current != null) window.clearTimeout(restartTimerRef.current);
      recognition.abort();
      recognitionRef.current = null;
    };
  }, [language, onError, onInterimChange, onListeningChange, onTranscriptChange, transcript]);

  useEffect(() => {
    if (recognitionRef.current) recognitionRef.current.lang = language;
  }, [language]);

  const toggleListening = () => {
    if (!supported) {
      onError("This browser does not expose the Web Speech API. Try a Chromium-based browser.");
      return;
    }

    const recognition = recognitionRef.current;
    if (!recognition) return;

    if (listening) {
      requestedStopRef.current = true;
      recognition.stop();
      onListeningChange(false);
      return;
    }

    requestedStopRef.current = false;
    onError("");
    try {
      recognition.start();
    } catch {
      onListeningChange(true);
    }
  };

  return (
    <section className="panel voice-input-panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">SPEECH INPUT</div>
          <h2>Speak to VisionBridge</h2>
        </div>
        <span className={listening ? "status-chip dark" : "status-chip"}><i />{listening ? "LISTENING" : "IDLE"}</span>
      </div>

      <div className="voice-control-row">
        <button type="button" className={listening ? "voice-mic-btn active" : "voice-mic-btn"} onClick={toggleListening} aria-pressed={listening} disabled={!supported}>
          <span aria-hidden="true">{listening ? "■" : "●"}</span>
          {listening ? "STOP LISTENING" : "START LISTENING"}
        </button>
        <label className="voice-language">LANGUAGE
          <select value={language} onChange={(event) => setLanguage(event.target.value)}>
            <option value="en-IN">English · India</option>
            <option value="en-US">English · US</option>
          </select>
        </label>
      </div>

      {!supported && <div className="alert error">Speech recognition is not available in this browser. Use a browser with Web Speech API support and microphone access.</div>}
      {error && <div className="alert error" role="alert">{error}</div>}

      <div className="transcript-box" aria-live="polite">
        <span className="eyebrow">LIVE TRANSCRIPT</span>
        <p>{transcript || "Start speaking to build the sign queue."}</p>
        {interimTranscript && <span className="transcript-interim">{interimTranscript}</span>}
      </div>
    </section>
  );
}
