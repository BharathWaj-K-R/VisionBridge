import type { SpeechEngine } from "./SpeechEngine";
import type { SpeechEngineCallbacks, SpeechResult, SpeechEngineState } from "./speechTypes";

type NativeSpeechResult = {
  isFinal: boolean;
  length: number;
  [index: number]: { transcript?: string; confidence?: number };
};

type NativeSpeechEvent = Event & {
  resultIndex: number;
  results: {
    length: number;
    [index: number]: NativeSpeechResult;
  };
};

type NativeSpeechErrorEvent = Event & {
  error?: string;
};

type NativeRecognition = {
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  lang: string;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onresult: ((event: NativeSpeechEvent) => void) | null;
  onerror: ((event: NativeSpeechErrorEvent) => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type NativeRecognitionConstructor = new () => NativeRecognition;

declare global {
  interface Window {
    SpeechRecognition?: NativeRecognitionConstructor;
    webkitSpeechRecognition?: NativeRecognitionConstructor;
  }
}

function getConstructor(): NativeRecognitionConstructor | null {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function readableSpeechError(code: string): string {
  switch (code) {
    case "not-allowed":
    case "service-not-allowed":
      return "Microphone or speech recognition permission was denied. Allow access and try again.";
    case "audio-capture":
      return "No microphone input is available. Check your microphone and browser permissions.";
    case "network":
      return "The browser speech service could not be reached. Check your connection and try again.";
    case "no-speech":
      return "No speech was detected. Try speaking a little closer to the microphone.";
    default:
      return "Speech recognition error: " + code + ".";
  }
}

export class WebSpeechEngine implements SpeechEngine {
  private recognition: NativeRecognition | null = null;
  private callbacks: SpeechEngineCallbacks;
  private language: string;
  private requestedStop = false;
  private restartTimer: number | null = null;
  private state: SpeechEngineState = "idle";

  readonly supported: boolean;

  constructor(callbacks: SpeechEngineCallbacks, language: string) {
    this.callbacks = callbacks;
    this.language = language;
    this.supported = Boolean(getConstructor());
    if (this.supported) this.createRecognition();
  }

  private emitState(state: SpeechEngineState): void {
    this.state = state;
    this.callbacks.onStateChange(state);
  }

  private createRecognition(): void {
    const Constructor = getConstructor();
    if (!Constructor) return;

    const recognition = new Constructor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.lang = this.language;

    recognition.onstart = () => {
      this.emitState("listening");
    };

    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";
      let confidence: number | undefined;

      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const value = result[0]?.transcript || "";
        if (result.isFinal) {
          finalText += value + " ";
          if (typeof result[0]?.confidence === "number") confidence = result[0].confidence;
        } else {
          interimText += value;
        }
      }

      const payload: SpeechResult = {
        finalText: finalText.replace(/\s+/g, " ").trim(),
        interimText: interimText.replace(/\s+/g, " ").trim(),
        confidence,
      };

      if (payload.finalText || payload.interimText) {
        this.callbacks.onResult(payload);
      }
    };

    recognition.onerror = (event) => {
      const code = event.error || "unknown";
      if (code === "aborted") return;
      this.emitState("error");
      this.callbacks.onError(readableSpeechError(code));
    };

    recognition.onend = () => {
      this.emitState("idle");
      if (this.requestedStop || !this.recognition) return;

      this.restartTimer = window.setTimeout(() => {
        if (this.requestedStop || !this.recognition) return;
        try {
          this.emitState("starting");
          this.recognition.start();
        } catch {
          // Native recognition can reject an immediate restart; the next end event can retry.
        }
      }, 250);
    };

    this.recognition = recognition;
  }

  start(): void {
    if (!this.supported || !this.recognition) {
      this.callbacks.onError("Speech recognition is not available in this browser.");
      return;
    }

    this.requestedStop = false;
    if (this.restartTimer != null) {
      window.clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }

    try {
      this.emitState("starting");
      this.recognition.start();
    } catch {
      this.callbacks.onError("Speech recognition could not start. Check microphone permission and try again.");
      this.emitState("error");
    }
  }

  stop(): void {
    this.requestedStop = true;
    if (this.restartTimer != null) {
      window.clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    if (!this.recognition) return;

    try {
      this.emitState("stopping");
      this.recognition.stop();
    } catch {
      this.emitState("idle");
    }
  }

  abort(): void {
    this.requestedStop = true;
    if (this.restartTimer != null) {
      window.clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    try {
      this.recognition?.abort();
    } catch {
      // The native object may already have ended.
    }
    this.emitState("idle");
  }

  setLanguage(language: string): void {
    this.language = language;
    if (this.recognition) this.recognition.lang = language;
  }

  updateCallbacks(callbacks: SpeechEngineCallbacks): void {
    this.callbacks = callbacks;
  }
}
