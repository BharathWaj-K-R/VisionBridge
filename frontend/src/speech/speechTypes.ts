export type SpeechResult = {
  finalText: string;
  interimText: string;
  confidence?: number;
};

export type SpeechEngineState =
  | "idle"
  | "starting"
  | "listening"
  | "stopping"
  | "error";

export type SpeechEngineCallbacks = {
  onResult: (result: SpeechResult) => void;
  onStateChange: (state: SpeechEngineState) => void;
  onError: (message: string) => void;
};
