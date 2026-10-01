import type { SpeechEngineCallbacks } from "./speechTypes";

export interface SpeechEngine {
  readonly supported: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  setLanguage(language: string): void;
}

export type SpeechEngineFactory = (
  callbacks: SpeechEngineCallbacks,
  language: string,
) => SpeechEngine;
