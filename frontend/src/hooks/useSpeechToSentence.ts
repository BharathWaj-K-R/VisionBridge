import { useCallback, useEffect, useRef, useState, type MutableRefObject } from "react";
import { WebSpeechEngine } from "../speech/WebSpeechEngine";
import type { SpeechEngine } from "../speech/SpeechEngine";
import type { SpeechEngineState } from "../speech/speechTypes";
import { SentenceBuffer } from "../sentence/sentenceBuffer";
import { appendReconciledText, reconcileInterimText } from "../sentence/transcriptReconciler";
import { normalizeSentence } from "../sentence/normalizeSentence";
import { resolveSentence } from "../signing/signResolver";
import { durationForToken } from "../signing/signPlayback";
import type { ResolvedSentence } from "../signing/signTypes";
import type { VocabularyItem } from "../data/vocabulary";

const SENTENCE_PAUSE_MS = 1400;

function createSentenceId(counter: MutableRefObject<number>): string {
  counter.current += 1;
  return "sentence-" + Date.now().toString(36) + "-" + counter.current.toString(36);
}

type PlaybackState = "idle" | "playing" | "paused" | "complete";

export type UseSpeechToSentenceResult = {
  transcript: string;
  interimTranscript: string;
  sentences: ResolvedSentence[];
  currentSentenceIndex: number;
  currentTokenIndex: number;
  currentSentence: ResolvedSentence | null;
  currentToken: ResolvedSentence["signSequence"][number] | null;
  listening: boolean;
  supported: boolean;
  speechState: SpeechEngineState;
  speechError: string;
  language: string;
  playback: PlaybackState;
  speed: number;
  setLanguage: (language: string) => void;
  setSpeed: (speed: number) => void;
  startListening: () => void;
  stopListening: () => void;
  endSentence: () => void;
  play: () => void;
  pause: () => void;
  repeat: () => void;
  nextSentence: () => void;
  previousSentence: () => void;
  selectSentence: (index: number) => void;
  clear: () => void;
};

export function useSpeechToSentence(
  vocabulary: VocabularyItem[],
  initialSpeed = 1,
): UseSpeechToSentenceResult {
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [sentences, setSentences] = useState<ResolvedSentence[]>([]);
  const [currentSentenceIndex, setCurrentSentenceIndex] = useState(-1);
  const [currentTokenIndex, setCurrentTokenIndex] = useState(0);
  const [listening, setListening] = useState(false);
  const [supported, setSupported] = useState(false);
  const [speechState, setSpeechState] = useState<SpeechEngineState>("idle");
  const [speechError, setSpeechError] = useState("");
  const [language, setLanguageState] = useState("en-IN");
  const [playback, setPlayback] = useState<PlaybackState>("idle");
  const [speed, setSpeed] = useState(initialSpeed || 1);

  const vocabularyRef = useRef(vocabulary);
  const transcriptRef = useRef("");
  const bufferRef = useRef(new SentenceBuffer());
  const engineRef = useRef<SpeechEngine | null>(null);
  const pauseTimerRef = useRef<number | null>(null);
  const sentenceCounterRef = useRef(0);
  const sentencesRef = useRef<ResolvedSentence[]>([]);
  const languageRef = useRef(language);
  const listeningRef = useRef(false);
  const speedRef = useRef(speed);

  vocabularyRef.current = vocabulary;
  transcriptRef.current = transcript;
  sentencesRef.current = sentences;
  languageRef.current = language;
  listeningRef.current = listening;
  speedRef.current = speed;

  useEffect(() => {
    setSpeed(initialSpeed || 1);
  }, [initialSpeed]);

  const addSentences = useCallback((texts: string[]) => {
    const resolved = texts
      .map((text) => normalizeSentence(text))
      .filter(Boolean)
      .map((text) => resolveSentence(
        {
          id: createSentenceId(sentenceCounterRef),
          text,
          createdAt: Date.now(),
          status: "ready",
        },
        vocabularyRef.current,
      ));

    if (!resolved.length) return;

    setSentences((current) => current.concat(resolved));
    setCurrentSentenceIndex((current) => (current < 0 ? 0 : current));
    setCurrentTokenIndex((current) => (current < 0 ? 0 : current));
    setPlayback((current) => (current === "complete" ? "idle" : current));
  }, []);

  const clearPauseTimer = useCallback(() => {
    if (pauseTimerRef.current != null) {
      window.clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = null;
    }
  }, []);

  const flushPendingSentence = useCallback(() => {
    clearPauseTimer();
    const pending = bufferRef.current.flush();
    if (pending) addSentences([pending]);
    setInterimTranscript("");
  }, [addSentences, clearPauseTimer]);

  const handleSpeechResult = useCallback((result: {
    finalText: string;
    interimText: string;
    confidence?: number;
  }) => {
    if (result.finalText) {
      const nextTranscript = appendReconciledText(transcriptRef.current, result.finalText);
      transcriptRef.current = nextTranscript;
      setTranscript(nextTranscript);

      const segmentation = bufferRef.current.append(result.finalText);
      if (segmentation.complete.length) addSentences(segmentation.complete);

      clearPauseTimer();
      pauseTimerRef.current = window.setTimeout(() => {
        const pending = bufferRef.current.flush();
        if (pending) addSentences([pending]);
        setInterimTranscript("");
        pauseTimerRef.current = null;
      }, SENTENCE_PAUSE_MS);
    }

    setInterimTranscript(reconcileInterimText(result.interimText));
  }, [addSentences, clearPauseTimer]);

  const handleSpeechState = useCallback((state: SpeechEngineState) => {
    setSpeechState(state);
    if (state === "listening") {
      setListening(true);
      listeningRef.current = true;
    } else if (state === "idle" || state === "stopping" || state === "error") {
      setListening(false);
      listeningRef.current = false;
    }
  }, []);

  const handleSpeechError = useCallback((message: string) => {
    setSpeechError(message);
    setListening(false);
    listeningRef.current = false;
  }, []);

  useEffect(() => {
    const engine = new WebSpeechEngine(
      {
        onResult: handleSpeechResult,
        onStateChange: handleSpeechState,
        onError: handleSpeechError,
      },
      languageRef.current,
    );
    engineRef.current = engine;
    setSupported(engine.supported);

    return () => {
      clearPauseTimer();
      engine.abort();
      engineRef.current = null;
    };
  }, [clearPauseTimer, handleSpeechError, handleSpeechResult, handleSpeechState]);

  const setLanguage = useCallback((nextLanguage: string) => {
    languageRef.current = nextLanguage;
    setLanguageState(nextLanguage);
    engineRef.current?.setLanguage(nextLanguage);
  }, []);

  const startListening = useCallback(() => {
    if (!engineRef.current) return;
    setSpeechError("");
    engineRef.current.start();
  }, []);

  const stopListening = useCallback(() => {
    engineRef.current?.stop();
    clearPauseTimer();
    pauseTimerRef.current = window.setTimeout(() => {
      const pending = bufferRef.current.flush();
      if (pending) addSentences([pending]);
      setInterimTranscript("");
      pauseTimerRef.current = null;
    }, 250);
  }, [addSentences, clearPauseTimer]);

  const endSentence = useCallback(() => {
    setInterimTranscript("");
    flushPendingSentence();
  }, [flushPendingSentence]);

  const updateSentenceStatuses = useCallback((
    list: ResolvedSentence[],
    activeIndex: number,
    isPlaying: boolean,
    completedIndex = -1,
  ): ResolvedSentence[] => list.map((sentence, index) => ({
    ...sentence,
    status:
      index <= completedIndex ? "completed"
        : index === activeIndex ? (isPlaying ? "playing" : "ready")
          : "ready",
  })), []);

  const play = useCallback(() => {
    if (!sentencesRef.current.length) return;

    const target = currentSentenceIndex >= 0
      ? currentSentenceIndex
      : 0;
    const sentence = sentencesRef.current[target];
    if (!sentence?.signSequence.length) return;

    setCurrentSentenceIndex(target);
    setCurrentTokenIndex((index) => Math.min(index, sentence.signSequence.length - 1));
    setSentences((list) => updateSentenceStatuses(list, target, true, target - 1));
    setPlayback("playing");
  }, [currentSentenceIndex, updateSentenceStatuses]);

  const pause = useCallback(() => {
    if (!sentencesRef.current.length) return;
    setSentences((list) => updateSentenceStatuses(list, currentSentenceIndex, false, currentSentenceIndex - 1));
    setPlayback("paused");
  }, [currentSentenceIndex, updateSentenceStatuses]);

  const repeat = useCallback(() => {
    const target = currentSentenceIndex >= 0 ? currentSentenceIndex : 0;
    if (!sentencesRef.current[target]) return;
    setCurrentSentenceIndex(target);
    setCurrentTokenIndex(0);
    setSentences((list) => updateSentenceStatuses(list, target, true, target - 1));
    setPlayback("playing");
  }, [currentSentenceIndex, updateSentenceStatuses]);

  const nextSentence = useCallback(() => {
    const next = Math.min(currentSentenceIndex + 1, sentencesRef.current.length - 1);
    if (next < 0 || next === currentSentenceIndex) return;
    setCurrentSentenceIndex(next);
    setCurrentTokenIndex(0);
    setSentences((list) => updateSentenceStatuses(list, next, playback === "playing", next - 1));
    setPlayback("playing");
  }, [currentSentenceIndex, playback, updateSentenceStatuses]);

  const previousSentence = useCallback(() => {
    const previous = Math.max(currentSentenceIndex - 1, 0);
    if (currentSentenceIndex < 0 || previous === currentSentenceIndex) return;
    setCurrentSentenceIndex(previous);
    setCurrentTokenIndex(0);
    setSentences((list) => updateSentenceStatuses(list, previous, playback === "playing", previous - 1));
    setPlayback("playing");
  }, [currentSentenceIndex, playback, updateSentenceStatuses]);

  const selectSentence = useCallback((index: number) => {
    if (index < 0 || index >= sentencesRef.current.length) return;
    setCurrentSentenceIndex(index);
    setCurrentTokenIndex(0);
    setSentences((list) => updateSentenceStatuses(list, index, false, index - 1));
    setPlayback("paused");
  }, [updateSentenceStatuses]);

  const clear = useCallback(() => {
    engineRef.current?.abort();
    clearPauseTimer();
    bufferRef.current.clear();
    transcriptRef.current = "";
    sentencesRef.current = [];
    setTranscript("");
    setInterimTranscript("");
    setSentences([]);
    setCurrentSentenceIndex(-1);
    setCurrentTokenIndex(0);
    setListening(false);
    setSpeechError("");
    setPlayback("idle");
  }, [clearPauseTimer]);

  useEffect(() => {
    if (!playingOrPaused(playback)) return;
    if (playback !== "playing") return;

    const sentence = sentences[currentSentenceIndex];
    const item = sentence?.signSequence[currentTokenIndex];

    if (!sentence) {
      setPlayback("idle");
      return;
    }

    if (!item) {
      if (currentSentenceIndex < sentences.length - 1) {
        const nextIndex = currentSentenceIndex + 1;
        setCurrentSentenceIndex(nextIndex);
        setCurrentTokenIndex(0);
        setSentences((list) => updateSentenceStatuses(list, nextIndex, true, currentSentenceIndex));
      } else {
        setSentences((list) => updateSentenceStatuses(list, currentSentenceIndex, false, currentSentenceIndex));
        setPlayback("complete");
      }
      return;
    }

    const timer = window.setTimeout(() => {
      if (currentTokenIndex + 1 < sentence.signSequence.length) {
        setCurrentTokenIndex((index) => index + 1);
      } else if (currentSentenceIndex + 1 < sentences.length) {
        const nextIndex = currentSentenceIndex + 1;
        setCurrentSentenceIndex(nextIndex);
        setCurrentTokenIndex(0);
        setSentences((list) => updateSentenceStatuses(list, nextIndex, true, currentSentenceIndex));
      } else {
        setSentences((list) => updateSentenceStatuses(list, currentSentenceIndex, false, currentSentenceIndex));
        setPlayback("complete");
      }
    }, durationForToken(item.kind, speed));

    return () => window.clearTimeout(timer);
  }, [
    playback,
    currentSentenceIndex,
    currentTokenIndex,
    sentences,
    speed,
    updateSentenceStatuses,
  ]);

  const currentSentence = sentences[currentSentenceIndex] || null;
  const currentToken = currentSentence?.signSequence[currentTokenIndex] || null;

  return {
    transcript,
    interimTranscript,
    sentences,
    currentSentenceIndex,
    currentTokenIndex,
    currentSentence,
    currentToken,
    listening,
    supported,
    speechState,
    speechError,
    language,
    playback,
    speed,
    setLanguage,
    setSpeed,
    startListening,
    stopListening,
    endSentence,
    play,
    pause,
    repeat,
    nextSentence,
    previousSentence,
    selectSentence,
    clear,
  };
}

function playingOrPaused(playback: PlaybackState): boolean {
  return playback === "playing" || playback === "paused";
}
