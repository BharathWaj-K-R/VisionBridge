import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { DEFAULT_VOCABULARY } from "../data/vocabulary";
import { buildSubtitleUnits } from "../data/signSequence";
import SpeechRecognizer from "../components/SpeechRecognizer";
import SignQueue, { normalizeSignText, toSignQueue } from "../components/SignQueue";
import SignSubtitle from "../components/SignSubtitle";
import { usePersonalization } from "../components/PersonalizationContext";
import { Page } from "../components/Page";

export default function VoiceToSign() {
  const { activeProfile, updateConfig } = usePersonalization();
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [speechError, setSpeechError] = useState("");
  const [listening, setListening] = useState(false);
  const [customWords, setCustomWords] = useState<Array<{ phrase: string; category: string }>>([]);
  const [queue, setQueue] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [lastPhrase, setLastPhrase] = useState("");
  const lastQueuedPhraseRef = useRef("");

  useEffect(() => {
    let mounted = true;
    api.customWords()
      .then((items) => {
        if (mounted) setCustomWords(items.map((item: any) => ({ phrase: String(item.phrase), category: String(item.category || "Custom") })));
      })
      .catch(() => undefined);
    return () => { mounted = false; };
  }, []);

  const normalizedTranscript = useMemo(() => normalizeSignText(transcript), [transcript]);
  const normalizedInterim = useMemo(() => normalizeSignText(interimTranscript), [interimTranscript]);
  const vocabulary = useMemo(
    () => [...DEFAULT_VOCABULARY, ...customWords.map((item, index) => ({
      id: "custom-" + index,
      phrase: item.phrase,
      category: item.category as any,
      custom: true,
    }))],
    [customWords],
  );
  const subtitleUnits = useMemo(() => buildSubtitleUnits(normalizedTranscript, vocabulary), [normalizedTranscript, vocabulary]);
  const vocabularyMatches = useMemo(() => subtitleUnits.filter((item) => item.kind === "word").map((item) => item.text), [subtitleUnits]);

  useEffect(() => {
    const previous = lastQueuedPhraseRef.current;
    if (!normalizedTranscript || normalizedTranscript === previous) return;

    if (previous && normalizedTranscript.startsWith(previous)) {
      const additions = toSignQueue(normalizedTranscript.slice(previous.length));
      if (additions.length) {
        setQueue((items) => items.concat(additions));
        setPlaying(true);
      }
    } else {
      const next = toSignQueue(normalizedTranscript);
      setQueue(next);
      setCurrentIndex(next.length ? 0 : -1);
      setPlaying(next.length > 0);
    }

    lastQueuedPhraseRef.current = normalizedTranscript;
    setLastPhrase(normalizedTranscript);
  }, [normalizedTranscript]);

  useEffect(() => {
    if (!playing || currentIndex < 0 || currentIndex >= queue.length) return;
    const timer = window.setTimeout(() => {
      if (currentIndex + 1 >= queue.length) {
        setPlaying(false);
        return;
      }
      setCurrentIndex((index) => index + 1);
    }, 900 / (activeProfile?.config.signingSpeed || 1));
    return () => window.clearTimeout(timer);
  }, [playing, currentIndex, queue.length, activeProfile?.config.signingSpeed]);

  const play = () => {
    if (!queue.length) return;
    if (currentIndex < 0 || currentIndex >= queue.length) setCurrentIndex(0);
    setPlaying(true);
  };
  const pause = () => setPlaying(false);
  const repeat = () => {
    const next = toSignQueue(lastPhrase || normalizedTranscript);
    setQueue(next);
    setCurrentIndex(next.length ? 0 : -1);
    setPlaying(next.length > 0);
  };
  const clearQueue = () => {
    setQueue([]);
    setCurrentIndex(-1);
    setPlaying(false);
  };

  const supported = Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  const speed = activeProfile?.config.signingSpeed || 1;
  const currentSign = queue[currentIndex] && /^[A-Z]$/.test(queue[currentIndex]) ? queue[currentIndex] : "";
  const currentSignIndex = currentSign ? currentSign.charCodeAt(0) - 65 : -1;
  const atlasColumn = currentSignIndex >= 0 ? currentSignIndex % 7 : 0;
  const atlasRow = currentSignIndex >= 0 ? Math.floor(currentSignIndex / 7) : 0;
  const atlasPosition = currentSignIndex >= 0
    ? (atlasColumn / 6) * 100 + "% " + (atlasRow / 3) * 100 + "%"
    : "50% 50%";

  useEffect(() => {
    const image = new Image();
    image.src = "/signs/sign-atlas.webp";
  }, []);

  return (
    <Page
      title="Voice to Sign"
      subtitle={"Speak naturally and VisionBridge turns your speech into synchronized signing for the " + (activeProfile?.name || "active") + " profile."}
    >
      <div className="voice-layout">
        <div className="voice-main">
          <SpeechRecognizer
            transcript={transcript}
            interimTranscript={normalizedInterim}
            listening={listening}
            supported={supported}
            error={speechError}
            onTranscriptChange={setTranscript}
            onInterimChange={setInterimTranscript}
            onListeningChange={setListening}
            onError={setSpeechError}
          />
          <SignQueue
            queue={queue}
            currentIndex={currentIndex}
            playing={playing}
            onPlay={play}
            onPause={pause}
            onRepeat={repeat}
            onClear={clearQueue}
            speed={speed}
            onSpeedChange={(value) => void updateConfig({ signingSpeed: value })}
          />
        </div>

        <div className="voice-side">
          <section className="panel avatar-panel">
            <div className="avatar-panel-head">
              <div>
                <div className="eyebrow">SIGN REFERENCE</div>
                <h2>A–Z signing sequence</h2>
              </div>
              <span className="status-pill">{playing ? "CYCLING" : "STANDBY"}</span>
            </div>

            <div className="sign-reference-meta">
              <span>UPLOADED A–Z REFERENCE</span>
              <strong>{currentSign || "—"}</strong>
              <small>{currentSignIndex >= 0 ? String(currentSignIndex + 1).padStart(2, "0") + " / 26" : "READY"}</small>
            </div>

            <div className="sign-reference-stage">
              <div
                className={"sign-reference-frame" + (currentSign ? " active" : "")}
                role="img"
                aria-label={currentSign ? "ISL reference image for letter " + currentSign : "ISL reference image waiting for speech"}
                style={{ backgroundPosition: atlasPosition }}
              />
              <div className="sign-reference-crosshair" />
              <div className="sign-reference-corner top-left">REFERENCE · 26 LETTERS</div>
              <div className="sign-reference-corner top-right">{playing ? "AUTO CYCLE" : "PAUSED"}</div>
              <div className="sign-reference-corner bottom-left">ENHANCED SOURCE · IMAGE ATLAS</div>
              <div className="sign-reference-corner bottom-right">{currentSign || "IDLE"}</div>
            </div>

            <SignSubtitle units={subtitleUnits} currentIndex={currentIndex} />

            <p className="avatar-note">
              The signing viewport cycles through the uploaded A–Z hand-reference images in queue order. Each source tile was cropped from the supplied sheet, enlarged and sharpened for clearer display. The images are visual references and do not claim additional word-level ISL motion validation.
            </p>

            {vocabularyMatches.length > 0 && (
              <div className="vocabulary-match-note">
                <span className="eyebrow">KNOWN VOCABULARY</span>
                <strong>{vocabularyMatches.join(" · ")}</strong>
              </div>
            )}
          </section>
        </div>
      </div>

      <section className="panel voice-limitation-panel">
        <div className="eyebrow">CURRENT SIGNING SCOPE</div>
        <p>Vocabulary recognition is phrase-aware for the subtitle layer. Unknown words fall back to letter-by-letter fingerspelling. Word-level motion remains an extensibility point rather than a claim of validated ISL animation accuracy.</p>
      </section>
    </Page>
  );
}
