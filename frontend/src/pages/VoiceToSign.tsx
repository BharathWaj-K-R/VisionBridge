import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { DEFAULT_VOCABULARY } from "../data/vocabulary";
import { buildSubtitleUnits } from "../data/signSequence";
import SpeechRecognizer from "../components/SpeechRecognizer";
import SignAvatar3D from "../components/SignAvatar3D";
import AvatarCustomizer, { DEFAULT_AVATAR } from "../components/AvatarCustomizer";
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
  const [view, setView] = useState<"full" | "close">("full");
  const [expression, setExpression] = useState<"neutral" | "question" | "emphasis">("neutral");
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
  const avatar = activeProfile?.config.avatar;
  const speed = activeProfile?.config.signingSpeed || 1;

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
                <div className="eyebrow">HOLOGRAPHIC AVATAR</div>
                <h2>3D signing viewport</h2>
              </div>
              <span className="status-pill">{playing ? "ANIMATING" : "STANDBY"}</span>
            </div>

            <div className="avatar-blueprint-strip">
              <span>BLUEPRINT 03</span>
              <strong>HOLOGRAPHIC GEOMETRIC ENVELOPE</strong>
              <small>Low-poly shell · neon joints · confidence field</small>
            </div>

            <div className="avatar-tool-row">
              <div className="avatar-segment">
                <span className="eyebrow">VIEW</span>
                <button type="button" className={view === "full" ? "active" : ""} onClick={() => setView("full")}>FULL BODY</button>
                <button type="button" className={view === "close" ? "active" : ""} onClick={() => setView("close")}>HAND + FACE</button>
              </div>
              <div className="avatar-segment">
                <span className="eyebrow">EXPRESSION</span>
                {(["neutral", "question", "emphasis"] as const).map((item) => (
                  <button type="button" key={item} className={expression === item ? "active" : ""} onClick={() => setExpression(item)}>{item.toUpperCase()}</button>
                ))}
              </div>
            </div>

            <SignAvatar3D
              letter={queue[currentIndex] || ""}
              preferences={avatar || DEFAULT_AVATAR}
              playing={playing}
              view={view}
              expression={expression}
            />

            <SignSubtitle units={subtitleUnits} currentIndex={currentIndex} />

            <p className="avatar-note">
              The 3D signer uses the existing letter-pose rig and renders it as a holographic volumetric avatar. Unknown words still fall back to letter-by-letter fingerspelling until validated word-level ISL motion assets are available.
            </p>

            {vocabularyMatches.length > 0 && (
              <div className="vocabulary-match-note">
                <span className="eyebrow">KNOWN VOCABULARY</span>
                <strong>{vocabularyMatches.join(" · ")}</strong>
              </div>
            )}
          </section>

          <AvatarCustomizer
            value={avatar || DEFAULT_AVATAR}
            onChange={(next) => void updateConfig({ avatar: next })}
            onResetView={() => setView("full")}
          />
        </div>
      </div>

      <section className="panel voice-limitation-panel">
        <div className="eyebrow">CURRENT SIGNING SCOPE</div>
        <p>Vocabulary recognition is phrase-aware for the subtitle layer. Unknown words fall back to letter-by-letter fingerspelling. Word-level motion remains an extensibility point rather than a claim of validated ISL animation accuracy.</p>
      </section>
    </Page>
  );
}
