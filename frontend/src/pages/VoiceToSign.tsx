import { useEffect, useMemo, useRef, useState } from "react";
import { DEFAULT_VOCABULARY } from "../data/vocabulary";
import SpeechRecognizer from "../components/SpeechRecognizer";
import SignAvatar from "../components/SignAvatar";
import AvatarCustomizer, { DEFAULT_AVATAR, loadAvatarPreferences, type AvatarPreferences } from "../components/AvatarCustomizer";
import SignQueue, { normalizeSignText, toSignQueue } from "../components/SignQueue";
import { Page } from "../components/Page";

export default function VoiceToSign() {
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [speechError, setSpeechError] = useState("");
  const [listening, setListening] = useState(false);
  const [avatar, setAvatar] = useState<AvatarPreferences>(() => loadAvatarPreferences());
  const [queue, setQueue] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(Number(localStorage.getItem("visionbridge_sign_speed") || "1"));
  const [lastPhrase, setLastPhrase] = useState("");
  const lastQueuedPhraseRef = useRef("");

  const normalizedTranscript = useMemo(() => normalizeSignText(transcript), [transcript]);
  const normalizedInterim = useMemo(() => normalizeSignText(interimTranscript), [interimTranscript]);
  const vocabularyMatches = useMemo(() => normalizedTranscript.split(" ").filter(Boolean).filter((word) => DEFAULT_VOCABULARY.some((item) => item.phrase.toUpperCase() === word)), [normalizedTranscript]);

  useEffect(() => {
    const previous = lastQueuedPhraseRef.current;
    if (!normalizedTranscript || normalizedTranscript === previous) return;

    if (previous && normalizedTranscript.startsWith(previous)) {
      const delta = normalizedTranscript.slice(previous.length);
      const additions = toSignQueue(delta);
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
    localStorage.setItem("visionbridge_sign_speed", String(speed));
  }, [speed]);

  useEffect(() => {
    if (!playing || currentIndex < 0 || currentIndex >= queue.length) return;
    const timer = window.setTimeout(() => {
      if (currentIndex + 1 >= queue.length) {
        setPlaying(false);
        return;
      }
      setCurrentIndex((index) => index + 1);
    }, 900 / speed);
    return () => window.clearTimeout(timer);
  }, [playing, currentIndex, queue.length, speed]);

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

  return (
    <Page title="Voice to Sign" subtitle="Speak naturally and VisionBridge fingerspells your phrase with a lightweight animated avatar.">
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
            onSpeedChange={setSpeed}
          />
        </div>

        <div className="voice-side">
          <section className="panel avatar-panel">
            <div className="avatar-panel-head">
              <div>
                <div className="eyebrow">ANIMATED AVATAR</div>
                <h2>ISL fingerspelling</h2>
              </div>
              <span className="status-pill">{playing ? "ANIMATING" : "STANDBY"}</span>
            </div>
            <SignAvatar letter={queue[currentIndex] || ""} preferences={avatar || DEFAULT_AVATAR} playing={playing} />
            <p className="avatar-note">Each queued character drives a distinct arm and hand motion. Vocabulary matches are identified, but the current avatar still fingerspells them because no validated word-level ISL motion library is bundled.</p>
            {vocabularyMatches.length > 0 && <div className="vocabulary-match-note"><span className="eyebrow">VOCABULARY MATCH</span><strong>{vocabularyMatches.join(" · ")}</strong></div>}
          </section>

          <AvatarCustomizer value={avatar || DEFAULT_AVATAR} onChange={setAvatar} />
        </div>
      </div>

      <section className="panel voice-limitation-panel">
        <div className="eyebrow">CURRENT SCOPE</div>
        <p>Voice-to-Sign currently converts recognized speech into normalized A–Z characters and spaces. It does not yet use a validated word-level ISL animation library, so names, numbers, punctuation, and nuanced sentence-level signs are simplified or omitted.</p>
      </section>
    </Page>
  );
}
