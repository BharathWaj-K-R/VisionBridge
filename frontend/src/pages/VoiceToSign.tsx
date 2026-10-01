import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { DEFAULT_VOCABULARY } from "../data/vocabulary";
import SpeechRecognizer from "../components/SpeechRecognizer";
import SentencePlayer from "../components/SentencePlayer";
import SignReferenceStage from "../components/SignReferenceStage";
import { usePersonalization } from "../components/PersonalizationContext";
import { useSpeechToSentence } from "../hooks/useSpeechToSentence";
import { Page } from "../components/Page";

export default function VoiceToSign() {
  const { activeProfile, updateConfig } = usePersonalization();
  const [customWords, setCustomWords] = useState<Array<{ phrase: string; category: string }>>([]);

  useEffect(() => {
    let mounted = true;
    api.customWords()
      .then((items) => {
        if (mounted) {
          setCustomWords(
            items.map((item: any) => ({
              phrase: String(item.phrase),
              category: String(item.category || "Custom"),
            })),
          );
        }
      })
      .catch(() => undefined);

    return () => {
      mounted = false;
    };
  }, []);

  const vocabulary = useMemo(
    () => [
      ...DEFAULT_VOCABULARY,
      ...customWords.map((item, index) => ({
        id: "custom-" + index,
        phrase: item.phrase,
        category: item.category as any,
        custom: true,
      })),
    ],
    [customWords],
  );

  const profileSpeed = activeProfile?.config.signingSpeed || 1;
  const speech = useSpeechToSentence(vocabulary, profileSpeed);

  const handleSpeedChange = (value: number) => {
    speech.setSpeed(value);
    void updateConfig({ signingSpeed: value });
  };

  return (
    <Page
      title="Speak to Sentence"
      subtitle="Speak naturally. Your sentence appears as a sequence of sign references you can play, pause, and repeat."
    >
      <div className="voice-layout speak-to-sentence-layout">
        <div className="voice-main">
          <SpeechRecognizer
            transcript={speech.transcript}
            interimTranscript={speech.interimTranscript}
            listening={speech.listening}
            supported={speech.supported}
            error={speech.speechError}
            language={speech.language}
            onLanguageChange={speech.setLanguage}
            onStart={speech.startListening}
            onStop={speech.stopListening}
            onEndSentence={speech.endSentence}
          />

          <SentencePlayer
            sentences={speech.sentences}
            currentSentenceIndex={speech.currentSentenceIndex}
            currentTokenIndex={speech.currentTokenIndex}
            playing={speech.playback === "playing"}
            speed={speech.speed}
            onPlay={speech.play}
            onPause={speech.pause}
            onRepeat={speech.repeat}
            onNext={speech.nextSentence}
            onPrevious={speech.previousSentence}
            onSelectSentence={speech.selectSentence}
            onClear={speech.clear}
            onSpeedChange={handleSpeedChange}
          />
        </div>

        <div className="voice-side">
          <section className="panel avatar-panel sentence-stage-panel">
            <div className="avatar-panel-head">
              <div>
                <div className="eyebrow">SIGN REFERENCE</div>
                <h2>Sentence signing stage</h2>
              </div>
              <span className="status-pill">
                {speech.playback === "playing"
                  ? "PLAYING"
                  : speech.playback === "complete"
                    ? "COMPLETE"
                    : "STANDBY"}
              </span>
            </div>

            <div className="sentence-stage-meta">
              <div>
                <span className="eyebrow">CURRENT SENTENCE</span>
                <strong>{speech.currentSentence?.text || "Speak to build a sentence."}</strong>
              </div>
              <div className="sentence-stage-count">
                <span>{speech.currentSentenceIndex >= 0 ? String(speech.currentSentenceIndex + 1).padStart(2, "0") : "--"}</span>
                <small>{speech.sentences.length ? "/ " + String(speech.sentences.length).padStart(2, "0") : "/ --"}</small>
              </div>
            </div>

            <SignReferenceStage
              item={speech.currentToken}
              sentenceText={speech.currentSentence?.text || ""}
              sentenceIndex={speech.currentSentenceIndex}
              tokenIndex={speech.currentTokenIndex}
            />

            <div className="sentence-stage-foot">
              <div>
                <span className="eyebrow">RESOLUTION</span>
                <strong>
                  {speech.currentToken
                    ? speech.currentToken.kind === "letter"
                      ? "A–Z FALLBACK"
                      : speech.currentToken.kind.toUpperCase()
                    : "WAITING"}
                </strong>
              </div>
              <div>
                <span className="eyebrow">SPEECH ENGINE</span>
                <strong>WEB SPEECH · {speech.language.toUpperCase()}</strong>
              </div>
            </div>

            <p className="avatar-note">
              The visual stage uses the uploaded A–Z reference atlas for letter-level fallback. Word and phrase sign assets are only used when validated assets are explicitly added to the signing asset registry.
            </p>
          </section>
        </div>
      </div>

      <details className="panel advanced-details">
        <summary>About how Speak works</summary>
        <div className="assistive-copy">Speak-to-Sentence groups your speech into sentences, uses saved vocabulary when available, and uses letter-by-letter references when a matching visual asset is not available.</div>
      </details>
    </Page>
  );
}
