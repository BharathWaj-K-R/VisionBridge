import { useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "../api";
import { usePersonalization } from "./PersonalizationContext";

type Props = {
  text: string;
  className?: string;
  label?: string;
  children?: ReactNode;
};

export default function Speakable({ text, className = "", label, children }: Props) {
  const [speaking, setSpeaking] = useState(false);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const { activeProfile, voices } = usePersonalization();

  useEffect(() => () => {
    if (utteranceRef.current) {
      utteranceRef.current.onend = null;
      utteranceRef.current.onerror = null;
    }
  }, []);

  const speak = () => {
    if (!("speechSynthesis" in window) || !text.trim()) return;

    window.speechSynthesis.cancel();
    setSpeaking(true);

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = text.trim().length === 1 ? 0.8 : 0.95;
    utterance.pitch = 1;
    utterance.lang = "en-IN";

    if (activeProfile?.config.ttsVoice) {
      const preferred = voices.find((voice) => voice.name === activeProfile.config.ttsVoice);
      if (preferred) utterance.voice = preferred;
    }

    const finish = () => setSpeaking(false);
    utterance.onend = finish;
    utterance.onerror = finish;
    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
    void api.recordUsage(text, activeProfile?.id).finally(() => window.dispatchEvent(new Event("visionbridge:usage-changed")));
  };

  return (
    <button
      type="button"
      className={(speaking ? "speakable speaking " : "speakable ") + className}
      onClick={speak}
      aria-label={label || "Speak " + text}
      aria-pressed={speaking}
      title={"Speak: " + text}
    >
      {children || <span>{text}</span>}
      <small aria-hidden="true">{speaking ? "●" : "◌"}</small>
    </button>
  );
}
