import { useEffect, useRef, useState } from "react";

type Props = {
  text: string;
  className?: string;
  label?: string;
};

export default function Speakable({ text, className = "", label }: Props) {
  const [speaking, setSpeaking] = useState(false);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => () => {
    if (utteranceRef.current) utteranceRef.current.onend = null;
  }, []);

  const speak = () => {
    if (!("speechSynthesis" in window) || !text.trim()) return;

    window.speechSynthesis.cancel();
    setSpeaking(true);

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = text.trim().length === 1 ? 0.8 : 0.95;
    utterance.pitch = 1;
    utterance.lang = "en-IN";
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
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
      <span>{text}</span>
      <small aria-hidden="true">{speaking ? "●" : "◌"}</small>
    </button>
  );
}
