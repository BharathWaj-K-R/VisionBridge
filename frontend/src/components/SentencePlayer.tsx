import type { ResolvedSentence } from "../signing/signTypes";

type Props = {
  sentences: ResolvedSentence[];
  currentSentenceIndex: number;
  currentTokenIndex: number;
  playing: boolean;
  speed: number;
  onPlay: () => void;
  onPause: () => void;
  onRepeat: () => void;
  onNext: () => void;
  onPrevious: () => void;
  onSelectSentence: (index: number) => void;
  onClear: () => void;
  onSpeedChange: (value: number) => void;
};

function sentenceState(
  index: number,
  currentSentenceIndex: number,
  currentTokenIndex: number,
  currentTokenCount: number,
  playing: boolean,
): "completed" | "playing" | "waiting" {
  if (index < currentSentenceIndex) return "completed";
  if (index > currentSentenceIndex) return "waiting";
  if (!playing && currentTokenIndex >= Math.max(0, currentTokenCount - 1)) return "playing";
  return "playing";
}

export default function SentencePlayer({
  sentences,
  currentSentenceIndex,
  currentTokenIndex,
  playing,
  speed,
  onPlay,
  onPause,
  onRepeat,
  onNext,
  onPrevious,
  onSelectSentence,
  onClear,
  onSpeedChange,
}: Props) {
  const currentSentence = sentences[currentSentenceIndex] || null;

  return (
    <section className="panel sentence-player-panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">SENTENCE QUEUE</div>
          <h2>Communication sequence</h2>
        </div>
        <span className="mono">{sentences.length} SENTENCE{sentences.length === 1 ? "" : "S"}</span>
      </div>

      <div className="sentence-queue" aria-live="polite">
        {sentences.length ? sentences.map((sentence, index) => {
          const state = sentenceState(
            index,
            currentSentenceIndex,
            currentTokenIndex,
            currentSentence?.signSequence.length || 0,
            playing,
          );
          return (
            <button
              type="button"
              key={sentence.id}
              className={"sentence-queue-item " + state + (index === currentSentenceIndex ? " active" : "")}
              onClick={() => {
                onSelectSentence(index);
              }}
              title={sentence.text}
            >
              <span className="sentence-queue-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="sentence-queue-text">{sentence.text}</span>
              <span className="sentence-queue-state">{state.toUpperCase()}</span>
            </button>
          );
        }) : (
          <div className="empty">Speak a complete sentence, then it will appear here.</div>
        )}
      </div>

      {currentSentence && (
        <div className="sentence-token-strip">
          <span className="eyebrow">CURRENT SENTENCE</span>
          <strong>{currentSentence.text}</strong>
          <div className="sentence-token-list">
            {currentSentence.signSequence.map((item, index) => (
              <span
                key={item.id}
                className={
                  "sentence-token " +
                  (index === currentTokenIndex ? "current " : "") +
                  (index < currentTokenIndex ? "done" : "")
                }
              >
                {item.sourceText}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="sentence-player-controls">
        {!playing ? (
          <button type="button" className="primary-btn" onClick={onPlay} disabled={!sentences.length}>PLAY</button>
        ) : (
          <button type="button" className="primary-btn" onClick={onPause}>PAUSE</button>
        )}
        <button type="button" className="ghost-btn" onClick={onPrevious} disabled={currentSentenceIndex <= 0}>PREVIOUS</button>
        <button type="button" className="ghost-btn" onClick={onNext} disabled={currentSentenceIndex < 0 || currentSentenceIndex >= sentences.length - 1}>NEXT</button>
        <button type="button" className="ghost-btn" onClick={onRepeat} disabled={!sentences.length}>REPEAT</button>
        <button type="button" className="ghost-btn" onClick={onClear} disabled={!sentences.length}>CLEAR</button>
        <label className="speed-control">SPEED
          <select value={speed} onChange={(event) => onSpeedChange(Number(event.target.value))}>
            <option value="0.5">0.5×</option>
            <option value="0.75">0.75×</option>
            <option value="1">1×</option>
            <option value="1.25">1.25×</option>
          </select>
        </label>
      </div>
    </section>
  );
}
