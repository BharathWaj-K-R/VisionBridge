export function normalizeSignText(input: string): string {
  return input
    .toUpperCase()
    .replace(/[^A-Z ]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function toSignQueue(input: string): string[] {
  const normalized = normalizeSignText(input);
  return normalized ? normalized.split("") : [];
}

export default function SignQueue({
  queue,
  currentIndex,
  playing,
  onPlay,
  onPause,
  onRepeat,
  onClear,
  speed,
  onSpeedChange,
}: {
  queue: string[];
  currentIndex: number;
  playing: boolean;
  onPlay: () => void;
  onPause: () => void;
  onRepeat: () => void;
  onClear: () => void;
  speed: number;
  onSpeedChange: (value: number) => void;
}) {
  const current = queue[currentIndex];

  return (
    <section className="panel sign-queue-panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">SIGN QUEUE</div>
          <h2>Fingerspelling sequence</h2>
        </div>
        <span className="mono">{queue.length} TOKENS</span>
      </div>

      <div className="queue-strip" aria-live="polite">
        {queue.length ? queue.map((item, index) => (
          <span key={item + "-" + index} className={index === currentIndex ? "queue-token current" : index < currentIndex ? "queue-token done" : "queue-token"}>
            {item === " " ? "·" : item}
          </span>
        )) : <span className="empty">Speak a phrase, then the letters will be queued here.</span>}
      </div>

      <div className="queue-controls">
        {!playing ? (
          <button type="button" className="primary-btn" onClick={onPlay} disabled={!queue.length || currentIndex >= queue.length}>PLAY</button>
        ) : (
          <button type="button" className="primary-btn" onClick={onPause}>PAUSE</button>
        )}
        <button type="button" className="ghost-btn" onClick={onRepeat} disabled={!queue.length}>REPEAT</button>
        <button type="button" className="ghost-btn" onClick={onClear} disabled={!queue.length}>CLEAR</button>
        <label className="speed-control">SPEED
          <select value={speed} onChange={(event) => onSpeedChange(Number(event.target.value))}>
            <option value="0.75">0.75×</option>
            <option value="1">1×</option>
            <option value="1.25">1.25×</option>
            <option value="1.5">1.5×</option>
          </select>
        </label>
      </div>

      <div className="current-sign-banner">
        <span className="eyebrow">ACTIVE LETTER</span>
        <strong>{current ? (current === " " ? "SPACE" : current) : "READY"}</strong>
      </div>
    </section>
  );
}
