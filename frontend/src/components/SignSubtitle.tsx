import Speakable from "./Speakable";

export default function SignSubtitle({
  queue,
  currentIndex,
}: {
  queue: string[];
  currentIndex: number;
}) {
  const phrase = queue.join("");
  if (!phrase) {
    return (
      <div className="sign-subtitle empty" aria-live="polite">
        Speak a phrase to see synchronized signing subtitles.
      </div>
    );
  }

  return (
    <div className="sign-subtitle" aria-live="polite" aria-label="Synchronized sign subtitles">
      {queue.map((item, index) => (
        <Speakable
          key={item + "-" + index}
          text={item === " " ? "space" : item}
          label={"Speak subtitle character " + (item === " " ? "space" : item)}
          className={index === currentIndex ? "subtitle-token active" : index < currentIndex ? "subtitle-token done" : "subtitle-token"}
        />
      ))}
    </div>
  );
}
