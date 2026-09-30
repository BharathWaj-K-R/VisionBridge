import Speakable from "./Speakable";

export type SubtitleUnit = {
  text: string;
  startIndex: number;
  endIndex: number;
  kind: "word" | "letter";
};

export default function SignSubtitle({
  units,
  currentIndex,
}: {
  units: SubtitleUnit[];
  currentIndex: number;
}) {
  if (!units.length) {
    return (
      <div className="sign-subtitle empty" aria-live="polite">
        Speak a phrase to see synchronized signing subtitles.
      </div>
    );
  }

  return (
    <div className="sign-subtitle" aria-live="polite" aria-label="Synchronized sign subtitles">
      {units.map((unit, index) => {
        const active = currentIndex >= unit.startIndex && currentIndex <= unit.endIndex;
        const done = currentIndex > unit.endIndex;
        return (
          <Speakable
            key={unit.text + "-" + index + "-" + unit.startIndex}
            text={unit.text}
            label={"Speak subtitle " + unit.text}
            className={
              active
                ? "subtitle-token active " + (unit.kind === "word" ? "word-unit" : "letter-unit")
                : done
                  ? "subtitle-token done " + (unit.kind === "word" ? "word-unit" : "letter-unit")
                  : "subtitle-token " + (unit.kind === "word" ? "word-unit" : "letter-unit")
            }
          />
        );
      })}
    </div>
  );
}
