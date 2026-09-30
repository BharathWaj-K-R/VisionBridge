import type { VocabularyItem } from "./vocabulary";
import type { SubtitleUnit } from "../components/SignSubtitle";

export type SignAnimationUnit = {
  phrase: string;
  kind: "word" | "fingerspell";
  animationKey: string | null;
};

function normalizedWords(text: string) {
  const matches = [...text.matchAll(/[A-Z]+/g)];
  return matches.map((match) => ({
    value: match[0],
    start: match.index ?? 0,
    end: (match.index ?? 0) + match[0].length - 1,
  }));
}

export function buildSubtitleUnits(
  normalizedText: string,
  vocabulary: VocabularyItem[],
): SubtitleUnit[] {
  const words = normalizedWords(normalizedText);
  const phrases = vocabulary
    .map((item) => item.phrase.toUpperCase().replace(/[^A-Z ]/g, "").replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .sort((a, b) => b.split(" ").length - a.split(" ").length || b.length - a.length);

  const units: SubtitleUnit[] = [];
  let wordIndex = 0;

  while (wordIndex < words.length) {
    let matchedPhrase: string | null = null;
    let matchedCount = 0;

    for (const phrase of phrases) {
      const parts = phrase.split(" ");
      if (wordIndex + parts.length > words.length) continue;
      let matches = true;
      for (let offset = 0; offset < parts.length; offset += 1) {
        if (words[wordIndex + offset].value !== parts[offset]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        matchedPhrase = phrase;
        matchedCount = parts.length;
        break;
      }
    }

    if (matchedPhrase && matchedCount) {
      units.push({
        text: matchedPhrase,
        startIndex: words[wordIndex].start,
        endIndex: words[wordIndex + matchedCount - 1].end,
        kind: "word",
      });
      wordIndex += matchedCount;
      continue;
    }

    const word = words[wordIndex];
    for (let cursor = word.start; cursor <= word.end; cursor += 1) {
      units.push({
        text: normalizedText[cursor],
        startIndex: cursor,
        endIndex: cursor,
        kind: "letter",
      });
    }
    wordIndex += 1;
  }

  return units;
}

export function resolveWordAnimation(phrase: string): SignAnimationUnit {
  return {
    phrase,
    kind: "word",
    animationKey: null,
  };
}
