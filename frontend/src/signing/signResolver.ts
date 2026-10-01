import { getAnySignAsset, getLetterAsset, getPhraseAsset, getWordAsset } from "./signAssets";
import type { ResolvedSentence, SignPlaybackItem, SignTokenKind } from "./signTypes";
import type { Sentence } from "../sentence/sentenceTypes";
import type { VocabularyItem } from "../data/vocabulary";

type PhraseMatch = {
  phrase: string;
  words: string[];
};

function normalizedPhrase(input: string): string {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9'\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function vocabularyPhrases(vocabulary: VocabularyItem[]): PhraseMatch[] {
  return vocabulary
    .map((item) => {
      const phrase = normalizedPhrase(item.phrase);
      return phrase ? { phrase, words: phrase.split(" ") } : null;
    })
    .filter((item): item is PhraseMatch => Boolean(item))
    .sort((a, b) => b.words.length - a.words.length || b.phrase.length - a.phrase.length);
}

function wordsFromSentence(text: string): string[] {
  return normalizedPhrase(text).split(" ").filter(Boolean);
}

function makeToken(
  sourceText: string,
  kind: SignTokenKind,
  asset: ReturnType<typeof getAnySignAsset> = null,
): SignPlaybackItem {
  return {
    id: kind + "-" + sourceText.toLowerCase().replace(/[^a-z0-9]+/g, "-") + "-" + Math.random().toString(36).slice(2, 8),
    sourceText,
    kind,
    asset,
    durationBaseMs: 0,
  };
}

function fingerspellWord(word: string): SignPlaybackItem[] {
  return [...word.toUpperCase()]
    .filter((letter) => /[A-Z]/.test(letter))
    .map((letter) => makeToken(letter, "letter", getLetterAsset(letter)));
}

export function resolveSentence(
  sentence: Sentence,
  vocabulary: VocabularyItem[],
): ResolvedSentence {
  const phraseMatches = vocabularyPhrases(vocabulary);
  const words = wordsFromSentence(sentence.text);
  const tokens: SignPlaybackItem[] = [];
  let index = 0;

  while (index < words.length) {
    let matched: PhraseMatch | null = null;

    for (const candidate of phraseMatches) {
      if (index + candidate.words.length > words.length) continue;
      let valid = true;
      for (let offset = 0; offset < candidate.words.length; offset += 1) {
        if (words[index + offset] !== candidate.words[offset]) {
          valid = false;
          break;
        }
      }
      if (valid) {
        matched = candidate;
        break;
      }
    }

    if (matched) {
      const phraseAsset = getPhraseAsset(matched.phrase);
      if (phraseAsset) {
        tokens.push(makeToken(matched.phrase, "phrase", phraseAsset));
      } else {
        for (const word of matched.words) {
          const wordAsset = getWordAsset(word);
          if (wordAsset) {
            tokens.push(makeToken(word, "word", wordAsset));
          } else {
            tokens.push(...fingerspellWord(word));
          }
        }
      }
      index += matched.words.length;
      continue;
    }

    const word = words[index];
    const wordAsset = getWordAsset(word);
    if (wordAsset) {
      tokens.push(makeToken(word, "word", wordAsset));
    } else {
      const spelled = fingerspellWord(word);
      if (spelled.length) {
        tokens.push(...spelled);
      } else {
        tokens.push(makeToken(word, "unsupported", null));
      }
    }
    index += 1;
  }

  return {
    ...sentence,
    signSequence: tokens,
  };
}
