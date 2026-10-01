import { normalizeSentence } from "./normalizeSentence";

export type SentenceSegmentResult = {
  complete: string[];
  remainder: string;
};

const SENTENCE_END = /[.!?]+(?=\s|$)/;

export function segmentCompleteSentences(input: string): SentenceSegmentResult {
  const normalized = normalizeSentence(input);
  if (!normalized) return { complete: [], remainder: "" };

  const complete: string[] = [];
  let remainder = normalized;

  while (true) {
    const match = SENTENCE_END.exec(remainder);
    if (!match || match.index < 0) break;

    const end = match.index + match[0].length;
    const sentence = normalizeSentence(remainder.slice(0, end));
    if (sentence) complete.push(sentence);
    remainder = normalizeSentence(remainder.slice(end));
  }

  return { complete, remainder };
}
