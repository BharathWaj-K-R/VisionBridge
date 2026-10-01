import { normalizeSentence, sentenceKey } from "./normalizeSentence";

function words(input: string): string[] {
  return normalizeSentence(input).split(" ").filter(Boolean);
}

function overlapLength(left: string[], right: string[]): number {
  const max = Math.min(left.length, right.length);
  for (let size = max; size > 0; size -= 1) {
    let matches = true;
    for (let index = 0; index < size; index += 1) {
      if (sentenceKey(left[left.length - size + index]) !== sentenceKey(right[index])) {
        matches = false;
        break;
      }
    }
    if (matches) return size;
  }
  return 0;
}

export function appendReconciledText(current: string, incoming: string): string {
  const normalizedIncoming = normalizeSentence(incoming);
  if (!normalizedIncoming) return normalizeSentence(current);

  const currentWords = words(current);
  const incomingWords = words(normalizedIncoming);

  if (!currentWords.length) return normalizedIncoming;

  const currentNormalized = sentenceKey(current);
  const incomingNormalized = sentenceKey(normalizedIncoming);

  if (currentNormalized === incomingNormalized || currentNormalized.endsWith(" " + incomingNormalized)) {
    return normalizeSentence(current);
  }

  const overlap = overlapLength(currentWords, incomingWords);
  const remainder = incomingWords.slice(overlap).join(" ");

  if (!remainder) return normalizeSentence(current);

  return normalizeSentence(current + " " + remainder);
}

export function reconcileInterimText(incoming: string): string {
  return normalizeSentence(incoming);
}
