export function normalizeSentence(input: string): string {
  return input
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\t\r\n]+/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.!?;:])/g, "$1")
    .trim();
}

export function normalizeTranscript(input: string): string {
  return normalizeSentence(input);
}

export function sentenceKey(input: string): string {
  return normalizeSentence(input).toLowerCase();
}
