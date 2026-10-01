import type { SignAsset } from "./signTypes";

const ATLAS_SOURCE = "/signs/sign-atlas.webp";
const ATLAS_COLUMNS = 7;
const ATLAS_ROWS = 4;

function atlasPosition(index: number): string {
  const column = index % ATLAS_COLUMNS;
  const row = Math.floor(index / ATLAS_COLUMNS);
  return (column / (ATLAS_COLUMNS - 1)) * 100 + "% " + (row / (ATLAS_ROWS - 1)) * 100 + "%";
}

export const LETTER_SIGN_ASSETS: Record<string, SignAsset> = Object.fromEntries(
  Array.from({ length: 26 }, (_, index) => {
    const letter = String.fromCharCode(65 + index);
    return [
      letter,
      {
        kind: "letter" as const,
        id: "letter-" + letter.toLowerCase(),
        label: letter,
        src: ATLAS_SOURCE,
        backgroundPosition: atlasPosition(index),
      },
    ];
  }),
);

export const WORD_SIGN_ASSETS: Record<string, SignAsset> = {};
export const PHRASE_SIGN_ASSETS: Record<string, SignAsset> = {};

export function getLetterAsset(letter: string): SignAsset | null {
  return LETTER_SIGN_ASSETS[letter.toUpperCase()] || null;
}

export function getWordAsset(word: string): SignAsset | null {
  return WORD_SIGN_ASSETS[word.toLowerCase().trim()] || null;
}

export function getPhraseAsset(phrase: string): SignAsset | null {
  return PHRASE_SIGN_ASSETS[phrase.toLowerCase().trim()] || null;
}

export function getAnySignAsset(text: string): SignAsset | null {
  return getPhraseAsset(text) || getWordAsset(text);
}
