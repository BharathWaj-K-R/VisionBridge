import type { Sentence } from "../sentence/sentenceTypes";

export type SignAsset =
  | {
      kind: "letter";
      id: string;
      label: string;
      src: string;
      backgroundPosition: string;
    }
  | {
      kind: "word" | "phrase";
      id: string;
      label: string;
      src: string;
      backgroundPosition?: string;
    };

export type SignTokenKind = "letter" | "word" | "phrase" | "unsupported";

export type SignPlaybackItem = {
  id: string;
  sourceText: string;
  kind: SignTokenKind;
  asset: SignAsset | null;
  durationBaseMs: number;
};

export type ResolvedSentence = Sentence & {
  signSequence: SignPlaybackItem[];
};
