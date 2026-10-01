export type SentenceStatus = "building" | "ready" | "playing" | "completed";

export type Sentence = {
  id: string;
  text: string;
  createdAt: number;
  status: SentenceStatus;
};
