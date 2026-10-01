import { appendReconciledText } from "./transcriptReconciler";
import { normalizeSentence } from "./normalizeSentence";
import { segmentCompleteSentences } from "./sentenceSegmenter";

export type SentenceBufferAppendResult = {
  complete: string[];
  remainder: string;
};

export class SentenceBuffer {
  private value = "";

  append(text: string): SentenceBufferAppendResult {
    this.value = appendReconciledText(this.value, text);
    const result = segmentCompleteSentences(this.value);
    this.value = result.remainder;
    return result;
  }

  flush(): string {
    const sentence = normalizeSentence(this.value);
    this.value = "";
    return sentence;
  }

  clear(): void {
    this.value = "";
  }

  get text(): string {
    return this.value;
  }
}
