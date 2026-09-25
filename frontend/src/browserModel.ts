import { normalizeHandPair } from "./landmarks";

export type BrowserLayer = {
  weight: number[] | number[][];
  bias: number[];
};

export type BrowserModelPayload = {
  model_version: string;
  model_sha256: string;
  preprocessing_version?: string;
  landmark_runtime?: string;
  input_dim: number;
  hidden_dim: number;
  embedding_dim: number;
  num_classes: number;
  labels: string[];
  layers: {
    input_norm: BrowserLayer;
    hidden: BrowserLayer;
    embedding: BrowserLayer;
    embedding_norm: BrowserLayer;
    head: BrowserLayer;
  };
};

export type BrowserAdapterPayload = {
  version: number;
  method: string;
  base_model_version: string;
  preprocessing_version: string;
  landmark_runtime: string;
  base_model_sha256: string;
  feature_dim: number;
  embedding_dim: number;
  prototypes: Record<string, number[]>;
  shots: Record<string, number>;
  calibration_samples?: Array<{ letter: string; hand_keypoints: number[] }>;
  base_model_labels?: string[];
};

function gelu(value: number): number {
  const cubic = value * value * value;
  return 0.5 * value * (1 + Math.tanh(0.7978845608028654 * (value + 0.044715 * cubic)));
}

function layerNorm(input: Float32Array, weight: Float32Array, bias: Float32Array): Float32Array {
  let mean = 0;
  for (let i = 0; i < input.length; i += 1) mean += input[i];
  mean /= input.length;

  let variance = 0;
  for (let i = 0; i < input.length; i += 1) {
    const diff = input[i] - mean;
    variance += diff * diff;
  }
  variance /= input.length;
  const invStd = 1 / Math.sqrt(variance + 1e-5);

  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    output[i] = (input[i] - mean) * invStd * weight[i] + bias[i];
  }
  return output;
}

function dense(
  input: Float32Array,
  weights: Float32Array,
  bias: Float32Array,
  rows: number,
  cols: number,
): Float32Array {
  const output = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    let sum = bias[row];
    const offset = row * cols;
    for (let col = 0; col < cols; col += 1) {
      sum += weights[offset + col] * input[col];
    }
    output[row] = sum;
  }
  return output;
}

function flatten(rows: number[][]): Float32Array {
  return Float32Array.from(rows.flat());
}

const MODEL_VERSION = "visionbridge-letter-base-v3";
const PREPROCESSING_VERSION = "two-hand-wrist-scale-v1";
const LANDMARK_RUNTIME = "mediapipe-hand-landmarker-0.10.35";

export class BrowserLetterModel {
  readonly modelVersion: string;
  readonly modelSha256: string;
  readonly inputDim: number;
  readonly hiddenDim: number;
  readonly embeddingDim: number;
  readonly labels: string[];

  private readonly inputNormWeight: Float32Array;
  private readonly inputNormBias: Float32Array;
  private readonly hiddenWeight: Float32Array;
  private readonly hiddenBias: Float32Array;
  private readonly embeddingWeight: Float32Array;
  private readonly embeddingBias: Float32Array;
  private readonly embeddingNormWeight: Float32Array;
  private readonly embeddingNormBias: Float32Array;
  private readonly headWeight: Float32Array;
  private readonly headBias: Float32Array;
  private readonly numClasses: number;

  constructor(payload: BrowserModelPayload) {
    if (payload.input_dim !== 126) throw new Error("Unsupported browser feature dimension");
    if (!payload.hidden_dim || !payload.embedding_dim || !payload.num_classes) {
      throw new Error("Invalid browser model dimensions");
    }
    if (payload.labels.length !== payload.num_classes) {
      throw new Error("Browser model label metadata is invalid");
    }

    if (payload.model_version !== MODEL_VERSION) {
      throw new Error("Unsupported browser model version");
    }
    if (!payload.model_sha256) {
      throw new Error("Browser model checksum is missing");
    }
    if (payload.preprocessing_version !== PREPROCESSING_VERSION) {
      throw new Error("Browser model preprocessing is incompatible");
    }
    if (payload.landmark_runtime !== LANDMARK_RUNTIME) {
      throw new Error("Browser model landmark runtime is incompatible");
    }
    if (
      payload.labels.length !== 26 ||
      payload.labels.join("") !== "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ) {
      throw new Error("Browser model label vocabulary is incompatible");
    }

    this.modelVersion = payload.model_version;
    this.modelSha256 = payload.model_sha256;
    this.inputDim = payload.input_dim;
    this.hiddenDim = payload.hidden_dim;
    this.embeddingDim = payload.embedding_dim;
    this.numClasses = payload.num_classes;
    this.labels = payload.labels;

    this.inputNormWeight = Float32Array.from(payload.layers.input_norm.weight as number[]);
    this.inputNormBias = Float32Array.from(payload.layers.input_norm.bias);
    this.hiddenWeight = flatten(payload.layers.hidden.weight as number[][]);
    this.hiddenBias = Float32Array.from(payload.layers.hidden.bias);
    this.embeddingWeight = flatten(payload.layers.embedding.weight as number[][]);
    this.embeddingBias = Float32Array.from(payload.layers.embedding.bias);
    this.embeddingNormWeight = Float32Array.from(payload.layers.embedding_norm.weight as number[]);
    this.embeddingNormBias = Float32Array.from(payload.layers.embedding_norm.bias);
    this.headWeight = flatten(payload.layers.head.weight as number[][]);
    this.headBias = Float32Array.from(payload.layers.head.bias);

    if (
      this.inputNormWeight.length !== this.inputDim ||
      this.inputNormBias.length !== this.inputDim ||
      this.hiddenWeight.length !== this.hiddenDim * this.inputDim ||
      this.hiddenBias.length !== this.hiddenDim ||
      this.embeddingWeight.length !== this.embeddingDim * this.hiddenDim ||
      this.embeddingBias.length !== this.embeddingDim ||
      this.embeddingNormWeight.length !== this.embeddingDim ||
      this.embeddingNormBias.length !== this.embeddingDim ||
      this.headWeight.length !== this.numClasses * this.embeddingDim ||
      this.headBias.length !== this.numClasses
    ) {
      throw new Error("Browser model tensors do not match their metadata");
    }
  }

  embed(raw: number[] | Float32Array): Float32Array {
    const input = raw instanceof Float32Array ? raw : Float32Array.from(raw);
    if (input.length !== this.inputDim) throw new Error("Invalid browser model input");

    const normalizedInput = layerNorm(input, this.inputNormWeight, this.inputNormBias);
    const hidden = dense(
      normalizedInput,
      this.hiddenWeight,
      this.hiddenBias,
      this.hiddenDim,
      this.inputDim,
    );
    for (let i = 0; i < hidden.length; i += 1) hidden[i] = gelu(hidden[i]);

    const embedding = dense(
      hidden,
      this.embeddingWeight,
      this.embeddingBias,
      this.embeddingDim,
      this.hiddenDim,
    );
    const normalizedEmbedding = layerNorm(
      embedding,
      this.embeddingNormWeight,
      this.embeddingNormBias,
    );
    for (let i = 0; i < normalizedEmbedding.length; i += 1) {
      normalizedEmbedding[i] = gelu(normalizedEmbedding[i]);
    }
    return normalizedEmbedding;
  }

  predictBase(raw: number[] | Float32Array): { label: string; confidence: number } {
    const embedding = this.embed(raw);
    const logits = dense(
      embedding,
      this.headWeight,
      this.headBias,
      this.numClasses,
      this.embeddingDim,
    );

    let maxIndex = 0;
    for (let i = 1; i < logits.length; i += 1) {
      if (logits[i] > logits[maxIndex]) maxIndex = i;
    }

    let denominator = 0;
    let numerator = 0;
    const maxLogit = logits[maxIndex];
    for (let i = 0; i < logits.length; i += 1) {
      const value = Math.exp(logits[i] - maxLogit);
      denominator += value;
      if (i === maxIndex) numerator = value;
    }

    return {
      label: this.labels[maxIndex],
      confidence: numerator / Math.max(denominator, 1e-12),
    };
  }
}

function unit(values: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < values.length; i += 1) norm += values[i] * values[i];
  norm = Math.sqrt(norm);
  if (norm < 1e-8) throw new Error("Degenerate embedding");
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) result[i] = values[i] / norm;
  return result;
}

export class BrowserLetterAdapter {
  private readonly model: BrowserLetterModel;
  private prototypes = new Map<string, Float32Array>();
  private readonly threshold: number;

  constructor(model: BrowserLetterModel, payload: BrowserAdapterPayload, threshold = 0.35) {
    this.model = model;
    this.threshold = threshold;
    this.load(payload);
  }

  private load(payload: BrowserAdapterPayload): void {
    if (
      payload.version !== 4 ||
      payload.method !== "dynamic-base-embedding-prototype" ||
      payload.feature_dim !== 126 ||
      payload.embedding_dim !== this.model.embeddingDim
    ) {
      throw new Error("Adapter contract does not match the current model");
    }

    if (payload.base_model_version !== this.model.modelVersion) {
      throw new Error("Adapter requires recalibration for the current model version");
    }

    if (payload.base_model_sha256 !== this.model.modelSha256) {
      throw new Error("Adapter requires recalibration for the current model");
    }

    if (payload.preprocessing_version !== PREPROCESSING_VERSION) {
      throw new Error("Adapter preprocessing is incompatible with the current model");
    }

    if (payload.landmark_runtime !== LANDMARK_RUNTIME) {
      throw new Error("Adapter landmark runtime is incompatible with the current model");
    }

    if (
      !payload.base_model_labels ||
      payload.base_model_labels.length !== this.model.labels.length ||
      payload.base_model_labels.some((label, index) => label !== this.model.labels[index])
    ) {
      throw new Error("Adapter label vocabulary is incompatible with the current model");
    }

    for (const [letter, values] of Object.entries(payload.prototypes)) {
      if (
        !this.model.labels.includes(letter) ||
        !Array.isArray(values) ||
        values.length !== this.model.embeddingDim ||
        !values.every((value) => Number.isFinite(value))
      ) {
        throw new Error("Adapter prototype is invalid");
      }
      this.prototypes.set(letter, unit(Float32Array.from(values)));
    }
    if (this.prototypes.size < 2) throw new Error("Adapter requires at least two prototypes");
  }

  predict(raw: number[]): { predicted_letter: string; confidence: number; similarity: number; inference_ms: number } {
    const started = performance.now();
    const normalized = normalizeHandPair(raw);
    const embedding = unit(this.model.embed(normalized));

    let bestLetter = "?";
    let bestScore = -Infinity;
    let bestIndex = -1;
    const scores: number[] = [];

    for (const [letter, prototype] of this.prototypes) {
      let score = 0;
      for (let i = 0; i < embedding.length; i += 1) score += embedding[i] * prototype[i];
      scores.push(score);
      if (score > bestScore) {
        bestScore = score;
        bestLetter = letter;
        bestIndex = scores.length - 1;
      }
    }

    const maxScore = Math.max(...scores);
    let denominator = 0;
    let numerator = 0;
    for (let i = 0; i < scores.length; i += 1) {
      const value = Math.exp((scores[i] - maxScore) * 10);
      denominator += value;
      if (i === bestIndex) numerator = value;
    }

    const confidence = numerator / Math.max(denominator, 1e-12);
    return {
      predicted_letter: bestScore < this.threshold ? "?" : bestLetter,
      confidence,
      similarity: bestScore,
      inference_ms: performance.now() - started,
    };
  }
}
