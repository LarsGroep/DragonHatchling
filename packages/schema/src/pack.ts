/**
 * ViTreous Explanation Pack — TypeScript type mirror of pack.schema.json.
 *
 * This file is the TypeScript arm of the single-source-of-truth trio
 * (JSON Schema -> Pydantic -> TS). It MUST stay structurally identical to
 * `../schema/pack.schema.json`. The round-trip test in
 * `tests/pack.typecheck.ts` imports the shared fixture and asserts it
 * `satisfies PackManifest`, so drift breaks the build.
 */

/** Element dtype (or container type) of a decoded asset. */
export type AssetDtype =
  | "uint8"
  | "int8"
  | "uint16"
  | "int16"
  | "int32"
  | "float16"
  | "float32"
  | "float64"
  | "json"
  | "webp"
  | "png";

/** On-disk encoding of an asset's bytes. */
export type AssetEncoding =
  | "raw"
  | "zstd"
  | "gzip"
  | "per_row_uint8"
  | "json"
  | "webp"
  | "png";

/** Provenance of the analyzed image. */
export type ImageSource = "gallery" | "upload";

export interface ModelInfo {
  arch: string;
  hf_repo: string;
  num_layers?: number;
  num_heads?: number;
  num_tokens?: number;
  embed_dim?: number;
  patch_size?: number;
}

export interface DatasetInfo {
  name: string;
  display_name?: string;
  num_classes: number;
  class_names: string[];
}

export interface ImageMeta {
  id: string;
  width: number;
  height: number;
  source: ImageSource;
}

export interface Prediction {
  label: string;
  class_index: number;
  confidence: number;
  probabilities: number[];
}

/** Descriptor for one binary/JSON asset referenced by the manifest. */
export interface AssetEntry {
  dtype: AssetDtype;
  shape: number[];
  encoding: AssetEncoding;
  bytes: number;
  checksum?: string;
}

/** Asset index: filename -> descriptor. */
export type AssetIndex = Record<string, AssetEntry>;

/** Wall-clock stage timings in milliseconds. */
export type Timings = Record<string, number>;

/** Top-level Explanation Pack manifest (manifest.json). */
export interface PackManifest {
  pack_version: string;
  model: ModelInfo;
  dataset: DatasetInfo;
  image: ImageMeta;
  prediction: Prediction;
  assets: AssetIndex;
  timings: Timings;
}
