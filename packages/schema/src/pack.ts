/**
 * ViTreous Explanation Pack — TypeScript type mirror of pack.schema.json.
 *
 * This file is the TypeScript arm of the single-source-of-truth trio
 * (JSON Schema -> Pydantic -> TS). It MUST stay structurally identical to
 * `../schema/pack.schema.json`. The round-trip test in
 * `tests/pack.typecheck.ts` imports the shared fixture and asserts it
 * `satisfies PackManifest`, so drift breaks the build.
 *
 * PACK FORMAT v1, FROZEN AT M2 (§5 of ARCHITECTURE.md). Any change here must be
 * mirrored in `../schema/pack.schema.json` and the Pydantic models in
 * `packages/core/.../packs/manifest.py`, and bump `pack_version`.
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

/**
 * Per-row max-quantization parameters (present only for `per_row_uint8`).
 * On-disk layout: uint8 data block (C-order, `row_axis` = last axis) followed
 * by the per-row float32 scales block. Dequantize row r as
 * `data[r] / 255 * scale[r]`; error <= 0.5/255 per element.
 */
export interface QuantInfo {
  scheme: "per_row_uint8";
  row_axis: number;
  scale_dtype: "float32";
  data_offset: number;
  data_bytes: number;
  scale_offset: number;
  scale_count: number;
}

/** Descriptor for one binary/JSON asset referenced by the manifest. */
export interface AssetEntry {
  dtype: AssetDtype;
  shape: number[];
  encoding: AssetEncoding;
  bytes: number;
  checksum?: string;
  quant?: QuantInfo;
  /**
   * Optional additive free-form asset metadata (e.g. channel order for
   * gaussians.bin). Added at M3; existing v1 assets omit it. Never describes
   * the frozen binary layout — that stays fixed by dtype/shape/encoding/quant.
   */
  meta?: Record<string, unknown>;
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
