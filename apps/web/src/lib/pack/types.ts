/**
 * TypeScript shapes for the pack's *JSON sidecar* assets (§5–§10). The frozen
 * manifest itself is typed by the single source of truth `@vitreous/schema`
 * (pack.ts); these describe the payloads of the JSON assets the manifest
 * indexes (attributions.json, faithfulness.json, graph.json, concepts.json),
 * mirroring the structures emitted by packages/core.
 */

import type { PackManifest } from "@vitreous/schema";

/** attributions.json — index of per-method attribution assets. */
export type AttributionKind = "per_layer_tokens" | "token_grid" | "tokens";
export interface AttributionEntry {
  asset: string;
  kind: AttributionKind;
  pixel_asset?: string;
}
export type AttributionsIndex = Record<string, AttributionEntry>;

/** faithfulness.json (vitreous.xai.eval.FaithfulnessResult.to_json). */
export interface FaithfulnessJson {
  steps: number;
  deletion_curves: Record<string, number[]>;
  insertion_curves: Record<string, number[]>;
  deletion_auc: Record<string, number>;
  insertion_auc: Record<string, number>;
  agreement: Record<string, Record<string, number>>;
}

/** graph.json (vitreous.graph.build_graph_asset). */
export interface GraphNodeJson {
  idx: number;
  kind: string;
  community: number;
}
/** Compact edge triple: [srcIdx, dstIdx, weight]. */
export type GraphEdgeJson = [number, number, number];
export interface GraphLayerJson {
  layer: number;
  nodes: GraphNodeJson[];
  edges: GraphEdgeJson[];
}
export interface GraphJson {
  num_layers: number;
  num_tokens: number;
  k: number;
  grid: number;
  cls_index: number;
  seed: number;
  edge_semantics: string;
  residual: {
    kind: string;
    materialized: boolean;
    weight: number;
    count: number;
    description: string;
  };
  layers: GraphLayerJson[];
}

/** concepts.json (§9). May be ABSENT — loaders return null. */
export interface ConceptsJson {
  layer: number;
  topk: number;
  dictionary_id: string;
  provider_kind: string;
  n_concepts: number;
  num_tokens: number;
  feature_ids: number[][]; // [num_tokens][topk]
  activations: number[][]; // [num_tokens][topk]
}

/** Loaded attribution asset, decoded to a flat Float32Array + its shape/kind. */
export interface LoadedAttribution {
  method: string;
  kind: AttributionKind;
  data: Float32Array;
  shape: number[];
  /** Optional pixel-level heatmap asset name (IG). */
  pixelAsset?: string;
}

/** Loaded Gaussian Feature Field: flat fp32 data + channel names + geometry. */
export interface LoadedGaussians {
  data: Float32Array; // [S, N, C] C-order
  channels: string[];
  steps: number;
  tokens: number;
  channelCount: number;
}

/** Decoded token embeddings. */
export interface LoadedTokens {
  data: Float32Array; // [S, T, D] C-order
  steps: number;
  tokens: number;
  dim: number;
}

/** Decoded attention. */
export interface LoadedAttention {
  data: Float32Array; // [L, H, T, T] C-order
  layers: number;
  heads: number;
  tokens: number;
}

export type { PackManifest };
