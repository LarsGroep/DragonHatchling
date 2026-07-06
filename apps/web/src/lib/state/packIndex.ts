/**
 * PackIndex — the O(1) index maps the resolver (§11) uses to link an EntityRef
 * to the full cross-view set. Built from the manifest geometry (always) and,
 * additively, the concept tier (when concepts.json is present). Concepts are
 * the only asset that requires an extra fetch to enrich the index; everything
 * else is derivable from the frozen grid convention (token 0 = CLS; patches
 * `[1:]` reshape to 14×14 via `divmod(i-1, grid)`).
 */
import type { PackManifest } from "@vitreous/schema";
import type { ConceptsJson } from "../pack/types";

export interface PackIndex {
  imageId: string;
  numLayers: number; // attention layers L
  numSteps: number; // timeline steps L+1 (0..L)
  numTokens: number; // T
  grid: number; // 14
  clsIndex: number; // 0
  /** Concept probe layer (from concepts.json) or null when absent. */
  conceptLayer: number | null;
  /** token idx -> concept ids (strings). Empty when no concept tier. */
  tokenConcepts: string[][];
  /** concept id -> token idxs that fire it (reverse map). */
  conceptTokens: Map<string, number[]>;
  /** dictionary id for concept refs, or null. */
  dictionaryId: string | null;
}

/** token idx (1..T-1) -> [row, col] on the patch grid; CLS (0) -> null. */
export function tokenToPatch(idx: number, grid: number, clsIndex = 0): [number, number] | null {
  if (idx === clsIndex) return null;
  const j = idx - 1;
  return [Math.floor(j / grid), j % grid];
}

/** [row, col] -> token idx (inverse of tokenToPatch). */
export function patchToToken(row: number, col: number, grid: number): number {
  return row * grid + col + 1;
}

/** Canonical per-layer graph node id (matches vitreous.graph.node_id). */
export function nodeId(layer: number, idx: number): string {
  return `L${layer}_T${idx}`;
}

/** Parse a canonical node id back to {layer, idx}; null if malformed. */
export function parseNodeId(id: string): { layer: number; idx: number } | null {
  const m = /^L(\d+)_T(\d+)$/.exec(id);
  if (!m) return null;
  return { layer: Number(m[1]), idx: Number(m[2]) };
}

export function buildPackIndex(
  manifest: PackManifest,
  concepts?: ConceptsJson | null,
): PackIndex {
  const numTokens = manifest.model.num_tokens ?? 197;
  const numLayers = manifest.model.num_layers ?? 12;
  const grid = Math.round(Math.sqrt(numTokens - 1));

  const tokenConcepts: string[][] = Array.from({ length: numTokens }, () => []);
  const conceptTokens = new Map<string, number[]>();
  let conceptLayer: number | null = null;
  let dictionaryId: string | null = null;

  if (concepts) {
    conceptLayer = concepts.layer;
    dictionaryId = concepts.dictionary_id;
    for (let t = 0; t < concepts.feature_ids.length && t < numTokens; t++) {
      const ids = concepts.feature_ids[t].map((f) => String(f));
      tokenConcepts[t] = ids;
      for (const id of ids) {
        const arr = conceptTokens.get(id);
        if (arr) {
          if (arr[arr.length - 1] !== t) arr.push(t);
        } else {
          conceptTokens.set(id, [t]);
        }
      }
    }
  }

  return {
    imageId: manifest.image.id,
    numLayers,
    numSteps: numLayers + 1,
    numTokens,
    grid,
    clsIndex: 0,
    conceptLayer,
    tokenConcepts,
    conceptTokens,
    dictionaryId,
  };
}
