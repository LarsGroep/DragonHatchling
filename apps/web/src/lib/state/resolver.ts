/**
 * resolve — the pure resolver at the heart of bidirectional sync (§11).
 *
 * Maps any EntityRef to the full linked set (token ↔ gaussian ↔ graph node ↔
 * embedding point ↔ image patch ↔ concepts) using only the O(1) index maps in
 * PackIndex. No I/O, no view imports; every view calls this and highlights the
 * fields it renders. Simplicity is the point — this is the entire
 * synchronization backbone the future views subscribe to.
 *
 * Linking rules (frozen grid convention):
 *   • token idx ↔ patch (row,col): `divmod(idx-1, grid)`, CLS(0) has no patch.
 *   • gaussian idx == token idx (distinct kind only for hit-testing).
 *   • graph node id == `L{layer}_T{idx}`.
 *   • embedding point exists only for CLS (idx 0) — the per-layer trajectory.
 *   • concepts via concepts.json feature ids (token→features, feature→tokens).
 *
 * A ref that lacks a layer (patch) resolves at `atLayer` (the caller passes the
 * current timeline layer). Concept refs fan out to every firing token.
 */
import type { EntityRef } from "./refs";
import { refKey } from "./refs";
import {
  nodeId,
  parseNodeId,
  patchToToken,
  tokenToPatch,
  type PackIndex,
} from "./packIndex";

export interface ResolvedSelection {
  /** Effective layer (token/gaussian/node/point); -1 when not applicable. */
  layer: number;
  /** Effective token index; -1 when the ref is a fan-out (concept). */
  idx: number;
  isCls: boolean;
  patch: { row: number; col: number } | null;
  token: { layer: number; idx: number } | null;
  gaussian: { layer: number; idx: number } | null;
  node: string | null;
  point: { imageId: string; layer: number } | null;
  /** Concept ids linked to the selection. */
  concepts: string[];
  /** The full linked set as EntityRefs (deduped, includes the input ref). */
  refs: EntityRef[];
}

function empty(): ResolvedSelection {
  return {
    layer: -1,
    idx: -1,
    isCls: false,
    patch: null,
    token: null,
    gaussian: null,
    node: null,
    point: null,
    concepts: [],
    refs: [],
  };
}

function dedup(refs: EntityRef[]): EntityRef[] {
  const seen = new Set<string>();
  const out: EntityRef[] = [];
  for (const r of refs) {
    const k = refKey(r);
    if (!seen.has(k)) {
      seen.add(k);
      out.push(r);
    }
  }
  return out;
}

/** Build the linked set for a single (layer, idx) token. O(1). */
function resolveToken(index: PackIndex, layer: number, idx: number): ResolvedSelection {
  const isCls = idx === index.clsIndex;
  const patchRC = tokenToPatch(idx, index.grid, index.clsIndex);
  const patch = patchRC ? { row: patchRC[0], col: patchRC[1] } : null;
  const concepts = index.tokenConcepts[idx] ?? [];
  const point = isCls ? { imageId: index.imageId, layer } : null;

  const refs: EntityRef[] = [
    { kind: "token", layer, idx },
    { kind: "gaussian", layer, idx },
    { kind: "node", id: nodeId(layer, idx) },
  ];
  if (patch) refs.push({ kind: "patch", row: patch.row, col: patch.col });
  if (point) refs.push({ kind: "point", imageId: point.imageId, layer });
  for (const id of concepts) refs.push({ kind: "concept", id });

  return {
    layer,
    idx,
    isCls,
    patch,
    token: { layer, idx },
    gaussian: { layer, idx },
    node: nodeId(layer, idx),
    point,
    concepts,
    refs: dedup(refs),
  };
}

/**
 * Resolve `ref` into its full cross-view linked set using `index`. `atLayer` is
 * the timeline layer used for refs that carry no layer of their own (patch);
 * defaults to the last attention layer.
 */
export function resolve(
  ref: EntityRef,
  index: PackIndex,
  atLayer?: number,
): ResolvedSelection {
  const defaultLayer = atLayer ?? Math.max(0, index.numLayers - 1);

  switch (ref.kind) {
    case "token":
    case "gaussian":
      return resolveToken(index, ref.layer, ref.idx);

    case "node": {
      const parsed = parseNodeId(ref.id);
      if (!parsed) return empty();
      return resolveToken(index, parsed.layer, parsed.idx);
    }

    case "patch": {
      const idx = patchToToken(ref.row, ref.col, index.grid);
      return resolveToken(index, defaultLayer, idx);
    }

    case "point":
      // Embedding point == CLS token trajectory at that layer.
      return resolveToken(index, ref.layer, index.clsIndex);

    case "concept": {
      // Fan out to every token that fires this concept (at the probe layer).
      const layer = index.conceptLayer ?? defaultLayer;
      const tokens = index.conceptTokens.get(ref.id) ?? [];
      const refs: EntityRef[] = [{ kind: "concept", id: ref.id }];
      for (const idx of tokens) {
        refs.push({ kind: "token", layer, idx });
        refs.push({ kind: "gaussian", layer, idx });
        refs.push({ kind: "node", id: nodeId(layer, idx) });
        const pc = tokenToPatch(idx, index.grid, index.clsIndex);
        if (pc) refs.push({ kind: "patch", row: pc[0], col: pc[1] });
      }
      return {
        ...empty(),
        layer,
        concepts: [ref.id],
        refs: dedup(refs),
      };
    }
  }
}

/**
 * Convenience: resolve a ref and return the set of patch cells it lights up
 * (used by Image Space to highlight from any-view selections). Returns a Set of
 * `"row:col"` keys.
 */
export function resolvedPatches(
  ref: EntityRef | null,
  index: PackIndex | null,
  atLayer?: number,
): Set<string> {
  const out = new Set<string>();
  if (!ref || !index) return out;
  const r = resolve(ref, index, atLayer);
  for (const e of r.refs) {
    if (e.kind === "patch") out.add(`${e.row}:${e.col}`);
  }
  return out;
}
