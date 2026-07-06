/**
 * EntityRef — the single selection vocabulary shared by every view (§11). This
 * union is the contract the resolver links across the four spaces; it is copied
 * VERBATIM from ARCHITECTURE.md §11. No view invents its own reference type.
 */

export type AttributionMethod = "chefer" | "rollout" | "gradcam" | "ig";

export type EntityRef =
  | { kind: "token"; layer: number; idx: number }
  | { kind: "gaussian"; layer: number; idx: number } // ≅ token, distinct for hit-testing
  | { kind: "node"; id: string }
  | { kind: "concept"; id: string }
  | { kind: "point"; imageId: string; layer: number }
  | { kind: "patch"; row: number; col: number };

/** Stable string key for a ref (dedup, Set membership, React keys). */
export function refKey(ref: EntityRef): string {
  switch (ref.kind) {
    case "token":
      return `token:${ref.layer}:${ref.idx}`;
    case "gaussian":
      return `gaussian:${ref.layer}:${ref.idx}`;
    case "node":
      return `node:${ref.id}`;
    case "concept":
      return `concept:${ref.id}`;
    case "point":
      return `point:${ref.imageId}:${ref.layer}`;
    case "patch":
      return `patch:${ref.row}:${ref.col}`;
  }
}

export function refsEqual(a: EntityRef | null, b: EntityRef | null): boolean {
  if (a === null || b === null) return a === b;
  return refKey(a) === refKey(b);
}
