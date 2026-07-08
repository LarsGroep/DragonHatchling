/**
 * concepts.ts — pure helpers for the Concepts + Prediction stub (Sprint B1).
 * The full concepts panel (particle-fed, per-community reveal) is Sprint B2;
 * here we only surface the pack's top firing SAE features from concepts.json.
 */
import type { ConceptsJson } from "@/src/lib/pack/types";

export interface FiringFeature {
  id: number;
  /** Summed activation across all firing tokens (relative strength). */
  strength: number;
  /** How many tokens fire it in their top-k. */
  count: number;
}

/**
 * Aggregate the top firing features across all tokens: for each feature id, sum
 * its activation wherever it appears in a token's top-k, and count occurrences.
 * Returns the strongest `topN`, sorted by summed activation (desc). CLS (token
 * 0) is excluded so labels reflect the image content, not the class token.
 */
export function topFiringFeatures(concepts: ConceptsJson, topN = 6): FiringFeature[] {
  const strength = new Map<number, number>();
  const count = new Map<number, number>();
  const nTok = Math.min(concepts.feature_ids.length, concepts.activations.length);
  for (let t = 1; t < nTok; t++) {
    const ids = concepts.feature_ids[t];
    const acts = concepts.activations[t];
    for (let k = 0; k < ids.length; k++) {
      const id = ids[k];
      const a = acts[k] ?? 0;
      strength.set(id, (strength.get(id) ?? 0) + a);
      count.set(id, (count.get(id) ?? 0) + 1);
    }
  }
  const out: FiringFeature[] = [];
  for (const [id, s] of strength) out.push({ id, strength: s, count: count.get(id) ?? 0 });
  out.sort((a, b) => b.strength - a.strength);
  return out.slice(0, topN);
}
