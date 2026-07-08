import { describe, expect, it } from "vitest";
import { topFiringFeatures } from "./concepts";
import type { ConceptsJson } from "@/src/lib/pack/types";

function make(feature_ids: number[][], activations: number[][]): ConceptsJson {
  return {
    layer: 9,
    topk: feature_ids[0]?.length ?? 0,
    dictionary_id: "test",
    provider_kind: "sae_topk",
    n_concepts: 100,
    num_tokens: feature_ids.length,
    feature_ids,
    activations,
  };
}

describe("topFiringFeatures", () => {
  it("aggregates strength across tokens and excludes CLS", () => {
    // token0 (CLS) fires feature 99 strongly — must be ignored.
    const c = make(
      [
        [99],
        [7],
        [7],
        [3],
      ],
      [
        [9],
        [0.5],
        [0.5],
        [0.9],
      ],
    );
    const top = topFiringFeatures(c, 6);
    expect(top.find((f) => f.id === 99)).toBeUndefined();
    // feature 7 fires in two tokens summing to 1.0 → strongest.
    expect(top[0].id).toBe(7);
    expect(top[0].strength).toBeCloseTo(1.0);
    expect(top[0].count).toBe(2);
  });

  it("respects the topN cap and sorts by strength desc", () => {
    const c = make(
      [
        [0],
        [1, 2],
        [2, 3],
      ],
      [
        [0],
        [0.2, 0.9],
        [0.8, 0.1],
      ],
    );
    const top = topFiringFeatures(c, 2);
    expect(top).toHaveLength(2);
    expect(top[0].strength).toBeGreaterThanOrEqual(top[1].strength);
    // feature 2 appears twice (0.9+0.8=1.7) → strongest
    expect(top[0].id).toBe(2);
  });
});
