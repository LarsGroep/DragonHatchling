/**
 * SGP bundle parser + pure BMU derivations — the exact rules the /sgp views
 * rely on, exercised headless (mirrors umtvit.test.ts).
 */
import { describe, expect, it } from "vitest";
import {
  SgpValidationError,
  bmuTrail,
  depthActivations,
  migrationCurve,
  neuronVoxels,
  parseSgpJson,
  type SgpBundle,
} from "./sgp";

// ── fixture: a tiny but complete valid bundle (2×2×2 SOM, Z=2, 2×2 voxels) ──

function validBundle(): Record<string, unknown> {
  const grid: [number, number, number] = [2, 2, 2];
  const K = 8;
  const nodes = Array.from({ length: K }, (_, k) => ({
    idx: k,
    grid: [Math.floor(k / 4), Math.floor(k / 2) % 2, k % 2],
    hits: k === 3 ? 0 : k + 1,
    umatrix: 0.1 * k,
    community: k % 3,
    dead: k === 3,
  }));
  return {
    sgp_schema_version: 1,
    dataset: "test",
    som: {
      provider: "som",
      grid,
      num_neurons: K,
      depth_steps: 2,
      depth_semantics: "learned hierarchy (encoder depth), not physical depth",
      volume_grid: [2, 2],
      adjacency: "6-connected",
      nodes,
      edges: [
        [0, 1, 0.9],
        [0, 2, 0.5],
        [4, 5, 0.25],
      ],
      edge_semantics: "w = 1/(1+d)",
      communities: { method: "kmeans_weights", k: 3, seed: 0 },
      dead_neurons: 1,
      provenance: { dataset: "test" },
    },
    probes: [
      {
        index: 7,
        thumb_png_b64: "iVBORw0KGgo=",
        bmu: [
          [
            [0, 1],
            [1, 2],
          ],
          [
            [4, 4],
            [4, 5],
          ],
        ],
      },
    ],
    provenance: { run: "unit" },
  };
}

describe("parseSgpJson", () => {
  it("accepts a valid bundle and preserves every field", () => {
    const b: SgpBundle = parseSgpJson(JSON.stringify(validBundle()));
    expect(b.version).toBe(1);
    expect(b.dataset).toBe("test");
    expect(b.som.num_neurons).toBe(8);
    expect(b.som.nodes).toHaveLength(8);
    expect(b.som.nodes[3].dead).toBe(true);
    expect(b.som.edges).toHaveLength(3);
    expect(b.som.volume_grid).toEqual([2, 2]);
    expect(b.probes).toHaveLength(1);
    expect(b.probes[0].bmu[1][1][1]).toBe(5);
  });

  it("rejects non-JSON with a clear message", () => {
    expect(() => parseSgpJson("not json {")).toThrow(SgpValidationError);
    expect(() => parseSgpJson("not json {")).toThrow(/not valid JSON/);
  });

  it("rejects a wrong schema version by name", () => {
    const doc = validBundle();
    doc.sgp_schema_version = 2;
    expect(() => parseSgpJson(JSON.stringify(doc))).toThrow(/sgp_schema_version/);
  });

  it("rejects a grid/num_neurons mismatch by name", () => {
    const doc = validBundle();
    (doc.som as Record<string, unknown>).num_neurons = 9;
    expect(() => parseSgpJson(JSON.stringify(doc))).toThrow(/som\.num_neurons/);
  });

  it("rejects an out-of-range BMU index naming the exact cell", () => {
    const doc = validBundle();
    const probes = doc.probes as Array<{ bmu: number[][][] }>;
    probes[0].bmu[0][1][0] = 99;
    expect(() => parseSgpJson(JSON.stringify(doc))).toThrow(/probes\[0\]\.bmu\[0\]\[1\]\[0\]/);
  });

  it("rejects a BMU map whose depth count disagrees with som.depth_steps", () => {
    const doc = validBundle();
    const probes = doc.probes as Array<{ bmu: number[][][] }>;
    probes[0].bmu = probes[0].bmu.slice(0, 1);
    expect(() => parseSgpJson(JSON.stringify(doc))).toThrow(/depth_steps/);
  });

  it("rejects an edge referencing a nonexistent neuron", () => {
    const doc = validBundle();
    (doc.som as { edges: unknown[] }).edges = [[0, 42, 0.5]];
    expect(() => parseSgpJson(JSON.stringify(doc))).toThrow(/som\.edges\[0\]/);
  });
});

describe("depthActivations", () => {
  it("builds a normalized per-depth histogram", () => {
    const b = parseSgpJson(JSON.stringify(validBundle()));
    const acts = depthActivations(b.probes[0].bmu, b.som.num_neurons);
    expect(acts).toHaveLength(2);
    // depth 0: neurons 0,1,1,2 over 4 voxels.
    expect(acts[0][0]).toBeCloseTo(0.25);
    expect(acts[0][1]).toBeCloseTo(0.5);
    expect(acts[0][2]).toBeCloseTo(0.25);
    expect(acts[0][5]).toBe(0);
    // each depth sums to 1.
    for (const a of acts) {
      let s = 0;
      for (const v of a) s += v;
      expect(s).toBeCloseTo(1);
    }
  });
});

describe("migrationCurve", () => {
  it("measures the fraction of voxels re-assigned between depths", () => {
    const b = parseSgpJson(JSON.stringify(validBundle()));
    // depth0 [[0,1],[1,2]] vs depth1 [[4,4],[4,5]] — all 4 change.
    expect(migrationCurve(b.probes[0].bmu)).toEqual([1]);
  });

  it("is zero for a static map", () => {
    const bmu = [
      [
        [1, 1],
        [1, 1],
      ],
      [
        [1, 1],
        [1, 1],
      ],
    ];
    expect(migrationCurve(bmu)).toEqual([0]);
  });
});

describe("neuronVoxels / bmuTrail (the sync maps)", () => {
  it("finds the voxels a neuron owns at a depth", () => {
    const b = parseSgpJson(JSON.stringify(validBundle()));
    expect(neuronVoxels(b.probes[0].bmu, 0, 1)).toEqual([
      [0, 1],
      [1, 0],
    ]);
    expect(neuronVoxels(b.probes[0].bmu, 1, 4)).toHaveLength(3);
    expect(neuronVoxels(b.probes[0].bmu, 0, 7)).toEqual([]);
  });

  it("traces one voxel's neuron across depths", () => {
    const b = parseSgpJson(JSON.stringify(validBundle()));
    expect(bmuTrail(b.probes[0].bmu, 0, 0)).toEqual([0, 4]);
    expect(bmuTrail(b.probes[0].bmu, 1, 1)).toEqual([2, 5]);
  });
});
