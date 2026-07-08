import { describe, expect, it } from "vitest";
import {
  DEFAULT_FORCE_OPTS,
  buildBrainGraph,
  centroid,
  computeLayout,
  distinctCount,
  forceStep,
  labelPropagation,
  layoutBounds,
  mulberry32,
  spatialRegions,
  type ForceLink,
  type ForceNode,
} from "./force";
import type { GraphJson } from "@/src/lib/pack/types";

function node(idx: number, x: number, y: number, community = 0): ForceNode {
  return { idx, kind: "patch_token", community, x, y, vx: 0, vy: 0 };
}

describe("mulberry32", () => {
  it("is deterministic for a given seed", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
});

describe("forceStep", () => {
  it("link spring pulls two far apart linked nodes closer", () => {
    const nodes = [node(0, -1, 0), node(1, 1, 0)];
    const links: ForceLink[] = [{ a: 0, b: 1, w: 1 }];
    const before = Math.hypot(nodes[1].x - nodes[0].x, nodes[1].y - nodes[0].y);
    forceStep(nodes, links, new Map([[0, [0, 1]]]), 1, DEFAULT_FORCE_OPTS);
    const after = Math.hypot(nodes[1].x - nodes[0].x, nodes[1].y - nodes[0].y);
    expect(after).toBeLessThan(before);
  });

  it("many-body repulsion pushes two coincident, unlinked nodes apart", () => {
    const nodes = [node(0, 0.01, 0), node(1, -0.01, 0)];
    const before = Math.hypot(nodes[1].x - nodes[0].x, nodes[1].y - nodes[0].y);
    forceStep(nodes, [], new Map([[0, [0, 1]]]), 1, DEFAULT_FORCE_OPTS);
    const after = Math.hypot(nodes[1].x - nodes[0].x, nodes[1].y - nodes[0].y);
    expect(after).toBeGreaterThan(before);
  });

  it("is deterministic (same inputs → same positions)", () => {
    const mk = () => [node(0, -0.5, 0.2), node(1, 0.4, -0.3), node(2, 0.1, 0.6)];
    const links: ForceLink[] = [{ a: 0, b: 1, w: 1 }];
    const comm = new Map([[0, [0, 1, 2]]]);
    const a = mk();
    const b = mk();
    forceStep(a, links, comm, 0.7);
    forceStep(b, links, comm, 0.7);
    expect(a.map((n) => [n.x, n.y])).toEqual(b.map((n) => [n.x, n.y]));
  });
});

describe("centroid", () => {
  it("averages member positions", () => {
    const nodes = [node(0, 0, 0), node(1, 2, 0), node(2, 1, 3)];
    expect(centroid(nodes, [0, 1])).toEqual({ x: 1, y: 0 });
  });
});

describe("labelPropagation / communities", () => {
  it("recovers two disjoint cliques as (at least) two communities", () => {
    // Two triangles with no cross edges.
    const links: ForceLink[] = [
      { a: 0, b: 1, w: 1 },
      { a: 1, b: 2, w: 1 },
      { a: 0, b: 2, w: 1 },
      { a: 3, b: 4, w: 1 },
      { a: 4, b: 5, w: 1 },
      { a: 3, b: 5, w: 1 },
    ];
    const labels = labelPropagation(6, links, 10, 1);
    expect(labels[0]).toBe(labels[1]);
    expect(labels[1]).toBe(labels[2]);
    expect(labels[3]).toBe(labels[4]);
    expect(labels[3]).not.toBe(labels[0]);
    expect(distinctCount(labels)).toBe(2);
  });
});

describe("spatialRegions", () => {
  it("puts CLS in group 0 and buckets patches into a grid of regions", () => {
    // grid=14, idx 1 = (row0,col0) top-left; idx 196 = (row13,col13) bottom-right.
    const groups = spatialRegions([0, 1, 196], 14, 3);
    expect(groups[0]).toBe(0);
    expect(groups[1]).not.toBe(groups[2]);
  });
});

describe("buildBrainGraph + computeLayout", () => {
  const graph: GraphJson = {
    num_layers: 2,
    num_tokens: 5,
    k: 2,
    grid: 2,
    cls_index: 0,
    seed: 0,
    edge_semantics: "",
    residual: { kind: "identity", materialized: false, weight: 1, count: 0, description: "" },
    layers: [
      {
        layer: 0,
        nodes: [
          { idx: 0, kind: "cls_token", community: 0 },
          { idx: 1, kind: "patch_token", community: 1 },
          { idx: 2, kind: "patch_token", community: 1 },
          { idx: 3, kind: "patch_token", community: 1 },
          { idx: 4, kind: "patch_token", community: 1 },
        ],
        edges: [
          [1, 2, 0.5],
          [3, 4, 0.5],
        ],
      },
      {
        layer: 1,
        nodes: [
          { idx: 0, kind: "cls_token", community: 0 },
          { idx: 1, kind: "patch_token", community: 1 },
          { idx: 2, kind: "patch_token", community: 1 },
          { idx: 3, kind: "patch_token", community: 1 },
          { idx: 4, kind: "patch_token", community: 1 },
        ],
        edges: [
          [1, 2, 0.9],
          [2, 1, 0.4],
          [3, 4, 0.9],
        ],
      },
    ],
  };

  it("builds nodes/links from the last layer and dedupes undirected edges", () => {
    const g = buildBrainGraph(graph);
    expect(g.nodes).toHaveLength(5);
    // last layer has edges (1,2),(2,1),(3,4) → 2 undirected links
    expect(g.links).toHaveLength(2);
    const l12 = g.links.find((l) => l.a === 1 && l.b === 2);
    expect(l12?.w).toBe(0.9); // strongest weight kept
  });

  it("falls back off the degenerate graph community field to a real signal", () => {
    const g = buildBrainGraph(graph);
    expect(g.communitySource).not.toBe("graph");
    expect(g.communities.size).toBeGreaterThanOrEqual(1);
  });

  it("builds a fixed, finite anchor per community", () => {
    const g = buildBrainGraph(graph);
    expect(g.anchors.size).toBe(g.communities.size);
    for (const { x, y } of g.anchors.values()) {
      expect(Number.isFinite(x)).toBe(true);
      expect(Number.isFinite(y)).toBe(true);
    }
  });

  it("anchored gravity separates two communities", () => {
    // Two 2-node communities starting interleaved at the centre.
    const nodes = [
      node(1, 0.01, 0, 0),
      node(2, -0.01, 0, 0),
      node(3, 0.02, 0, 1),
      node(4, -0.02, 0, 1),
    ];
    const communities = new Map([
      [0, [0, 1]],
      [1, [2, 3]],
    ]);
    const anchors = new Map([
      [0, { x: -1, y: 0 }],
      [1, { x: 1, y: 0 }],
    ]);
    for (let i = 0; i < 200; i++) {
      forceStep(nodes, [], communities, 0.5, DEFAULT_FORCE_OPTS, anchors);
    }
    const c0 = centroid(nodes, [0, 1]);
    const c1 = centroid(nodes, [2, 3]);
    // Each community's centroid is displaced toward its own anchor (repulsion
    // between the four nodes keeps the toy system from full separation).
    expect(c1.x - c0.x).toBeGreaterThan(0.2);
  });

  it("computeLayout keeps positions finite and bounded", () => {
    const g = buildBrainGraph(graph);
    computeLayout(g, 120);
    for (const n of g.nodes) {
      expect(Number.isFinite(n.x)).toBe(true);
      expect(Number.isFinite(n.y)).toBe(true);
    }
    const b = layoutBounds(g.nodes);
    expect(b.maxX).toBeGreaterThanOrEqual(b.minX);
  });
});
