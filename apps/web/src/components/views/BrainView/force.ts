/**
 * force.ts — the pure, dependency-free force-directed layout for the Brain
 * (UX-VISION-2). We hand-roll a small d3-force-style integrator (velocity +
 * cooling alpha) rather than pull in d3-force: the whole graph is ≤197 nodes so
 * an O(n²) many-body repulsion is trivially fast, and keeping it pure lets
 * vitest exercise every rule headless (no canvas/WebGL).
 *
 * Forces per tick (all scaled by the cooling `alpha`):
 *   • many-body repulsion  — every node pushes every other apart (∝ 1/d²).
 *   • link springs         — the LAST layer's attention edges are the resting
 *                            "memory" topology; each pulls its endpoints toward
 *                            a target distance (∝ edge weight).
 *   • community gravity     — each node is drawn to its community centroid, so
 *                            clusters emerge and stay separated (Obsidian look).
 *   • centering            — a weak pull to the origin keeps the graph framed.
 *
 * Everything is deterministic given the seed, so the precomputed layout is
 * stable across reloads and the continuous low-alpha drift never wanders.
 */
import type { GraphJson } from "@/src/lib/pack/types";
import { tokenToPatch } from "@/src/lib/state/packIndex";

export interface ForceNode {
  /** Token index (0 = CLS). */
  idx: number;
  kind: string;
  /** Gravity-group id (see deriveCommunities). */
  community: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/** Undirected link between two node ARRAY positions (not token idxs). */
export interface ForceLink {
  a: number;
  b: number;
  w: number;
}

export interface ForceOpts {
  /** Repulsion magnitude. */
  charge: number;
  /** Spring rest length. */
  linkDist: number;
  /** Spring stiffness (0..1). */
  linkStrength: number;
  /** Pull toward origin (0..1). */
  centerStrength: number;
  /** Pull toward the node's community centroid (0..1). */
  communityStrength: number;
  /** Per-tick velocity retention (0..1); lower = more damping. */
  velocityDecay: number;
}

export const DEFAULT_FORCE_OPTS: ForceOpts = {
  charge: 0.008,
  linkDist: 0.12,
  linkStrength: 0.05,
  centerStrength: 0.008,
  communityStrength: 0.2,
  velocityDecay: 0.55,
};

/** Per-tick displacement clamp — keeps the explicit integrator from exploding. */
const MAX_SPEED = 0.12;

/** Springs that BRIDGE two communities pull this much weaker, so clusters can
 * breathe apart instead of collapsing into one hairball (Obsidian look). */
export const INTER_COMMUNITY_LINK = 0.08;

/** How far out the fixed community anchors sit (world units). */
export const ANCHOR_SPREAD = 2.1;

/** Deterministic tiny PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Label propagation over an undirected weighted graph — a lightweight community
 * detector. Deterministic (nodes visited in index order; ties broken by lowest
 * label). Returns a compacted community id per node.
 */
export function labelPropagation(
  n: number,
  links: ForceLink[],
  iterations = 8,
  seed = 1,
): number[] {
  const adj: Array<Array<{ j: number; w: number }>> = Array.from({ length: n }, () => []);
  for (const { a, b, w } of links) {
    if (a === b) continue;
    adj[a].push({ j: b, w });
    adj[b].push({ j: a, w });
  }
  const label = Array.from({ length: n }, (_, i) => i);
  const rng = mulberry32(seed);
  // Randomized-but-deterministic visit order for stability.
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  for (let it = 0; it < iterations; it++) {
    let changed = false;
    for (const i of order) {
      const nb = adj[i];
      if (!nb.length) continue;
      const tally = new Map<number, number>();
      for (const { j, w } of nb) tally.set(label[j], (tally.get(label[j]) ?? 0) + w);
      let best = label[i];
      let bestW = -Infinity;
      for (const [lab, w] of tally) {
        if (w > bestW || (w === bestW && lab < best)) {
          best = lab;
          bestW = w;
        }
      }
      if (best !== label[i]) {
        label[i] = best;
        changed = true;
      }
    }
    if (!changed) break;
  }
  return compactLabels(label);
}

function compactLabels(label: number[]): number[] {
  const remap = new Map<number, number>();
  return label.map((l) => {
    let id = remap.get(l);
    if (id === undefined) {
      id = remap.size;
      remap.set(l, id);
    }
    return id;
  });
}

/** Distinct-value count of a label array. */
export function distinctCount(labels: number[]): number {
  return new Set(labels).size;
}

/**
 * Spatial-region fallback: partition patch tokens into a `regions×regions` grid
 * over their real patch positions. Honest (derived from measured token→patch
 * geometry) and always yields well-separated gravity groups when the graph's
 * own community structure is degenerate. CLS gets its own group.
 */
export function spatialRegions(
  tokenIdxs: number[],
  grid: number,
  regions = 3,
): number[] {
  return tokenIdxs.map((idx) => {
    const pc = tokenToPatch(idx, grid);
    if (!pc) return 0; // CLS → group 0
    const rr = Math.min(regions - 1, Math.floor((pc[0] / grid) * regions));
    const cc = Math.min(regions - 1, Math.floor((pc[1] / grid) * regions));
    return 1 + rr * regions + cc;
  });
}

export interface BrainGraph {
  nodes: ForceNode[];
  links: ForceLink[];
  /** Node array positions grouped by community id. */
  communities: Map<number, number[]>;
  /** Fixed gravity anchor per community (from real member patch centroids). */
  anchors: Map<number, { x: number; y: number }>;
  /** Source used for the gravity groups (diagnostic). */
  communitySource: "graph" | "label-propagation" | "spatial";
}

/**
 * Build the Brain's resting topology from the LAST layer of graph.json: nodes
 * seeded near their real patch centroid (so the initial layout already echoes
 * the image before forces relax it into clusters), undirected deduped links
 * from the last-layer attention edges, and community gravity groups chosen from
 * the best available real signal (graph field → label propagation → spatial).
 */
export function buildBrainGraph(graph: GraphJson, seed = graph.seed ?? 0): BrainGraph {
  const last = graph.layers[graph.layers.length - 1];
  const grid = graph.grid;
  const rng = mulberry32(seed + 1);

  const idxToPos = new Map<number, number>();
  last.nodes.forEach((n, i) => idxToPos.set(n.idx, i));

  const nodes: ForceNode[] = last.nodes.map((n) => {
    const pc = tokenToPatch(n.idx, grid);
    // Normalize patch (row,col) to roughly [-0.9, 0.9]; CLS near center.
    const x = pc ? ((pc[1] + 0.5) / grid) * 1.8 - 0.9 : 0;
    const y = pc ? ((pc[0] + 0.5) / grid) * 1.8 - 0.9 : 0;
    const jitter = 0.06;
    return {
      idx: n.idx,
      kind: n.kind,
      community: 0,
      x: x + (rng() - 0.5) * jitter,
      y: y + (rng() - 0.5) * jitter,
      vx: 0,
      vy: 0,
    };
  });

  // Undirected, deduped links (keep the strongest weight per pair).
  const seen = new Map<string, ForceLink>();
  for (const [s, d, w] of last.edges) {
    const ai = idxToPos.get(s);
    const bi = idxToPos.get(d);
    if (ai === undefined || bi === undefined || ai === bi) continue;
    const a = Math.min(ai, bi);
    const b = Math.max(ai, bi);
    const key = `${a}-${b}`;
    const prev = seen.get(key);
    if (!prev || w > prev.w) seen.set(key, { a, b, w });
  }
  const links = [...seen.values()];

  // -- community gravity groups: pick the best real signal ------------------ //
  const graphComm = compactLabels(last.nodes.map((n) => n.community));
  let community: number[];
  let communitySource: BrainGraph["communitySource"];
  if (distinctCount(graphComm) >= 3) {
    community = graphComm;
    communitySource = "graph";
  } else {
    const lp = labelPropagation(nodes.length, links, 10, seed + 7);
    if (distinctCount(lp) >= 3 && distinctCount(lp) <= 20) {
      community = lp;
      communitySource = "label-propagation";
    } else {
      community = spatialRegions(nodes.map((n) => n.idx), grid, 3);
      community = compactLabels(community);
      communitySource = "spatial";
    }
  }

  const communities = new Map<number, number[]>();
  nodes.forEach((node, i) => {
    node.community = community[i];
    const arr = communities.get(node.community);
    if (arr) arr.push(i);
    else communities.set(node.community, [i]);
  });

  // Fixed gravity anchors: each community's REAL patch-centroid direction,
  // pushed out to ANCHOR_SPREAD so clusters separate instead of collapsing
  // onto their shared (moving) centre of mass. Honest geometry, stable layout.
  const anchors = new Map<number, { x: number; y: number }>();
  for (const [id, members] of communities) {
    let sx = 0;
    let sy = 0;
    let count = 0;
    for (const mi of members) {
      const pc = tokenToPatch(nodes[mi].idx, grid);
      if (!pc) continue;
      sx += ((pc[1] + 0.5) / grid) * 2 - 1;
      sy += ((pc[0] + 0.5) / grid) * 2 - 1;
      count++;
    }
    if (!count) {
      anchors.set(id, { x: 0, y: 0 }); // CLS-only group rests at the centre
      continue;
    }
    const cx = sx / count;
    const cy = sy / count;
    const len = Math.hypot(cx, cy);
    // Push the anchor outward along its centroid direction; central regions
    // stay near the middle (len≈0 → no artificial displacement).
    const gain = len > 1e-6 ? ANCHOR_SPREAD : 0;
    anchors.set(id, { x: cx * gain, y: cy * gain });
  }

  return { nodes, links, communities, anchors, communitySource };
}

/** Mean position of a set of node array-positions. */
export function centroid(nodes: ForceNode[], members: number[]): { x: number; y: number } {
  let x = 0;
  let y = 0;
  for (const i of members) {
    x += nodes[i].x;
    y += nodes[i].y;
  }
  const n = members.length || 1;
  return { x: x / n, y: y / n };
}

/**
 * One force-simulation tick (mutates node positions/velocities in place). Pure
 * and deterministic given its inputs; `alpha` is the cooling factor scaling
 * every force this tick.
 */
export function forceStep(
  nodes: ForceNode[],
  links: ForceLink[],
  communities: Map<number, number[]>,
  alpha: number,
  opts: ForceOpts = DEFAULT_FORCE_OPTS,
  anchors?: Map<number, { x: number; y: number }>,
): void {
  const n = nodes.length;
  const eps = 1e-4;

  // Many-body repulsion (O(n²)).
  for (let i = 0; i < n; i++) {
    const ni = nodes[i];
    for (let j = i + 1; j < n; j++) {
      const nj = nodes[j];
      let dx = ni.x - nj.x;
      let dy = ni.y - nj.y;
      let d2 = dx * dx + dy * dy;
      if (d2 < eps) {
        // Coincident: nudge apart deterministically.
        dx = (i - j) * 1e-3 + eps;
        dy = (j - i) * 1e-3 + eps;
        d2 = dx * dx + dy * dy;
      }
      const f = (opts.charge * alpha) / d2;
      const d = Math.sqrt(d2);
      const ux = dx / d;
      const uy = dy / d;
      ni.vx += ux * f;
      ni.vy += uy * f;
      nj.vx -= ux * f;
      nj.vy -= uy * f;
    }
  }

  // Link springs (bridges between communities pull far weaker, so the
  // clusters the gravity groups define can actually separate).
  for (const { a, b, w } of links) {
    const na = nodes[a];
    const nb = nodes[b];
    const dx = nb.x - na.x;
    const dy = nb.y - na.y;
    const d = Math.sqrt(dx * dx + dy * dy) || eps;
    const cross = na.community !== nb.community ? INTER_COMMUNITY_LINK : 1;
    const diff = ((d - opts.linkDist) / d) * opts.linkStrength * cross * alpha * (0.5 + w);
    na.vx += dx * diff;
    na.vy += dy * diff;
    nb.vx -= dx * diff;
    nb.vy -= dy * diff;
  }

  // Community gravity (toward the FIXED anchor when provided — this is what
  // pulls clusters apart; the moving-centroid fallback only keeps groups
  // cohesive) + weak centering.
  const cents = new Map<number, { x: number; y: number }>();
  for (const [id, members] of communities) {
    cents.set(id, anchors?.get(id) ?? centroid(nodes, members));
  }
  for (const node of nodes) {
    const c = cents.get(node.community);
    if (c) {
      node.vx += (c.x - node.x) * opts.communityStrength * alpha;
      node.vy += (c.y - node.y) * opts.communityStrength * alpha;
    }
    node.vx += -node.x * opts.centerStrength * alpha;
    node.vy += -node.y * opts.centerStrength * alpha;
  }

  // Integrate + damp (with a per-tick speed clamp for numerical stability).
  for (const node of nodes) {
    const sp = Math.hypot(node.vx, node.vy);
    if (sp > MAX_SPEED) {
      const s = MAX_SPEED / sp;
      node.vx *= s;
      node.vy *= s;
    }
    node.x += node.vx;
    node.y += node.vy;
    node.vx *= opts.velocityDecay;
    node.vy *= opts.velocityDecay;
  }
  // Pin the centre of mass to the origin: floating-point asymmetries otherwise
  // let the whole cluster slowly translate away (rendering it off-canvas).
  recenter(nodes);
}

/** Translate all nodes so their centre of mass sits at the origin. */
export function recenter(nodes: ForceNode[]): void {
  let cx = 0;
  let cy = 0;
  for (const n of nodes) {
    cx += n.x;
    cy += n.y;
  }
  const inv = 1 / (nodes.length || 1);
  cx *= inv;
  cy *= inv;
  for (const n of nodes) {
    n.x -= cx;
    n.y -= cy;
  }
}

/**
 * Run the layout to rest: `iterations` cooling ticks from `alphaStart` decaying
 * geometrically. Mutates and returns `graph.nodes` (positions relaxed).
 */
export function computeLayout(
  graph: BrainGraph,
  iterations = 300,
  alphaStart = 1,
  alphaMin = 0.02,
  opts: ForceOpts = DEFAULT_FORCE_OPTS,
): ForceNode[] {
  const decay = Math.pow(alphaMin / alphaStart, 1 / Math.max(1, iterations));
  let alpha = alphaStart;
  for (let i = 0; i < iterations; i++) {
    forceStep(graph.nodes, graph.links, graph.communities, alpha, opts, graph.anchors);
    alpha *= decay;
  }
  return graph.nodes;
}

/** Axis-aligned bounds of the current node positions (for the fit transform). */
export function layoutBounds(nodes: ForceNode[]): {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
} {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const n of nodes) {
    minX = Math.min(minX, n.x);
    minY = Math.min(minY, n.y);
    maxX = Math.max(maxX, n.x);
    maxY = Math.max(maxY, n.y);
  }
  if (!Number.isFinite(minX)) return { minX: -1, minY: -1, maxX: 1, maxY: 1 };
  return { minX, minY, maxX, maxY };
}
