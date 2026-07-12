/**
 * SGP web-bundle schema (`sgp_<dataset>.json`, version 1) — typed model plus a
 * versioned, defensive validator, mirroring `lib/umtvit.ts`. The SGP Kaggle
 * notebook (`experiments/umtvit/notebooks/kaggle_umtvit_sgp.ipynb`) writes
 * exactly this shape via the numpy-only `vitreous.som` core; both sides
 * implement the same contract (`docs/SGP-ARCHITECTURE.md` §4). Validation is
 * strict and names the offending field so a malformed drop-in shows an
 * actionable message instead of crashing a renderer.
 *
 * Also home to the pure BMU derivations the views share (per-depth activation
 * histograms, migration curve, neuron ↔ voxel maps) — kept here, headless and
 * dependency-free, so vitest exercises every rule the renderers rely on.
 */

export const SGP_SCHEMA_VERSION = 1 as const;

// ── types (mirror of som.json + the bundle wrapper) ────────────────────────

export interface SgpNode {
  idx: number;
  /** Lattice position [z, y, x] — the REAL neuron coordinates (honesty rule). */
  grid: [number, number, number];
  /** BMU win count over the probe/training pass; 0 ⇒ dead. */
  hits: number;
  /** Mean weight-space distance to lattice neighbours (cluster-boundary height). */
  umatrix: number;
  community: number;
  dead: boolean;
}

/** [a, b, similarity] — a and b are neuron idxs, ALWAYS lattice neighbours. */
export type SgpEdge = [number, number, number];

export interface SgpSom {
  provider: "som";
  grid: [number, number, number]; // (Gz, Gy, Gx)
  num_neurons: number;
  depth_steps: number; // Z of the BMU maps = encoder depth
  depth_semantics: string;
  volume_grid: [number, number] | null; // (H', W') of the BMU maps
  adjacency: string;
  nodes: SgpNode[];
  edges: SgpEdge[];
  edge_semantics: string;
  communities: { method: string; k: number; seed: number };
  dead_neurons: number;
  provenance: Record<string, unknown>;
}

export interface SgpProbe {
  /** Eval-dataset index of this probe image (provenance only). */
  index: number;
  /** Base64 PNG thumbnail of the input image. */
  thumb_png_b64: string;
  /** [Z][H'][W'] BMU neuron index per voxel. */
  bmu: number[][][];
}

export interface SgpBundle {
  version: number;
  dataset: string;
  som: SgpSom;
  probes: SgpProbe[];
  provenance: Record<string, unknown>;
}

/** Thrown on any schema violation; `message` always names the failing field. */
export class SgpValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SgpValidationError";
  }
}

// ── low-level field guards ─────────────────────────────────────────────────

function fail(path: string, expected: string): never {
  throw new SgpValidationError(`${path}: expected ${expected}`);
}

function obj(v: unknown, path: string): Record<string, unknown> {
  if (v === null || typeof v !== "object" || Array.isArray(v)) fail(path, "an object");
  return v as Record<string, unknown>;
}

function num(v: unknown, path: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) fail(path, "a finite number");
  return v;
}

function int(v: unknown, path: string): number {
  const n = num(v, path);
  if (!Number.isInteger(n)) fail(path, "an integer");
  return n;
}

function str(v: unknown, path: string): string {
  if (typeof v !== "string") fail(path, "a string");
  return v;
}

function bool(v: unknown, path: string): boolean {
  if (typeof v !== "boolean") fail(path, "a boolean");
  return v;
}

function arr(v: unknown, path: string): unknown[] {
  if (!Array.isArray(v)) fail(path, "an array");
  return v;
}

function vec3(v: unknown, path: string): [number, number, number] {
  const a = arr(v, path);
  if (a.length !== 3) fail(path, "3 values");
  return [int(a[0], `${path}[0]`), int(a[1], `${path}[1]`), int(a[2], `${path}[2]`)];
}

// ── section validators ─────────────────────────────────────────────────────

function parseNode(v: unknown, path: string, K: number): SgpNode {
  const o = obj(v, path);
  const idx = int(o.idx, `${path}.idx`);
  if (idx < 0 || idx >= K) fail(`${path}.idx`, `an index in [0, ${K})`);
  return {
    idx,
    grid: vec3(o.grid, `${path}.grid`),
    hits: int(o.hits, `${path}.hits`),
    umatrix: num(o.umatrix, `${path}.umatrix`),
    community: int(o.community, `${path}.community`),
    dead: bool(o.dead, `${path}.dead`),
  };
}

function parseSom(v: unknown): SgpSom {
  const o = obj(v, "som");
  if (o.provider !== "som") fail("som.provider", '"som"');
  const grid = vec3(o.grid, "som.grid");
  const K = int(o.num_neurons, "som.num_neurons");
  if (grid[0] * grid[1] * grid[2] !== K) {
    throw new SgpValidationError(
      `som.num_neurons: ${K} does not match grid ${grid.join("x")} (${grid[0] * grid[1] * grid[2]})`,
    );
  }
  const nodes = arr(o.nodes, "som.nodes").map((n, i) => parseNode(n, `som.nodes[${i}]`, K));
  if (nodes.length !== K) fail("som.nodes", `${K} entries`);

  const edges: SgpEdge[] = arr(o.edges, "som.edges").map((e, i) => {
    const t = arr(e, `som.edges[${i}]`);
    if (t.length !== 3) fail(`som.edges[${i}]`, "[a, b, weight]");
    const a = int(t[0], `som.edges[${i}][0]`);
    const b = int(t[1], `som.edges[${i}][1]`);
    const w = num(t[2], `som.edges[${i}][2]`);
    if (a < 0 || a >= K || b < 0 || b >= K) fail(`som.edges[${i}]`, `neuron idxs in [0, ${K})`);
    return [a, b, w];
  });

  const comm = obj(o.communities, "som.communities");
  const vg = o.volume_grid;
  let volume_grid: [number, number] | null = null;
  if (vg !== null && vg !== undefined) {
    const a = arr(vg, "som.volume_grid");
    if (a.length !== 2) fail("som.volume_grid", "2 values [H', W']");
    volume_grid = [int(a[0], "som.volume_grid[0]"), int(a[1], "som.volume_grid[1]")];
  }

  return {
    provider: "som",
    grid,
    num_neurons: K,
    depth_steps: int(o.depth_steps, "som.depth_steps"),
    depth_semantics: str(o.depth_semantics, "som.depth_semantics"),
    volume_grid,
    adjacency: str(o.adjacency, "som.adjacency"),
    nodes,
    edges,
    edge_semantics: str(o.edge_semantics, "som.edge_semantics"),
    communities: {
      method: str(comm.method, "som.communities.method"),
      k: int(comm.k, "som.communities.k"),
      seed: int(comm.seed, "som.communities.seed"),
    },
    dead_neurons: int(o.dead_neurons, "som.dead_neurons"),
    provenance: obj(o.provenance ?? {}, "som.provenance"),
  };
}

function parseProbe(v: unknown, path: string, som: SgpSom): SgpProbe {
  const o = obj(v, path);
  const K = som.num_neurons;
  const bmuRaw = arr(o.bmu, `${path}.bmu`);
  if (bmuRaw.length !== som.depth_steps) {
    throw new SgpValidationError(
      `${path}.bmu: ${bmuRaw.length} depth slices, expected som.depth_steps=${som.depth_steps}`,
    );
  }
  let shape = "";
  const bmu = bmuRaw.map((slice, z) => {
    const rows = arr(slice, `${path}.bmu[${z}]`);
    const grid = rows.map((row, y) => {
      const cells = arr(row, `${path}.bmu[${z}][${y}]`);
      return cells.map((c, x) => {
        const k = int(c, `${path}.bmu[${z}][${y}][${x}]`);
        if (k < 0 || k >= K) fail(`${path}.bmu[${z}][${y}][${x}]`, `a neuron idx in [0, ${K})`);
        return k;
      });
    });
    const s = `${grid.length}x${grid[0]?.length ?? 0}`;
    if (shape === "") shape = s;
    else if (s !== shape) {
      throw new SgpValidationError(`${path}.bmu[${z}]: ragged slice (${s}, expected ${shape})`);
    }
    return grid;
  });
  return {
    index: int(o.index, `${path}.index`),
    thumb_png_b64: str(o.thumb_png_b64, `${path}.thumb_png_b64`),
    bmu,
  };
}

/** Parse + validate an SGP web bundle. Throws `SgpValidationError` naming the field. */
export function parseSgpJson(text: string): SgpBundle {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new SgpValidationError(`not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }
  const o = obj(raw, "$");
  const version = int(o.sgp_schema_version, "sgp_schema_version");
  if (version !== SGP_SCHEMA_VERSION) {
    throw new SgpValidationError(
      `sgp_schema_version: ${version} not supported (expected ${SGP_SCHEMA_VERSION})`,
    );
  }
  const som = parseSom(o.som);
  const probes = arr(o.probes, "probes").map((p, i) => parseProbe(p, `probes[${i}]`, som));
  return {
    version,
    dataset: str(o.dataset, "dataset"),
    som,
    probes,
    provenance: obj(o.provenance ?? {}, "provenance"),
  };
}

// ── pure BMU derivations (shared by the views; headless-tested) ────────────

/**
 * Per-depth neuron activation from a probe's BMU map: `out[z][k]` = fraction of
 * this image's voxels whose BMU at depth `z` is neuron `k` (each depth slice
 * sums to 1). This is the measured signal that lights the lattice — the analog
 * of tokens.bin norms in the ViT brain.
 */
export function depthActivations(bmu: number[][][], numNeurons: number): Float32Array[] {
  return bmu.map((slice) => {
    const hist = new Float32Array(numNeurons);
    let n = 0;
    for (const row of slice)
      for (const k of row) {
        hist[k] += 1;
        n += 1;
      }
    if (n > 0) for (let k = 0; k < numNeurons; k++) hist[k] /= n;
    return hist;
  });
}

/**
 * Fraction of voxels whose BMU changes between consecutive depths —
 * `out[i]` covers the transition `z = i → i+1`. Length `Z - 1`.
 */
export function migrationCurve(bmu: number[][][]): number[] {
  const out: number[] = [];
  for (let z = 1; z < bmu.length; z++) {
    const a = bmu[z - 1];
    const b = bmu[z];
    let changed = 0;
    let n = 0;
    for (let y = 0; y < a.length; y++)
      for (let x = 0; x < a[y].length; x++) {
        if (a[y][x] !== b[y][x]) changed += 1;
        n += 1;
      }
    out.push(n > 0 ? changed / n : 0);
  }
  return out;
}

/** Voxel cells `[row, col]` mapped to neuron `k` at depth `z` (may be empty). */
export function neuronVoxels(bmu: number[][][], z: number, k: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  const slice = bmu[z];
  if (!slice) return out;
  for (let y = 0; y < slice.length; y++)
    for (let x = 0; x < slice[y].length; x++) if (slice[y][x] === k) out.push([y, x]);
  return out;
}

/** One voxel's BMU trail across all depths: `out[z]` = neuron idx at depth z. */
export function bmuTrail(bmu: number[][][], row: number, col: number): number[] {
  return bmu.map((slice) => slice[row]?.[col] ?? -1);
}
