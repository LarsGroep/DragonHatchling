/**
 * UMT-ViT web-bundle schema (`umtvit_web.json`, version 1) — typed model plus a
 * versioned, defensive validator. The notebook export cell in
 * `experiments/umtvit/notebooks/kaggle_umtvit.ipynb` writes exactly this shape;
 * both sides implement the same contract. Validation is strict and names the
 * offending field so a malformed drop-in shows an actionable message rather
 * than crashing a panel deep in a render.
 */

export const UMTVIT_SCHEMA_VERSION = 1 as const;

export interface UmtvitDataset {
  name: string;
  image_size: number;
  augmentation: string;
  num_classes: number;
}

export interface UmtvitModel {
  dim: number;
  depth: number;
  volume_grid: number;
  volume_channels: number;
  som_grid: number[];
  cross_attention: string;
  params_millions: number;
}

export interface UmtvitMetrics {
  linear_probe: number | null;
  knn: number | null;
  chance: number | null;
  som_quantization_error: number | null;
  som_topographic_error: number | null;
  som_dead_fraction: number | null;
  trustworthiness: number | null;
}

export type UmtvitSeriesKey = "total" | "ntxent" | "som" | "smooth" | "order";

export interface UmtvitHistory {
  steps: number[];
  series: Record<UmtvitSeriesKey, number[]>;
}

/** Latent cube as [L][H][W] channel-mean, per-slice min-max normalized. */
export type UmtvitCube = number[][][];

export interface UmtvitProbe {
  label: string | null;
  image_png_b64: string;
  cube_initial: UmtvitCube;
  cube_final: UmtvitCube;
}

export interface UmtvitSom {
  /** [gz, gy, gx] neuron-grid shape. */
  grid: [number, number, number];
  epochs: number[];
  /** [epoch][gz][gy][gx] U-matrix. */
  umatrix: number[][][][];
  /** [gz][gy][gx] final BMU hit counts. */
  hits_final: number[][][];
}

export interface UmtvitEmbeddings {
  epochs: number[];
  /** [epoch][N][2] fixed-PCA coordinates. */
  coords: number[][][];
  labels: number[] | null;
}

export interface UmtvitBundle {
  version: number;
  dataset: UmtvitDataset;
  model: UmtvitModel;
  metrics: UmtvitMetrics;
  spectral_centroids: (number | null)[];
  classes: string[] | null;
  history: UmtvitHistory;
  probes: UmtvitProbe[];
  som: UmtvitSom;
  embeddings: UmtvitEmbeddings;
}

/** Thrown on any schema violation; `message` always names the failing field. */
export class UmtvitValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UmtvitValidationError";
  }
}

// ── low-level field guards (each names its path on failure) ───────────────

function fail(path: string, expected: string): never {
  throw new UmtvitValidationError(`${path}: expected ${expected}`);
}

function obj(v: unknown, path: string): Record<string, unknown> {
  if (v === null || typeof v !== "object" || Array.isArray(v)) fail(path, "an object");
  return v as Record<string, unknown>;
}

function num(v: unknown, path: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) fail(path, "a finite number");
  return v as number;
}

function numOrNull(v: unknown, path: string): number | null {
  if (v === null) return null;
  if (typeof v !== "number" || !Number.isFinite(v)) fail(path, "a finite number or null");
  return v;
}

function str(v: unknown, path: string): string {
  if (typeof v !== "string") fail(path, "a string");
  return v;
}

function strOrNull(v: unknown, path: string): string | null {
  if (v === null) return null;
  if (typeof v !== "string") fail(path, "a string or null");
  return v;
}

function arr(v: unknown, path: string): unknown[] {
  if (!Array.isArray(v)) fail(path, "an array");
  return v;
}

function numArray(v: unknown, path: string): number[] {
  return arr(v, path).map((x, i) => num(x, `${path}[${i}]`));
}

/** [rows][cols] rectangular numeric grid (ragged rows rejected, naming the row). */
function numGrid2d(v: unknown, path: string): number[][] {
  const rows = arr(v, path);
  let width = -1;
  return rows.map((row, i) => {
    const r = numArray(row, `${path}[${i}]`);
    if (width === -1) width = r.length;
    else if (r.length !== width) {
      throw new UmtvitValidationError(
        `${path}[${i}]: ragged array (row width ${r.length}, expected ${width})`,
      );
    }
    return r;
  });
}

/** [d0][d1][d2] rectangular numeric cube; ragged slices rejected by name. */
function numCube3d(v: unknown, path: string): number[][][] {
  const slices = arr(v, path);
  let shape = "";
  return slices.map((slice, i) => {
    const g = numGrid2d(slice, `${path}[${i}]`);
    const s = `${g.length}x${g[0]?.length ?? 0}`;
    if (shape === "") shape = s;
    else if (s !== shape) {
      throw new UmtvitValidationError(
        `${path}[${i}]: ragged array (slice shape ${s}, expected ${shape})`,
      );
    }
    return g;
  });
}

// ── section validators ────────────────────────────────────────────────────

function parseDataset(v: unknown): UmtvitDataset {
  const o = obj(v, "dataset");
  return {
    name: str(o.name, "dataset.name"),
    image_size: num(o.image_size, "dataset.image_size"),
    augmentation: str(o.augmentation, "dataset.augmentation"),
    num_classes: num(o.num_classes, "dataset.num_classes"),
  };
}

function parseModel(v: unknown): UmtvitModel {
  const o = obj(v, "model");
  const grid = numArray(o.som_grid, "model.som_grid");
  if (grid.length !== 3) fail("model.som_grid", "3 values [gz, gy, gx]");
  return {
    dim: num(o.dim, "model.dim"),
    depth: num(o.depth, "model.depth"),
    volume_grid: num(o.volume_grid, "model.volume_grid"),
    volume_channels: num(o.volume_channels, "model.volume_channels"),
    som_grid: grid,
    cross_attention: str(o.cross_attention, "model.cross_attention"),
    params_millions: num(o.params_millions, "model.params_millions"),
  };
}

function parseMetrics(v: unknown): UmtvitMetrics {
  const o = obj(v, "metrics");
  const f = (k: keyof UmtvitMetrics) => numOrNull(o[k], `metrics.${k}`);
  return {
    linear_probe: f("linear_probe"),
    knn: f("knn"),
    chance: f("chance"),
    som_quantization_error: f("som_quantization_error"),
    som_topographic_error: f("som_topographic_error"),
    som_dead_fraction: f("som_dead_fraction"),
    trustworthiness: f("trustworthiness"),
  };
}

const SERIES_KEYS: UmtvitSeriesKey[] = ["total", "ntxent", "som", "smooth", "order"];

function parseHistory(v: unknown): UmtvitHistory {
  const o = obj(v, "history");
  const steps = numArray(o.steps, "history.steps");
  const s = obj(o.series, "history.series");
  const series = {} as Record<UmtvitSeriesKey, number[]>;
  for (const k of SERIES_KEYS) {
    const seq = numArray(s[k], `history.series.${k}`);
    if (seq.length !== 0 && seq.length !== steps.length) {
      throw new UmtvitValidationError(
        `history.series.${k}: length ${seq.length} does not match history.steps length ${steps.length}`,
      );
    }
    series[k] = seq;
  }
  return { steps, series };
}

function parseProbes(v: unknown): UmtvitProbe[] {
  return arr(v, "probes").map((p, i) => {
    const o = obj(p, `probes[${i}]`);
    return {
      label: strOrNull(o.label, `probes[${i}].label`),
      image_png_b64: str(o.image_png_b64, `probes[${i}].image_png_b64`),
      cube_initial: numCube3d(o.cube_initial, `probes[${i}].cube_initial`),
      cube_final: numCube3d(o.cube_final, `probes[${i}].cube_final`),
    };
  });
}

function parseSom(v: unknown): UmtvitSom {
  const o = obj(v, "som");
  const grid = numArray(o.grid, "som.grid");
  if (grid.length !== 3) fail("som.grid", "3 values [gz, gy, gx]");
  const umatrix = arr(o.umatrix, "som.umatrix").map((u, i) =>
    numCube3d(u, `som.umatrix[${i}]`),
  );
  return {
    grid: grid as [number, number, number],
    epochs: numArray(o.epochs, "som.epochs"),
    umatrix,
    hits_final: numCube3d(o.hits_final, "som.hits_final"),
  };
}

function parseEmbeddings(v: unknown): UmtvitEmbeddings {
  const o = obj(v, "embeddings");
  const coords = arr(o.coords, "embeddings.coords").map((frame, i) => {
    const g = numGrid2d(frame, `embeddings.coords[${i}]`);
    g.forEach((pt, j) => {
      if (pt.length !== 2) fail(`embeddings.coords[${i}][${j}]`, "a 2-value [x, y] point");
    });
    return g;
  });
  const labels = o.labels === null ? null : numArray(o.labels, "embeddings.labels");
  return {
    epochs: numArray(o.epochs, "embeddings.epochs"),
    coords,
    labels,
  };
}

/**
 * Validate + narrow an untrusted JSON value to a {@link UmtvitBundle}.
 * Throws {@link UmtvitValidationError} (message names the field) on any breach,
 * including an unsupported schema version.
 */
export function parseUmtvitBundle(raw: unknown): UmtvitBundle {
  const o = obj(raw, "bundle");
  const version = num(o.version, "version");
  if (version !== UMTVIT_SCHEMA_VERSION) {
    throw new UmtvitValidationError(
      `version: unsupported bundle version ${version} (this viewer supports version ${UMTVIT_SCHEMA_VERSION})`,
    );
  }
  return {
    version,
    dataset: parseDataset(o.dataset),
    model: parseModel(o.model),
    metrics: parseMetrics(o.metrics),
    spectral_centroids: arr(o.spectral_centroids, "spectral_centroids").map((x, i) =>
      numOrNull(x, `spectral_centroids[${i}]`),
    ),
    classes: o.classes === null ? null : arr(o.classes, "classes").map((c, i) => str(c, `classes[${i}]`)),
    history: parseHistory(o.history),
    probes: parseProbes(o.probes),
    som: parseSom(o.som),
    embeddings: parseEmbeddings(o.embeddings),
  };
}

/** Parse raw JSON text; wraps JSON syntax errors as validation errors. */
export function parseUmtvitJson(text: string): UmtvitBundle {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch (e) {
    throw new UmtvitValidationError(
      `not valid JSON: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
  return parseUmtvitBundle(data);
}
