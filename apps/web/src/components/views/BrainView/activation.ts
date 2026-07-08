/**
 * activation.ts — the pure logic that turns the loop clock into what the Brain
 * shows (UX-VISION-2 §Inference). Node activation at timeline layer t is the
 * token's L2 activation magnitude at that layer (tokens.bin norms), normalized
 * per layer; frames blend it with an exponential moving average so brightening
 * and fading are gradual, never flashing. As t→12 nodes with high final Chefer
 * attribution blend toward green ("evidence confirmed"). Community labels fade
 * in only while a community's mean activation crosses a threshold.
 *
 * All functions are canvas/WebGL-free so vitest can exercise them headless.
 */
import { tokenToPatch } from "@/src/lib/state/packIndex";

/** L2 norm of every token's embedding at timeline step `step`. */
export function tokenNorms(
  data: Float32Array,
  step: number,
  tokens: number,
  dim: number,
): Float32Array {
  const out = new Float32Array(tokens);
  const base = step * tokens * dim;
  for (let n = 0; n < tokens; n++) {
    let s = 0;
    const o = base + n * dim;
    for (let c = 0; c < dim; c++) {
      const v = data[o + c];
      s += v * v;
    }
    out[n] = Math.sqrt(s);
  }
  return out;
}

/**
 * Min-max normalize to [0,1]. `excludeIdx` (e.g. CLS) is left out of the range
 * computation so a single outlier token doesn't flatten everything else, but is
 * still normalized against that range and clamped.
 */
export function normalizeUnit(values: Float32Array, excludeIdx = -1): Float32Array {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (i === excludeIdx) continue;
    lo = Math.min(lo, values[i]);
    hi = Math.max(hi, values[i]);
  }
  const out = new Float32Array(values.length);
  const span = hi - lo;
  if (span <= 0) return out;
  for (let i = 0; i < values.length; i++) {
    out[i] = Math.max(0, Math.min(1, (values[i] - lo) / span));
  }
  return out;
}

/** Scalar exponential moving average: move `prev` a fraction `alpha` to `target`. */
export function ema(prev: number, target: number, alpha: number): number {
  return prev + (target - prev) * alpha;
}

/** In-place array EMA: `prev[i] += (target[i]-prev[i])*alpha`. */
export function emaInto(prev: Float32Array, target: Float32Array, alpha: number): void {
  for (let i = 0; i < prev.length; i++) prev[i] = prev[i] + (target[i] - prev[i]) * alpha;
}

/**
 * The activation value at/above which a node is "hot" — chosen so that only the
 * top `fraction` of nodes qualify (caps simultaneous hot nodes). Returns a
 * threshold in the same units as `values`.
 */
export function hotThreshold(values: Float32Array, fraction: number): number {
  const n = values.length;
  if (n === 0) return Infinity;
  const sorted = Float32Array.from(values).sort();
  const k = Math.max(0, Math.min(n - 1, Math.floor((1 - fraction) * n)));
  return sorted[k];
}

/**
 * Quadratic-bezier control point for a soft curved edge: the midpoint pushed
 * perpendicular to the segment by `curvature × length` (consistent side so
 * bundles fan the same way).
 */
export function controlPoint(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  curvature: number,
): [number, number] {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  // Perpendicular unit vector (rotate the segment 90°).
  const px = -dy / len;
  const py = dx / len;
  const off = curvature * len;
  return [mx + px * off, my + py * off];
}

/** Point on the quadratic bezier P0→C→P1 at parameter u∈[0,1]. */
export function quadPoint(
  x1: number,
  y1: number,
  cx: number,
  cy: number,
  x2: number,
  y2: number,
  u: number,
): [number, number] {
  const iu = 1 - u;
  const a = iu * iu;
  const b = 2 * iu * u;
  const c = u * u;
  return [a * x1 + b * cx + c * x2, a * y1 + b * cy + c * y2];
}

/** Mean of `values` over the given member indices. */
export function meanOver(values: Float32Array, members: number[]): number {
  if (!members.length) return 0;
  let s = 0;
  for (const i of members) s += values[i];
  return s / members.length;
}

/**
 * A community label may show only while its mean activation is above
 * `threshold`; `showLabel` includes a small hysteresis so it doesn't flicker at
 * the boundary. Returns the next visibility given the previous one.
 */
export function shouldShowLabel(
  meanActivation: number,
  prevVisible: boolean,
  threshold: number,
  hysteresis = 0.06,
): boolean {
  return prevVisible
    ? meanActivation > threshold - hysteresis
    : meanActivation > threshold + hysteresis;
}

/**
 * Honest region descriptor from a community's member patch cells: an English
 * name for where in the image those patches sit (e.g. "upper-left patches").
 * Never fabricates semantics (§7 honesty rule) — it is purely the geometric
 * centroid of REAL token→patch positions.
 */
export function regionDescriptor(tokenIdxs: number[], grid: number): string {
  let sr = 0;
  let sc = 0;
  let count = 0;
  for (const idx of tokenIdxs) {
    const pc = tokenToPatch(idx, grid);
    if (!pc) continue;
    sr += pc[0];
    sc += pc[1];
    count++;
  }
  if (!count) return "attention hub";
  const r = sr / count / grid;
  const c = sc / count / grid;
  const vert = r < 0.34 ? "upper" : r > 0.66 ? "lower" : "";
  const horiz = c < 0.34 ? "left" : c > 0.66 ? "right" : "";
  const where = [vert, horiz].filter(Boolean).join("-");
  return where ? `${where} patches` : "central patches";
}

/**
 * Community label per the honesty rule (UX-VISION-2 §Labels): with a concept
 * dictionary present but no human-readable class stats, name the community by
 * its dominant firing SAE feature id; otherwise fall back to the geometric
 * region descriptor. Both are real-data-derived — no invented semantics.
 */
export function communityLabel(
  tokenIdxs: number[],
  grid: number,
  top1FeatureByToken: Record<number, number> | null,
): string {
  const region = regionDescriptor(tokenIdxs, grid);
  if (top1FeatureByToken) {
    const tally = new Map<number, number>();
    for (const idx of tokenIdxs) {
      const f = top1FeatureByToken[idx];
      if (f === undefined) continue;
      tally.set(f, (tally.get(f) ?? 0) + 1);
    }
    let bestFeat = -1;
    let bestN = 0;
    for (const [f, n] of tally) {
      if (n > bestN) {
        bestN = n;
        bestFeat = f;
      }
    }
    // Require a real plurality (>1 firing token) before showing a feature id.
    if (bestFeat >= 0 && bestN > 1) return `feature #${bestFeat} · ${region}`;
  }
  return region;
}
