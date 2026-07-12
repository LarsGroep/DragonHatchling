/**
 * lattice.ts — pure math for the SGP SOM lattice renderer. No three.js/WebGL
 * import so vitest exercises every rule headless (same discipline as
 * GaussianField/relief.ts and BrainView/force.ts).
 *
 * HONESTY RULE (SGP §1): node positions are a pure, shared mapping of the REAL
 * neuron lattice coordinates into world space — no force simulation, no fitted
 * layout. `latticeWorld` is that mapping; the renderer labels the axes. The
 * per-frame activation is a linear interpolation between two MEASURED per-depth
 * BMU histograms (fractional depth scrubbing), then EMA-blended for legibility —
 * a uniform temporal filter that encodes nothing.
 */

/** World half-extent of the lattice cube (matches the Gaussian field margin). */
export const LATTICE_EXTENT = 0.9;

/**
 * Map lattice coords `(z, y, x)` on a `(Gz, Gy, Gx)` grid into world space:
 * x → world X, y → world Z (ground plane), z (encoder-depth axis) → world Y
 * (up), each centered in [-EXTENT, +EXTENT]. Axes are labeled by the view;
 * "up = learned hierarchy" mirrors the UMT-ViT convention.
 */
export function latticeWorld(
  z: number,
  y: number,
  x: number,
  grid: readonly [number, number, number],
  extent: number = LATTICE_EXTENT,
): [number, number, number] {
  const [gz, gy, gx] = grid;
  const c = (v: number, g: number) => (g <= 1 ? 0 : (v / (g - 1)) * 2 * extent - extent);
  return [c(x, gx), c(z, gz), c(y, gy)];
}

/** Bracket a fractional depth `t` into (lo, hi, frac) over `steps` slices. */
export function bracketDepth(t: number, steps: number): { lo: number; hi: number; f: number } {
  const clamped = Math.max(0, Math.min(steps - 1, t));
  const lo = Math.floor(clamped);
  const hi = Math.min(steps - 1, lo + 1);
  return { lo, hi, f: clamped - lo };
}

/**
 * Per-neuron activation at fractional depth `t`: linear interpolation between
 * the two bracketing measured histograms, then normalized so the hottest
 * neuron is 1 (display exposure — relative structure is unchanged).
 */
export function activationAt(
  acts: readonly Float32Array[],
  t: number,
  out?: Float32Array,
): Float32Array {
  const K = acts[0]?.length ?? 0;
  const buf = out && out.length === K ? out : new Float32Array(K);
  if (acts.length === 0) return buf;
  const { lo, hi, f } = bracketDepth(t, acts.length);
  let max = 0;
  for (let k = 0; k < K; k++) {
    const v = acts[lo][k] * (1 - f) + acts[hi][k] * f;
    buf[k] = v;
    if (v > max) max = v;
  }
  if (max > 0) for (let k = 0; k < K; k++) buf[k] /= max;
  return buf;
}

/**
 * One EMA step of `current` toward `target` (frame-rate-independent):
 * `alpha = 1 − exp(−dt / tau)`. Mutates and returns `current`.
 */
export function emaToward(current: Float32Array, target: Float32Array, dt: number, tau = 0.18): Float32Array {
  const a = 1 - Math.exp(-Math.max(0, dt) / tau);
  for (let i = 0; i < current.length; i++) current[i] += (target[i] - current[i]) * a;
  return current;
}

/** Node display size from hits: sqrt-scaled into [minS, maxS]; dead pinned to minS. */
export function sizeForHits(hits: number, maxHits: number, minS = 0.16, maxS = 1.0): number {
  if (hits <= 0 || maxHits <= 0) return minS;
  return minS + (maxS - minS) * Math.sqrt(hits / maxHits);
}

/**
 * Normalize U-matrix values to [0,1] boundary strength (for edge dimming /
 * node rims). Constant fields map to 0 (no boundaries anywhere).
 */
export function boundaryStrength(umatrix: readonly number[]): Float32Array {
  const K = umatrix.length;
  const out = new Float32Array(K);
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of umatrix) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  if (span <= 0) return out;
  for (let k = 0; k < K; k++) out[k] = (umatrix[k] - lo) / span;
  return out;
}

/** Soft, desaturated community hues (mirrors GraphView's palette approach). */
export const COMMUNITY_HUES = [210, 258, 160, 32, 190, 340, 96, 280, 130, 300, 18, 174];

/** HSL → RGB in [0,1] (tiny local converter, no dependency). */
export function hslRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = ((h % 360) + 360) % 360 / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  const m = l - c / 2;
  let r = 0;
  let g = 0;
  let b = 0;
  if (hp < 1) [r, g, b] = [c, x, 0];
  else if (hp < 2) [r, g, b] = [x, c, 0];
  else if (hp < 3) [r, g, b] = [0, c, x];
  else if (hp < 4) [r, g, b] = [0, x, c];
  else if (hp < 5) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [r + m, g + m, b + m];
}

/** RGB in [0,1] for a community id (soft palette, wraps). */
export function communityRgb(community: number): [number, number, number] {
  const hue = COMMUNITY_HUES[((community % COMMUNITY_HUES.length) + COMMUNITY_HUES.length) % COMMUNITY_HUES.length];
  return hslRgb(hue, 0.52, 0.62);
}
