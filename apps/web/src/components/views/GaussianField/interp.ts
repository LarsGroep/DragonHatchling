/**
 * Pure math for the Gaussian Feature Field (§7, §12) — deliberately free of any
 * React / WebGL / three.js import so vitest exercises it headless. Two concerns:
 *
 *   1. CHANNEL INTERPOLATION — the pack ships `gaussians.bin` as [S, N, C] fp16
 *      (S = L+1 timeline steps, N = 197 tokens, C = 12 channels in the frozen
 *      order below). At a fractional timeline position `t ∈ [0, L]` every channel
 *      is LERP'd between floor(t) and ceil(t); `theta` uses shortest-path angular
 *      interpolation (frozen range [-π, π]), everything else is plain linear.
 *      The GPU vertex shader does the identical interpolation for rendering; this
 *      CPU path feeds hit-testing and is the unit-tested reference.
 *
 *   2. HIT-TEST MAPPING — pointer → token idx via an ANALYTIC ellipse test
 *      (chosen over an offscreen ID buffer: no GPU readback, no async, and it is
 *      a pure function we can unit-test). A pointer at data coords (px,py) hits
 *      the token whose anisotropic Gaussian has the smallest Mahalanobis radius,
 *      provided that radius is within `sigma` (default 3σ, matching the quad the
 *      shader draws). CLS (token 0) lives off-grid in a gutter and is excluded.
 */

/** Frozen channel order of gaussians.bin (DECISION-LOG M3; manifest meta.channels). */
export const GAUSSIAN_CHANNELS = [
  "x",
  "y",
  "rx",
  "ry",
  "theta",
  "r",
  "g",
  "b",
  "opacity",
  "glow",
  "halo",
  "activation_raw",
] as const;

export const CH = {
  x: 0,
  y: 1,
  rx: 2,
  ry: 3,
  theta: 4,
  r: 5,
  g: 6,
  b: 7,
  opacity: 8,
  glow: 9,
  halo: 10,
  activation: 11,
} as const;

export const CHANNEL_COUNT = 12;

/** Field-layout constants shared by the renderer and the CPU hit-test. */
export const CLS_INDEX = 0;
/** Fraction of the square the on-image field occupies; the border is the gutter. */
export const FIELD_MARGIN = 0.9;
/** Quad half-extent in σ units — the falloff the shader draws and we hit-test to. */
export const SIGMA_EXTENT = 3.0;

export interface GaussianInstance {
  idx: number;
  x: number;
  y: number;
  rx: number;
  ry: number;
  theta: number;
  r: number;
  g: number;
  b: number;
  opacity: number;
  glow: number;
  halo: number;
  activation: number;
}

/**
 * Bracket a fractional timeline position `t` into the two integer steps that
 * straddle it. `t` is the store clock in [0, L]; steps count is S = L+1, so the
 * valid step range is [0, S-1]. Clamps out-of-range `t`.
 */
export function bracketSteps(
  t: number,
  steps: number,
): { s0: number; s1: number; f: number } {
  const maxStep = Math.max(0, steps - 1);
  const tt = Math.min(Math.max(t, 0), maxStep);
  const s0 = Math.floor(tt);
  const s1 = Math.min(s0 + 1, maxStep);
  return { s0, s1, f: tt - s0 };
}

/** Shortest-path angular interpolation (radians, wraps across ±π). */
export function lerpAngle(a: number, b: number, f: number): number {
  const d = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + d * f;
}

/** Linear interpolation. */
export function lerp(a: number, b: number, f: number): number {
  return a + (b - a) * f;
}

/**
 * Interpolate all 12 channels of one token at fractional time `t`, returning a
 * fully-typed instance. `data` is the flat [S,N,C] C-order fp32 array from
 * PackClient.loadGaussians; `steps`/`tokens` are S/N.
 */
export function interpInstance(
  data: Float32Array,
  steps: number,
  tokens: number,
  token: number,
  t: number,
  channels = CHANNEL_COUNT,
): GaussianInstance {
  const { s0, s1, f } = bracketSteps(t, steps);
  const base0 = (s0 * tokens + token) * channels;
  const base1 = (s1 * tokens + token) * channels;
  const at = (base: number, c: number) => data[base + c];
  const li = (c: number) => lerp(at(base0, c), at(base1, c), f);
  return {
    idx: token,
    x: li(CH.x),
    y: li(CH.y),
    rx: li(CH.rx),
    ry: li(CH.ry),
    theta: lerpAngle(at(base0, CH.theta), at(base1, CH.theta), f),
    r: li(CH.r),
    g: li(CH.g),
    b: li(CH.b),
    opacity: li(CH.opacity),
    glow: li(CH.glow),
    halo: li(CH.halo),
    activation: li(CH.activation),
  };
}

/** Interpolate every token at time `t`. */
export function interpAll(
  data: Float32Array,
  steps: number,
  tokens: number,
  t: number,
  channels = CHANNEL_COUNT,
): GaussianInstance[] {
  const out: GaussianInstance[] = new Array(tokens);
  for (let n = 0; n < tokens; n++) {
    out[n] = interpInstance(data, steps, tokens, n, t, channels);
  }
  return out;
}

/**
 * Mahalanobis radius (in σ units) of point (px,py) inside token `g`'s ellipse.
 * Rotating the delta by -theta into the ellipse's principal frame then scaling
 * by (1/rx, 1/ry). A value ≤ 1 means "inside one σ".
 */
export function mahalanobis(px: number, py: number, g: GaussianInstance): number {
  const dx = px - g.x;
  const dy = py - g.y;
  const c = Math.cos(-g.theta);
  const s = Math.sin(-g.theta);
  const u = (dx * c - dy * s) / Math.max(g.rx, 1e-6);
  const v = (dx * s + dy * c) / Math.max(g.ry, 1e-6);
  return Math.sqrt(u * u + v * v);
}

/**
 * Analytic ellipse hit-test: return the token idx whose Gaussian best contains
 * (px,py) in data space, or -1 if none within `sigma`. CLS (idx 0) is off-grid
 * and excluded. Ties break to the nearest (smallest Mahalanobis radius).
 */
export function hitTest(
  px: number,
  py: number,
  instances: GaussianInstance[],
  sigma = SIGMA_EXTENT,
): number {
  let best = -1;
  let bestR = sigma;
  for (const g of instances) {
    if (g.idx === CLS_INDEX) continue;
    const rad = mahalanobis(px, py, g);
    if (rad < bestR) {
      bestR = rad;
      best = g.idx;
    }
  }
  return best;
}

/**
 * Invert the renderer's square→clip mapping: given (u,v) normalized within the
 * centered square (0..1, v down), recover data-space (x,y) in [0,1] (y down),
 * undoing the FIELD_MARGIN inset. Pure so the hit-test path is fully testable;
 * the DOM rect→(u,v) step lives in the component.
 */
export function squareToWorld(
  u: number,
  v: number,
  margin = FIELD_MARGIN,
): { x: number; y: number } {
  // clip coords (y up), then undo the centered margin scale.
  const cx = (u * 2 - 1) / margin;
  const cy = (1 - v * 2) / margin;
  return { x: (cx + 1) / 2, y: (1 - cy) / 2 };
}
