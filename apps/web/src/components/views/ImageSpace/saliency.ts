/**
 * Saliency helpers for Image Space (§13): reduce a loaded attribution to a
 * 14×14 grid at the current timeline layer, and a luminous colormap for the
 * overlay (magma-like ramp; alpha rises with intensity so the image shows
 * through cold regions).
 */
import type { LoadedAttribution } from "@/src/lib/pack/types";

/**
 * Reduce an attribution to a normalized [0,1] 14×14 grid (row-major) at the
 * given attention layer. Handles the three attribution kinds.
 */
export function attributionGrid(
  attr: LoadedAttribution,
  grid: number,
  layer: number,
): Float32Array {
  const nPatch = grid * grid;
  const out = new Float32Array(nPatch);

  if (attr.kind === "token_grid") {
    // [14,14] directly.
    for (let i = 0; i < nPatch && i < attr.data.length; i++) out[i] = attr.data[i];
  } else if (attr.kind === "per_layer_tokens") {
    // [L, T] — pick the layer row, drop CLS (token 0).
    const T = attr.shape[1];
    const L = attr.shape[0];
    const l = Math.max(0, Math.min(L - 1, layer));
    for (let p = 0; p < nPatch; p++) out[p] = attr.data[l * T + (p + 1)];
  } else {
    // "tokens" [T] — drop CLS.
    for (let p = 0; p < nPatch; p++) out[p] = attr.data[p + 1];
  }

  // min-max normalize
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < nPatch; i++) {
    lo = Math.min(lo, out[i]);
    hi = Math.max(hi, out[i]);
  }
  const span = hi - lo;
  if (span > 0) for (let i = 0; i < nPatch; i++) out[i] = (out[i] - lo) / span;
  else out.fill(0);
  return out;
}

// magma-ish control points (r,g,b) at t = 0, .25, .5, .75, 1
const RAMP: Array<[number, number, number]> = [
  [8, 6, 30],
  [80, 18, 110],
  [186, 54, 108],
  [244, 132, 68],
  [252, 236, 158],
];

/** Map v∈[0,1] to an rgba tuple (0..255). Alpha eases in with intensity. */
export function magma(v: number): [number, number, number, number] {
  const t = Math.max(0, Math.min(1, v));
  const seg = t * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(seg));
  const f = seg - i;
  const a = RAMP[i];
  const b = RAMP[i + 1];
  const r = a[0] + (b[0] - a[0]) * f;
  const g = a[1] + (b[1] - a[1]) * f;
  const bl = a[2] + (b[2] - a[2]) * f;
  // alpha: low for cold, opaque for hot
  const alpha = Math.round(255 * Math.pow(t, 0.7) * 0.92);
  return [Math.round(r), Math.round(g), Math.round(bl), alpha];
}
