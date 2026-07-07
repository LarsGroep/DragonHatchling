/**
 * Pure math for the 3D Gaussian relief mode (S2) — no three.js / WebGL import so
 * vitest exercises it headless. The renderer owns the matrices; this module owns
 * the honest z-mapping, the orbit camera placement, and the screen-space nearest
 * pick used for hover/selection in 3D.
 *
 * HONESTY RULE (§7): a splat's height `z` is a PURE function of its measured
 * `glow` channel (Chefer attribution) and nothing else — no fitted terrain, no
 * smoothing. `zForGlow` is that function; the renderer labels the axis.
 */

/**
 * World-units of height per unit glow. Glow ships normalized to [0, 1]; the
 * field plane spans [-FIELD_MARGIN, +FIELD_MARGIN] (width 2·0.9 = 1.8), so a
 * max-glow splat rises ~0.33 of the field width — the "~0..0.35" target.
 */
export const RELIEF_Z_GAIN = 0.6;

/**
 * Height of a splat from its measured attribution (glow). Pure and monotonic;
 * negatives (should not occur) clamp to the ground plane.
 */
export function zForGlow(glow: number, gain: number = RELIEF_Z_GAIN): number {
  return (glow > 0 ? glow : 0) * gain;
}

/** Eye position for a spherical orbit about `center`. Pure — unit-tested. */
export function orbitEye(
  radius: number,
  azimuth: number,
  polar: number,
  center: readonly [number, number, number] = [0, 0, 0],
): [number, number, number] {
  const sp = Math.sin(polar);
  const cp = Math.cos(polar);
  return [
    center[0] + radius * sp * Math.sin(azimuth),
    center[1] + radius * cp,
    center[2] + radius * sp * Math.cos(azimuth),
  ];
}

export interface ScreenPoint {
  idx: number;
  x: number;
  y: number;
}

/**
 * Index of the projected splat center nearest to (px, py) within `maxDist`
 * pixels, or -1. The 3D hover/pick path uses this (documented in index.tsx): the
 * analytic ellipse hit-test is plane-only, so in relief mode we project every
 * splat center with the live view-projection and take the nearest on screen.
 */
export function nearestScreenIndex(
  pts: readonly ScreenPoint[],
  px: number,
  py: number,
  maxDist: number,
): number {
  let best = -1;
  let bestD = maxDist * maxDist;
  for (const p of pts) {
    const dx = p.x - px;
    const dy = p.y - py;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = p.idx;
    }
  }
  return best;
}
