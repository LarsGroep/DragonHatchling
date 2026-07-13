/**
 * Malignancy lens — web mirror of `vitreous.malignancy` (docs/MALIGNANCY-LENS.md).
 *
 * Three honest readings of a HAM10000-style pack, all derived from data the pack
 * already ships — no new per-image asset:
 *   1. malignant probability = Σ P(class) over the malignant group;
 *   2. category coordinate = softmax-weighted position on the ordinal
 *      benign → in-situ → invasive axis (a CATEGORY reading, never clinical stage);
 *   3. manifold position + OOD = projection of the final-step CLS token onto the
 *      dataset's learned benign↔malignant axis (`malignancy_axis.json`), with an
 *      off-axis residual that flags out-of-distribution inputs.
 *
 * Pure + headless-tested (mirrors lib/sgp.ts). The decode math is the exact
 * inverse of the Python core so a pack reads identically on both sides.
 */
import type { LoadedTokens } from "./pack/types";

/** Class groupings delivered with the dataset (from DatasetSpec.taxonomy). */
export interface Taxonomy {
  /** class name -> malignant flag. */
  malignant: Record<string, boolean>;
  /** class name -> ordinal category level (0 = least advanced). */
  category_level: Record<string, number>;
  /** human labels for the levels, e.g. ["benign","in-situ","invasive"]. */
  category_labels: string[];
  axis_pair?: [string, string];
}

/** Dataset-level learned axis (`malignancy_axis.json`). */
export interface MalignancyAxis {
  provider: "malignancy_axis";
  space: string;
  dim: number;
  u: number[];
  centroid_benign: number[];
  anchor_lo: number;
  anchor_hi: number;
  residual_threshold: number;
  provenance?: Record<string, unknown>;
}

// ── derived readouts (axes 1 & 2) ──────────────────────────────────────────

/** Indices of the classes flagged malignant by the taxonomy. */
export function malignantIndices(classNames: string[], tax: Taxonomy): number[] {
  const out: number[] = [];
  classNames.forEach((c, i) => {
    if (tax.malignant[c]) out.push(i);
  });
  return out;
}

/** Σ P(class) over the malignant classes → [0,1]. */
export function malignantProbability(probabilities: number[], malignantIdx: number[]): number {
  let s = 0;
  for (const i of malignantIdx) s += probabilities[i] ?? 0;
  return Math.max(0, Math.min(1, s));
}

/** Per-class ordinal level array aligned to classNames. */
export function categoryLevels(classNames: string[], tax: Taxonomy): number[] {
  return classNames.map((c) => tax.category_level[c] ?? 0);
}

/** Softmax-weighted category coordinate Σ P(c)·level(c) → [0, K-1]. */
export function expectedCategory(probabilities: number[], levels: number[]): number {
  let mass = 0;
  let acc = 0;
  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i] ?? 0;
    mass += p;
    acc += p * (levels[i] ?? 0);
  }
  return mass > 0 ? acc / mass : 0;
}

/** Ordinal level of the argmax class. */
export function hardCategory(probabilities: number[], levels: number[]): number {
  let best = 0;
  let bestP = -Infinity;
  for (let i = 0; i < probabilities.length; i++) {
    if ((probabilities[i] ?? 0) > bestP) {
      bestP = probabilities[i] ?? 0;
      best = i;
    }
  }
  return levels[best] ?? 0;
}

// ── manifold projection (axis 3) ───────────────────────────────────────────

/** The final-step CLS token embedding [D] from decoded tokens.bin ([S,T,D]). */
export function clsFinalFeature(tokens: LoadedTokens, clsIndex = 0): Float32Array {
  const { data, steps, tokens: T, dim } = tokens;
  const step = steps - 1; // final norm output = last timeline step
  const base = (step * T + clsIndex) * dim;
  return data.subarray(base, base + dim);
}

export interface ManifoldReading {
  /** clamped [0,1] position along benign→malignant. */
  position: number;
  /** off-axis residual distance. */
  residual: number;
  /** residual exceeds the axis threshold → refuse rather than assert. */
  ood: boolean;
}

/** Project one feature onto a built axis → {position, residual, ood}. */
export function projectFeature(feature: ArrayLike<number>, axis: MalignancyAxis): ManifoldReading {
  const { u, centroid_benign: cb, anchor_lo: lo, anchor_hi: hi } = axis;
  let dotFU = 0; // f·u
  let along = 0; // (f-cb)·u
  for (let i = 0; i < u.length; i++) {
    const fi = feature[i] ?? 0;
    dotFU += fi * u[i];
    along += (fi - cb[i]) * u[i];
  }
  let resid2 = 0;
  for (let i = 0; i < u.length; i++) {
    const perp = (feature[i] ?? 0) - cb[i] - along * u[i];
    resid2 += perp * perp;
  }
  const residual = Math.sqrt(resid2);
  const span = hi > lo ? hi - lo : 1;
  const position = Math.max(0, Math.min(1, (dotFU - lo) / span));
  return { position, residual, ood: residual > axis.residual_threshold };
}
