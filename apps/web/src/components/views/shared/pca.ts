/**
 * Tiny PCA for the Latent Embedding view (§10). A fixed 2D basis is fit ONCE
 * per pack on the pooled token embeddings of every timeline step, then every
 * step is projected into that shared basis — so the cloud animates coherently
 * over t and the CLS trajectory lives in the same space. Power iteration on
 * the covariance (D=384) — no deps, fast enough for 13×197 rows.
 */

export interface PcaBasis {
  mean: Float64Array; // [D]
  c1: Float64Array; // [D]
  c2: Float64Array; // [D]
}

function matVec(rows: Float32Array, n: number, d: number, mean: Float64Array, v: Float64Array): Float64Array {
  // (X-mean)^T (X-mean) v  computed as sum over rows without materializing cov
  const out = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    let dot = 0;
    const off = i * d;
    for (let j = 0; j < d; j++) dot += (rows[off + j] - mean[j]) * v[j];
    for (let j = 0; j < d; j++) out[j] += (rows[off + j] - mean[j]) * dot;
  }
  return out;
}

function normalize(v: Float64Array): void {
  let s = 0;
  for (let j = 0; j < v.length; j++) s += v[j] * v[j];
  s = Math.sqrt(s) || 1;
  for (let j = 0; j < v.length; j++) v[j] /= s;
}

function orthogonalize(v: Float64Array, against: Float64Array): void {
  let dot = 0;
  for (let j = 0; j < v.length; j++) dot += v[j] * against[j];
  for (let j = 0; j < v.length; j++) v[j] -= dot * against[j];
}

/** Fit a 2-component PCA basis on `rows` ([n][d] flat, C-order). Deterministic. */
export function fitPca2(rows: Float32Array, n: number, d: number, iters = 12): PcaBasis {
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += rows[i * d + j];
  for (let j = 0; j < d; j++) mean[j] /= n || 1;

  // deterministic seeded start vectors
  const c1 = new Float64Array(d);
  const c2 = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    c1[j] = Math.sin(j * 12.9898) * 0.5 + 0.5;
    c2[j] = Math.cos(j * 78.233) * 0.5 + 0.5;
  }
  normalize(c1);
  for (let k = 0; k < iters; k++) {
    const nv = matVec(rows, n, d, mean, c1);
    normalize(nv);
    c1.set(nv);
  }
  orthogonalize(c2, c1);
  normalize(c2);
  for (let k = 0; k < iters; k++) {
    const nv = matVec(rows, n, d, mean, c2);
    orthogonalize(nv, c1);
    normalize(nv);
    c2.set(nv);
  }
  return { mean, c1, c2 };
}

/** Project `rows` ([n][d]) into the basis -> [n][2] flat. */
export function project2(rows: Float32Array, n: number, d: number, basis: PcaBasis): Float32Array {
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    let x = 0;
    let y = 0;
    const off = i * d;
    for (let j = 0; j < d; j++) {
      const c = rows[off + j] - basis.mean[j];
      x += c * basis.c1[j];
      y += c * basis.c2[j];
    }
    out[i * 2] = x;
    out[i * 2 + 1] = y;
  }
  return out;
}

/** Min-max normalize [n][2] coords into [pad, 1-pad]² (in place-safe copy). */
export function normalizeCoords(coords: Float32Array, pad = 0.08): Float32Array {
  let xlo = Infinity, xhi = -Infinity, ylo = Infinity, yhi = -Infinity;
  for (let i = 0; i < coords.length; i += 2) {
    xlo = Math.min(xlo, coords[i]); xhi = Math.max(xhi, coords[i]);
    ylo = Math.min(ylo, coords[i + 1]); yhi = Math.max(yhi, coords[i + 1]);
  }
  const sx = xhi - xlo || 1, sy = yhi - ylo || 1;
  const out = new Float32Array(coords.length);
  const span = 1 - 2 * pad;
  for (let i = 0; i < coords.length; i += 2) {
    out[i] = pad + ((coords[i] - xlo) / sx) * span;
    out[i + 1] = pad + ((coords[i + 1] - ylo) / sy) * span;
  }
  return out;
}
