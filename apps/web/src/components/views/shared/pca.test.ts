import { describe, expect, it } from "vitest";
import { fitPca2, normalizeCoords, project2 } from "./pca";

describe("pca", () => {
  it("recovers the dominant axis of anisotropic data", () => {
    // points along direction (3,1,0,...) in D=8 with small noise on axis 1
    const D = 8, N = 200;
    const rows = new Float32Array(N * D);
    for (let i = 0; i < N; i++) {
      const s = (i / N - 0.5) * 10;
      rows[i * D] = 3 * s;
      rows[i * D + 1] = s + Math.sin(i) * 0.1;
    }
    const basis = fitPca2(rows, N, D);
    // c1 should align with (3,1)/√10
    const ratio = Math.abs(basis.c1[0] / basis.c1[1]);
    expect(ratio).toBeGreaterThan(2.5);
    expect(ratio).toBeLessThan(3.5);
    // projection spreads mostly on component 1
    const p = project2(rows, N, D, basis);
    let vx = 0, vy = 0;
    for (let i = 0; i < N; i++) { vx += p[i * 2] ** 2; vy += p[i * 2 + 1] ** 2; }
    expect(vx).toBeGreaterThan(vy * 10);
  });

  it("is deterministic and normalizeCoords maps into [pad,1-pad]", () => {
    const rows = new Float32Array([0, 0, 1, 2, 3, 1, -1, 4]);
    const a = fitPca2(rows, 4, 2), b = fitPca2(rows, 4, 2);
    expect(Array.from(a.c1)).toEqual(Array.from(b.c1));
    const n = normalizeCoords(project2(rows, 4, 2, a), 0.1);
    for (const v of n) {
      expect(v).toBeGreaterThanOrEqual(0.1 - 1e-6);
      expect(v).toBeLessThanOrEqual(0.9 + 1e-6);
    }
  });
});
