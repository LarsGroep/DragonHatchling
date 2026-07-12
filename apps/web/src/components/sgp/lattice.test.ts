/** Pure lattice math for the SGP renderer — headless (no WebGL). */
import { describe, expect, it } from "vitest";
import {
  LATTICE_EXTENT,
  activationAt,
  boundaryStrength,
  bracketDepth,
  communityRgb,
  emaToward,
  latticeWorld,
  sizeForHits,
} from "./lattice";

describe("latticeWorld", () => {
  const grid: [number, number, number] = [8, 8, 8];

  it("centers the lattice and spans ±EXTENT", () => {
    expect(latticeWorld(0, 0, 0, grid)).toEqual([-LATTICE_EXTENT, -LATTICE_EXTENT, -LATTICE_EXTENT]);
    expect(latticeWorld(7, 7, 7, grid)).toEqual([LATTICE_EXTENT, LATTICE_EXTENT, LATTICE_EXTENT]);
  });

  it("maps the depth axis (z) to world Y (up)", () => {
    const lo = latticeWorld(0, 3, 3, grid);
    const hi = latticeWorld(7, 3, 3, grid);
    expect(hi[1]).toBeGreaterThan(lo[1]); // Y grows with depth
    expect(hi[0]).toBe(lo[0]); // X unchanged
    expect(hi[2]).toBe(lo[2]); // Z unchanged
  });

  it("degenerate single-cell axes sit at the center", () => {
    expect(latticeWorld(0, 0, 0, [1, 1, 1])).toEqual([0, 0, 0]);
  });
});

describe("bracketDepth", () => {
  it("brackets interior fractions", () => {
    expect(bracketDepth(2.25, 8)).toEqual({ lo: 2, hi: 3, f: 0.25 });
  });
  it("clamps below and above", () => {
    expect(bracketDepth(-1, 8)).toEqual({ lo: 0, hi: 1, f: 0 });
    expect(bracketDepth(99, 8)).toEqual({ lo: 7, hi: 7, f: 0 });
  });
});

describe("activationAt", () => {
  const acts = [Float32Array.from([1, 0, 0]), Float32Array.from([0, 1, 0])];

  it("interpolates between bracketing depths and normalizes the peak to 1", () => {
    const mid = activationAt(acts, 0.5);
    expect(mid[0]).toBeCloseTo(1); // 0.5/0.5 normalized
    expect(mid[1]).toBeCloseTo(1);
    expect(mid[2]).toBe(0);
    const at0 = activationAt(acts, 0);
    expect(Array.from(at0)).toEqual([1, 0, 0]);
  });

  it("reuses a provided output buffer", () => {
    const buf = new Float32Array(3);
    const out = activationAt(acts, 1, buf);
    expect(out).toBe(buf);
    expect(Array.from(out)).toEqual([0, 1, 0]);
  });
});

describe("emaToward", () => {
  it("moves toward the target and converges", () => {
    const cur = Float32Array.from([0]);
    emaToward(cur, Float32Array.from([1]), 0.016);
    expect(cur[0]).toBeGreaterThan(0);
    expect(cur[0]).toBeLessThan(1);
    for (let i = 0; i < 200; i++) emaToward(cur, Float32Array.from([1]), 0.016);
    expect(cur[0]).toBeCloseTo(1, 3);
  });
});

describe("sizeForHits / boundaryStrength / communityRgb", () => {
  it("sqrt-sizes hits and pins dead to the minimum", () => {
    expect(sizeForHits(0, 100)).toBeCloseTo(0.16);
    expect(sizeForHits(100, 100)).toBeCloseTo(1.0);
    expect(sizeForHits(25, 100)).toBeCloseTo(0.16 + 0.84 * 0.5);
  });

  it("normalizes U-matrix to [0,1] and handles constant fields", () => {
    const b = boundaryStrength([1, 2, 3]);
    expect(Array.from(b)).toEqual([0, 0.5, 1]);
    expect(Array.from(boundaryStrength([5, 5, 5]))).toEqual([0, 0, 0]);
  });

  it("community colors are valid RGB and wrap", () => {
    const a = communityRgb(0);
    const b = communityRgb(12); // wraps to hue[0]
    expect(a).toEqual(b);
    for (const c of a) {
      expect(c).toBeGreaterThanOrEqual(0);
      expect(c).toBeLessThanOrEqual(1);
    }
  });
});
