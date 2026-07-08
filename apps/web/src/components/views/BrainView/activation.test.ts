import { describe, expect, it } from "vitest";
import {
  communityLabel,
  controlPoint,
  ema,
  emaInto,
  hotThreshold,
  meanOver,
  normalizeUnit,
  quadPoint,
  regionDescriptor,
  shouldShowLabel,
  tokenNorms,
} from "./activation";

describe("tokenNorms", () => {
  it("computes the per-token L2 norm at a step", () => {
    // 1 step, 2 tokens, dim 2: token0=[3,4]→5, token1=[0,0]→0
    const data = new Float32Array([3, 4, 0, 0]);
    const norms = tokenNorms(data, 0, 2, 2);
    expect(norms[0]).toBeCloseTo(5);
    expect(norms[1]).toBeCloseTo(0);
  });

  it("indexes the requested step", () => {
    // 2 steps, 1 token, dim 2. step1 token=[6,8]→10
    const data = new Float32Array([1, 0, 6, 8]);
    expect(tokenNorms(data, 1, 1, 2)[0]).toBeCloseTo(10);
  });
});

describe("normalizeUnit", () => {
  it("maps min→0 and max→1", () => {
    const out = normalizeUnit(new Float32Array([2, 4, 6]));
    expect(out[0]).toBeCloseTo(0);
    expect(out[2]).toBeCloseTo(1);
    expect(out[1]).toBeCloseTo(0.5);
  });

  it("excludes an index from the range but still clamps it", () => {
    // idx0 is a huge outlier excluded from range; range over [1,3]=[10,30].
    const out = normalizeUnit(new Float32Array([1000, 10, 30]), 0);
    expect(out[1]).toBeCloseTo(0);
    expect(out[2]).toBeCloseTo(1);
    expect(out[0]).toBe(1); // clamped
  });

  it("returns zeros when the range is degenerate", () => {
    expect(Array.from(normalizeUnit(new Float32Array([5, 5, 5])))).toEqual([0, 0, 0]);
  });
});

describe("ema", () => {
  it("moves a fraction toward the target and converges", () => {
    expect(ema(0, 1, 0.5)).toBeCloseTo(0.5);
    let v = 0;
    for (let i = 0; i < 50; i++) v = ema(v, 1, 0.2);
    expect(v).toBeGreaterThan(0.99);
  });

  it("emaInto blends element-wise in place", () => {
    const prev = new Float32Array([0, 10]);
    emaInto(prev, new Float32Array([1, 0]), 0.5);
    expect(prev[0]).toBeCloseTo(0.5);
    expect(prev[1]).toBeCloseTo(5);
  });
});

describe("hotThreshold", () => {
  it("caps the hot set to roughly the top fraction", () => {
    const vals = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const thr = hotThreshold(vals, 0.2); // top 20% → 2 of 10
    const hot = Array.from(vals).filter((v) => v >= thr).length;
    expect(hot).toBeLessThanOrEqual(3);
    expect(hot).toBeGreaterThanOrEqual(2);
  });
});

describe("curve math", () => {
  it("control point is offset perpendicular to the segment by curvature×length", () => {
    // horizontal segment length 2 → perpendicular offset 0.5*2 = 1 in ±y.
    const [cx, cy] = controlPoint(-1, 0, 1, 0, 0.5);
    expect(cx).toBeCloseTo(0);
    expect(Math.abs(cy)).toBeCloseTo(1);
  });

  it("quadPoint hits endpoints at u=0/1 and the control-biased midpoint at u=0.5", () => {
    expect(quadPoint(0, 0, 1, 2, 2, 0, 0)).toEqual([0, 0]);
    expect(quadPoint(0, 0, 1, 2, 2, 0, 1)).toEqual([2, 0]);
    const mid = quadPoint(0, 0, 1, 2, 2, 0, 0.5);
    expect(mid[0]).toBeCloseTo(1);
    expect(mid[1]).toBeCloseTo(1); // 0.25*0 + 0.5*2 + 0.25*0
  });
});

describe("meanOver", () => {
  it("averages selected members", () => {
    expect(meanOver(new Float32Array([1, 2, 3, 4]), [0, 3])).toBeCloseTo(2.5);
    expect(meanOver(new Float32Array([1]), [])).toBe(0);
  });
});

describe("shouldShowLabel (threshold + hysteresis)", () => {
  it("needs to exceed threshold+hyst to appear, and drops below threshold-hyst", () => {
    expect(shouldShowLabel(0.5, false, 0.5, 0.06)).toBe(false); // not enough to appear
    expect(shouldShowLabel(0.57, false, 0.5, 0.06)).toBe(true); // appears
    expect(shouldShowLabel(0.46, true, 0.5, 0.06)).toBe(true); // stays (hysteresis)
    expect(shouldShowLabel(0.43, true, 0.5, 0.06)).toBe(false); // drops
  });
});

describe("regionDescriptor (honest, geometry-only)", () => {
  it("names the centroid region", () => {
    // grid 14: idx 1 = row0,col0 → upper-left; idx 196 = row13,col13 → lower-right
    expect(regionDescriptor([1], 14)).toBe("upper-left patches");
    expect(regionDescriptor([196], 14)).toBe("lower-right patches");
  });

  it("central patches when the centroid sits in the middle band", () => {
    // idx for row6,col6 = 6*14+6+1 = 91
    expect(regionDescriptor([91], 14)).toBe("central patches");
  });
});

describe("communityLabel (honesty rule)", () => {
  it("uses the geometric region when no concept dictionary is present", () => {
    expect(communityLabel([1], 14, null)).toBe("upper-left patches");
  });

  it("names the dominant firing feature id when a plurality fires it", () => {
    const top1 = { 1: 7, 2: 7, 3: 9 };
    // tokens 1,2 fire feature 7 (plurality) → feature #7 prefix
    expect(communityLabel([1, 2, 3], 14, top1)).toMatch(/^feature #7 · /);
  });

  it("falls back to region when no feature has a plurality", () => {
    const top1 = { 1: 7 };
    expect(communityLabel([1], 14, top1)).toBe("upper-left patches");
  });
});
