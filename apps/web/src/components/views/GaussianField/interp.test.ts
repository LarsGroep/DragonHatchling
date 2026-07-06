import { describe, expect, it } from "vitest";
import {
  CH,
  CHANNEL_COUNT,
  FIELD_MARGIN,
  bracketSteps,
  hitTest,
  interpAll,
  interpInstance,
  lerpAngle,
  mahalanobis,
  squareToWorld,
} from "./interp";

/**
 * Build a flat [S,N,C] fp32 field with per-(step,token) channel values from a
 * generator, matching PackClient.loadGaussians' C-order layout.
 */
function makeField(
  steps: number,
  tokens: number,
  fn: (s: number, n: number, c: number) => number,
): Float32Array {
  const C = CHANNEL_COUNT;
  const data = new Float32Array(steps * tokens * C);
  for (let s = 0; s < steps; s++)
    for (let n = 0; n < tokens; n++)
      for (let c = 0; c < C; c++) data[(s * tokens + n) * C + c] = fn(s, n, c);
  return data;
}

describe("bracketSteps", () => {
  it("brackets an interior fractional t", () => {
    expect(bracketSteps(3.25, 13)).toEqual({ s0: 3, s1: 4, f: 0.25 });
  });
  it("returns a zero fraction on an integer t", () => {
    expect(bracketSteps(5, 13)).toEqual({ s0: 5, s1: 6, f: 0 });
  });
  it("clamps at the last step (t == steps-1)", () => {
    expect(bracketSteps(12, 13)).toEqual({ s0: 12, s1: 12, f: 0 });
  });
  it("clamps out-of-range t on both ends", () => {
    expect(bracketSteps(-2, 13)).toEqual({ s0: 0, s1: 1, f: 0 });
    expect(bracketSteps(99, 13)).toEqual({ s0: 12, s1: 12, f: 0 });
  });
});

describe("lerpAngle", () => {
  it("interpolates linearly for small deltas", () => {
    expect(lerpAngle(0, 1, 0.5)).toBeCloseTo(0.5, 6);
  });
  it("takes the shortest path across the ±π seam", () => {
    // from +3.0 to -3.0 the short way is +0.283 (through π), not -6.
    const mid = lerpAngle(3.0, -3.0, 0.5);
    expect(Math.abs(mid)).toBeGreaterThan(3.0); // stepped outward through π
  });
});

describe("interpInstance", () => {
  const steps = 13;
  const tokens = 3;
  // channel c ramps linearly with step: value = s * (c + 1); constant across tokens.
  const data = makeField(steps, tokens, (s, _n, c) => s * (c + 1));

  it("LERPs every linear channel at a fractional t", () => {
    const g = interpInstance(data, steps, tokens, 1, 4.5);
    // x (c=0) = 4.5 * 1 = 4.5 ; rx (c=2) = 4.5 * 3 = 13.5 ; opacity (c=8) = 4.5*9
    expect(g.x).toBeCloseTo(4.5, 5);
    expect(g.rx).toBeCloseTo(13.5, 5);
    expect(g.opacity).toBeCloseTo(4.5 * (CH.opacity + 1), 5);
  });

  it("returns exact step values on integer t", () => {
    const g = interpInstance(data, steps, tokens, 2, 6);
    expect(g.y).toBeCloseTo(6 * (CH.y + 1), 5);
    expect(g.activation).toBeCloseTo(6 * (CH.activation + 1), 5);
  });

  it("interpolates theta angularly (shortest path)", () => {
    // theta channel: step0 = +3.0, step1 = -3.0, others irrelevant.
    const d = makeField(2, 1, (s, _n, c) => (c === CH.theta ? (s === 0 ? 3.0 : -3.0) : 0));
    const g = interpInstance(d, 2, 1, 0, 0.5);
    expect(g.theta).toBeCloseTo(lerpAngle(3.0, -3.0, 0.5), 6);
  });

  it("interpAll covers every token", () => {
    const all = interpAll(data, steps, tokens, 2.0);
    expect(all).toHaveLength(tokens);
    expect(all.map((g) => g.idx)).toEqual([0, 1, 2]);
  });
});

describe("mahalanobis + hitTest", () => {
  const g = {
    idx: 5,
    x: 0.5,
    y: 0.5,
    rx: 0.05,
    ry: 0.05,
    theta: 0,
    r: 0,
    g: 0,
    b: 0,
    opacity: 1,
    glow: 0,
    halo: 0,
    activation: 1,
  };

  it("is 0 at the center and 1 at one sigma", () => {
    expect(mahalanobis(0.5, 0.5, g)).toBeCloseTo(0, 6);
    expect(mahalanobis(0.55, 0.5, g)).toBeCloseTo(1, 6); // dx = rx
  });

  it("respects anisotropy under rotation", () => {
    const rot = { ...g, rx: 0.1, ry: 0.02, theta: Math.PI / 2 };
    // theta=90° swaps axes: moving in +y now scales by rx (0.1), not ry.
    expect(mahalanobis(0.5, 0.6, rot)).toBeCloseTo(0.1 / 0.1, 6); // dy=0.1 / rx=0.1 = 1
  });

  it("hitTest returns the containing token and -1 when outside", () => {
    const inst = [
      { ...g, idx: 0, x: 0.0, y: 0.0 }, // CLS — excluded even if hit
      g,
    ];
    expect(hitTest(0.5, 0.5, inst)).toBe(5);
    expect(hitTest(0.9, 0.9, inst)).toBe(-1); // far from token 5, CLS excluded
  });

  it("hitTest excludes CLS (idx 0) on the image plane", () => {
    const cls = { ...g, idx: 0, x: 0.5, y: 0.5 };
    expect(hitTest(0.5, 0.5, [cls])).toBe(-1);
  });

  it("hitTest picks the nearest of overlapping gaussians", () => {
    const a = { ...g, idx: 1, x: 0.5, y: 0.5 };
    const b = { ...g, idx: 2, x: 0.52, y: 0.5 };
    // pointer nearer to a's center
    expect(hitTest(0.505, 0.5, [a, b])).toBe(1);
  });
});

describe("squareToWorld", () => {
  it("maps the square center to the field center", () => {
    expect(squareToWorld(0.5, 0.5)).toEqual({ x: 0.5, y: 0.5 });
  });

  it("is the inverse of the renderer margin mapping at the field edges", () => {
    // world.x = 0 sits at clip -margin => u = (1-margin)/2 within the square.
    const uLeft = (1 - FIELD_MARGIN) / 2;
    const w = squareToWorld(uLeft, 0.5);
    expect(w.x).toBeCloseTo(0, 6);
  });

  it("puts v=top at world y=0 side, v=bottom at world y=1 side", () => {
    expect(squareToWorld(0.5, (1 - FIELD_MARGIN) / 2).y).toBeCloseTo(0, 6);
    expect(squareToWorld(0.5, 1 - (1 - FIELD_MARGIN) / 2).y).toBeCloseTo(1, 6);
  });
});
