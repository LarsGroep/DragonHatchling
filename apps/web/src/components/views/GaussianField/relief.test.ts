import { describe, expect, it } from "vitest";
import { RELIEF_Z_GAIN, nearestScreenIndex, orbitEye, zForGlow } from "./relief";

describe("zForGlow", () => {
  it("is zero at the ground and scales linearly with glow", () => {
    expect(zForGlow(0)).toBe(0);
    expect(zForGlow(1)).toBeCloseTo(RELIEF_Z_GAIN, 6);
    expect(zForGlow(0.5)).toBeCloseTo(RELIEF_Z_GAIN * 0.5, 6);
  });
  it("clamps negative glow to the ground plane", () => {
    expect(zForGlow(-0.3)).toBe(0);
  });
  it("is a pure, deterministic function (honesty rule)", () => {
    expect(zForGlow(0.77)).toBe(zForGlow(0.77));
  });
  it("keeps max glow within ~0.35 of the field width (1.8)", () => {
    expect(zForGlow(1) / 1.8).toBeLessThanOrEqual(0.35);
  });
});

describe("orbitEye", () => {
  it("places the eye on +Z looking down the axis at azimuth 0, polar 90°", () => {
    const [x, y, z] = orbitEye(2, 0, Math.PI / 2);
    expect(x).toBeCloseTo(0, 6);
    expect(y).toBeCloseTo(0, 6);
    expect(z).toBeCloseTo(2, 6);
  });
  it("puts the eye straight overhead at polar 0", () => {
    const [x, y, z] = orbitEye(3, 1.2, 0);
    expect(x).toBeCloseTo(0, 6);
    expect(y).toBeCloseTo(3, 6);
    expect(z).toBeCloseTo(0, 6);
  });
  it("offsets by the orbit center", () => {
    const [x, y, z] = orbitEye(1, 0, Math.PI / 2, [5, 6, 7]);
    expect(x).toBeCloseTo(5, 6);
    expect(y).toBeCloseTo(6, 6);
    expect(z).toBeCloseTo(8, 6);
  });
});

describe("nearestScreenIndex", () => {
  const pts = [
    { idx: 3, x: 10, y: 10 },
    { idx: 7, x: 100, y: 100 },
    { idx: 9, x: 12, y: 11 },
  ];
  it("returns the nearest point within the radius", () => {
    expect(nearestScreenIndex(pts, 11, 10, 20)).toBe(3);
    expect(nearestScreenIndex(pts, 101, 99, 20)).toBe(7);
  });
  it("returns -1 when nothing is within maxDist", () => {
    expect(nearestScreenIndex(pts, 500, 500, 20)).toBe(-1);
  });
  it("returns -1 for an empty set", () => {
    expect(nearestScreenIndex([], 0, 0, 20)).toBe(-1);
  });
});
