import { describe, expect, it } from "vitest";
import { decodeFloat16, halfToFloat } from "./fp16";

describe("halfToFloat", () => {
  it("decodes canonical bit patterns", () => {
    expect(halfToFloat(0x0000)).toBe(0);
    expect(halfToFloat(0x3c00)).toBe(1); // 1.0
    expect(halfToFloat(0xc000)).toBe(-2); // -2.0
    expect(halfToFloat(0x4000)).toBe(2); // 2.0
    expect(halfToFloat(0x3800)).toBe(0.5); // 0.5
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    expect(halfToFloat(0xfc00)).toBe(-Infinity);
    expect(Number.isNaN(halfToFloat(0x7e00))).toBe(true);
  });

  it("decodes the smallest positive subnormal", () => {
    // 2^-24
    expect(halfToFloat(0x0001)).toBeCloseTo(Math.pow(2, -24), 12);
  });

  it("decodes a representative fraction (0.333...)", () => {
    // 0x3555 ≈ 0.333251953125
    expect(halfToFloat(0x3555)).toBeCloseTo(0.33325, 5);
  });
});

describe("decodeFloat16", () => {
  it("decodes a little-endian fp16 buffer to Float32Array", () => {
    // [1.0, -2.0, 0.5] as LE uint16: 0x3c00, 0xc000, 0x3800
    const buf = new Uint8Array([0x00, 0x3c, 0x00, 0xc0, 0x00, 0x38]).buffer;
    const out = decodeFloat16(buf, 0, 3);
    expect(Array.from(out)).toEqual([1, -2, 0.5]);
  });

  it("respects byteOffset and count", () => {
    const buf = new Uint8Array([0xff, 0xff, 0x00, 0x3c, 0x00, 0x40]).buffer;
    const out = decodeFloat16(buf, 2, 2); // skip first uint16
    expect(Array.from(out)).toEqual([1, 2]);
  });
});
