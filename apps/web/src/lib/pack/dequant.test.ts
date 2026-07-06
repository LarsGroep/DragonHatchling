import { describe, expect, it } from "vitest";
import type { QuantInfo } from "@vitreous/schema";
import { dequantizePerRow } from "./dequant";

/**
 * Build a `per_row_uint8` blob for a [nRows, rowLen] float array the same way
 * PackWriter does: uint8 data (row = last axis, scale = row max), then a
 * trailing float32 scales block.
 */
function encodePerRow(rows: number[][]): { blob: ArrayBuffer; quant: QuantInfo; rowLen: number } {
  const nRows = rows.length;
  const rowLen = rows[0].length;
  const data = new Uint8Array(nRows * rowLen);
  const scales = new Float32Array(nRows);
  for (let r = 0; r < nRows; r++) {
    const mx = Math.max(...rows[r]);
    const safe = mx > 0 ? mx : 1;
    scales[r] = mx;
    for (let c = 0; c < rowLen; c++) {
      data[r * rowLen + c] = Math.min(255, Math.max(0, Math.round((rows[r][c] / safe) * 255)));
    }
  }
  const blob = new Uint8Array(data.byteLength + scales.byteLength);
  blob.set(data, 0);
  blob.set(new Uint8Array(scales.buffer), data.byteLength);
  const quant: QuantInfo = {
    scheme: "per_row_uint8",
    row_axis: -1,
    scale_dtype: "float32",
    data_offset: 0,
    data_bytes: data.byteLength,
    scale_offset: data.byteLength,
    scale_count: nRows,
  };
  return { blob: blob.buffer, quant, rowLen };
}

describe("dequantizePerRow", () => {
  it("inverts per-row max quantization within 0.5/255 * scale", () => {
    const rows = [
      [0.0, 0.25, 0.5, 1.0],
      [0.1, 0.9, 0.3, 0.6],
    ];
    const { blob, quant, rowLen } = encodePerRow(rows);
    const out = dequantizePerRow(blob, quant, rowLen);
    expect(out.length).toBe(8);
    for (let r = 0; r < rows.length; r++) {
      const scale = Math.max(...rows[r]);
      for (let c = 0; c < rowLen; c++) {
        expect(out[r * rowLen + c]).toBeCloseTo(rows[r][c], 2);
        // exact quant error bound
        expect(Math.abs(out[r * rowLen + c] - rows[r][c])).toBeLessThanOrEqual((0.5 / 255) * scale + 1e-6);
      }
    }
  });

  it("reproduces the row max exactly (quantizes to 255)", () => {
    const rows = [[0.2, 0.8, 0.4]];
    const { blob, quant, rowLen } = encodePerRow(rows);
    const out = dequantizePerRow(blob, quant, rowLen);
    expect(out[1]).toBeCloseTo(0.8, 6); // max element -> 255/255 * scale
  });

  it("handles an all-zero row (scale 0) as zeros", () => {
    const rows = [[0, 0, 0]];
    const { blob, quant, rowLen } = encodePerRow(rows);
    const out = dequantizePerRow(blob, quant, rowLen);
    expect(Array.from(out)).toEqual([0, 0, 0]);
  });
});
