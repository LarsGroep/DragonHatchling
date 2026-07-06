/**
 * Attention dequantization (§5). attention.bin is per-row max-quantized to
 * uint8: a uint8 data block (C-order) followed by a trailing float32 per-row
 * scales block. Row r (the last, key axis) dequantizes as
 * `data[r] / 255 * scale[r]` — error <= 0.5/255 per element. This is the exact
 * inverse of PackWriter._quantize_per_row in packages/core.
 */
import type { QuantInfo } from "@vitreous/schema";

/**
 * Dequantize a `per_row_uint8` blob to Float32Array using the manifest quant
 * offsets. `rowLen` is the length of the quantized last axis (== shape[-1]).
 * `blob` is the whole asset (data block + scales block).
 */
export function dequantizePerRow(
  blob: ArrayBuffer,
  quant: QuantInfo,
  rowLen: number,
): Float32Array {
  const data = new Uint8Array(blob, quant.data_offset, quant.data_bytes);
  const scales = new Float32Array(
    blob.slice(quant.scale_offset, quant.scale_offset + quant.scale_count * 4),
  );
  const out = new Float32Array(data.length);
  for (let r = 0; r < quant.scale_count; r++) {
    const scale = scales[r];
    const base = r * rowLen;
    for (let c = 0; c < rowLen; c++) {
      out[base + c] = (data[base + c] / 255) * scale;
    }
  }
  return out;
}
