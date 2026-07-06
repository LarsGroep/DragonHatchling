/**
 * IEEE-754 half-precision (float16) decoding (§5 — tokens.bin and gaussians.bin
 * are raw fp16). The frozen pack format stores these as little-endian uint16.
 * This is the TS counterpart of numpy's `float16` read in
 * `packages/core/.../packs/writer.py` (PackReader.read_array).
 */

/** Decode a single float16 bit pattern (Uint16) to a JS number. */
export function halfToFloat(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  let value: number;
  if (exp === 0) {
    // subnormal (or zero)
    value = frac * Math.pow(2, -24);
  } else if (exp === 0x1f) {
    value = frac === 0 ? Infinity : NaN;
  } else {
    value = (1 + frac / 1024) * Math.pow(2, exp - 15);
  }
  return sign ? -value : value;
}

/**
 * Decode a little-endian float16 buffer to a Float32Array of `count` elements
 * (or the full buffer when `count` is omitted). `byteOffset` selects a slice of
 * the underlying ArrayBuffer.
 */
export function decodeFloat16(
  buffer: ArrayBuffer,
  byteOffset = 0,
  count?: number,
): Float32Array {
  const n = count ?? (buffer.byteLength - byteOffset) >> 1;
  const view = new DataView(buffer, byteOffset, n * 2);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = halfToFloat(view.getUint16(i * 2, true));
  return out;
}
