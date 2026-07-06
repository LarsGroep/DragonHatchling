/**
 * gen-mock-pack.mjs — mint the committed MOCK Explanation Pack fixture.
 *
 * M5 needs the web app (and CI, and the M6/M7 renderers) to run with ZERO
 * backend. The Python `PackWriter` (packages/core) owns the frozen v1 binary
 * layouts, but it needs numpy/torch/PIL which are not part of the web toolchain.
 * This script reproduces the SAME frozen layouts (§5 ARCHITECTURE.md, and the
 * exact rules in packages/core/.../packs/writer.py) in pure Node so the fixture
 * can be regenerated from the frontend workspace alone:
 *
 *   attention.bin   uint8 per-row-max-quantized  [L,H,T,T] + trailing f32 scales
 *   tokens.bin      float16 raw                  [L+1,T,D]
 *   gaussians.bin   float16 raw                  [L+1,T,12]  (+ meta.channels)
 *   attr_*.bin      float32 raw
 *   *.json          manifest / attributions / faithfulness / graph / concepts
 *   image.png       display image (PNG; the format is recorded in the manifest)
 *
 * Output: apps/web/public/mock/{datasets.json, packs/<dataset>/<image>/...}
 * These artifacts are COMMITTED — this script only needs to run when the
 * fixture must change. `node scripts/gen-mock-pack.mjs`.
 */
import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const MOCK = join(HERE, "..", "public", "mock");

// ViT-S/16 dimensions (§ model = deit_small_patch16_224).
const L = 12; // attention layers
const H = 6; // heads
const T = 197; // tokens (CLS + 14*14)
const D = 384; // embed dim
const GRID = 14;
const S = L + 1; // timeline steps 0..12
const GAUSS_CHANNELS = [
  "x", "y", "rx", "ry", "theta", "r", "g", "b",
  "opacity", "glow", "halo", "activation_raw",
];

// --------------------------------------------------------------------------- //
// numeric encoders
// --------------------------------------------------------------------------- //

/** IEEE-754 float32 -> float16 bit pattern (round-to-nearest-even). */
const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);
function floatToHalf(val) {
  _f32[0] = val;
  const x = _i32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x007fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  exp = exp - 127 + 15;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let h = mant >> shift;
    if ((mant >> (shift - 1)) & 1) h += 1; // round
    return sign | h;
  }
  if (mant & 0x1000) {
    mant += 0x2000;
    if (mant & 0x800000) {
      mant = 0;
      exp += 1;
      if (exp >= 31) return sign | 0x7c00;
    }
  }
  return sign | (exp << 10) | (mant >> 13);
}

function float16Buffer(values) {
  const buf = Buffer.allocUnsafe(values.length * 2);
  for (let i = 0; i < values.length; i++) buf.writeUInt16LE(floatToHalf(values[i]), i * 2);
  return buf;
}

function float32Buffer(values) {
  const buf = Buffer.allocUnsafe(values.length * 4);
  for (let i = 0; i < values.length; i++) buf.writeFloatLE(values[i], i * 4);
  return buf;
}

/**
 * Per-row (last-axis) max-quantize a flat float array whose logical shape has
 * last dimension `rowLen`. Returns { data: uint8 Buffer, scales: f32 Buffer }.
 * Mirrors PackWriter._quantize_per_row exactly.
 */
function quantizePerRow(values, rowLen) {
  const nRows = values.length / rowLen;
  const data = Buffer.allocUnsafe(values.length);
  const scales = Buffer.allocUnsafe(nRows * 4);
  for (let r = 0; r < nRows; r++) {
    let mx = 0;
    for (let c = 0; c < rowLen; c++) mx = Math.max(mx, values[r * rowLen + c]);
    const safe = mx > 0 ? mx : 1.0;
    for (let c = 0; c < rowLen; c++) {
      const q = Math.round((values[r * rowLen + c] / safe) * 255);
      data[r * rowLen + c] = Math.min(255, Math.max(0, q));
    }
    scales.writeFloatLE(mx, r * 4);
  }
  return { data, scales };
}

// --------------------------------------------------------------------------- //
// minimal PNG encoder (pure Node, no deps)
// --------------------------------------------------------------------------- //

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function pngChunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, "ascii");
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}
/** Encode an RGBA Uint8 image [h][w][4] (flat) to a PNG Buffer. */
function encodePng(width, height, rgba) {
  const raw = Buffer.allocUnsafe(height * (1 + width * 4));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + width * 4)] = 0; // filter: none
    rgba.copy(raw, y * (1 + width * 4) + 1, y * width * 4, (y + 1) * width * 4);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type RGBA
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  return Buffer.concat([
    sig,
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

// --------------------------------------------------------------------------- //
// synthetic-but-plausible pack content
// --------------------------------------------------------------------------- //

// Deterministic PRNG so regeneration is stable.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function patchRowCol(i) {
  // token i in 1..196 -> (row, col) via divmod(i-1, 14)
  const j = i - 1;
  return [Math.floor(j / GRID), j % GRID];
}

/** A 14x14 importance field with a soft blob — the "salient" region. */
function saliencyField(rng, cx, cy, sharpness) {
  const f = new Float32Array(GRID * GRID);
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      const dx = (c - cx) / GRID;
      const dy = (r - cy) / GRID;
      const g = Math.exp(-(dx * dx + dy * dy) * sharpness);
      f[r * GRID + c] = g + rng() * 0.06;
    }
  }
  return f;
}

/** Build one head-averaged-ish attention matrix [T,T] per layer, softmax rows. */
function attentionForLayer(rng, layer, salient) {
  // rows = destination (query), cols = key. Each row is a distribution.
  const rows = new Float32Array(T * T);
  const depth = layer / (L - 1); // 0..1
  for (let q = 0; q < T; q++) {
    let sum = 0;
    for (let k = 0; k < T; k++) {
      // self + attend to CLS + attend to salient patches (grows with depth)
      let w = 0.15;
      if (k === q) w += 1.0;
      if (k === 0) w += 0.4 + depth * 0.6; // CLS becomes a hub late
      if (k > 0) w += salient[k - 1] * (0.3 + depth * 1.2);
      w += rng() * 0.05;
      rows[q * T + k] = w;
    }
    // softmax the row
    let mx = 0;
    for (let k = 0; k < T; k++) mx = Math.max(mx, rows[q * T + k]);
    for (let k = 0; k < T; k++) {
      const e = Math.exp(rows[q * T + k] - mx);
      rows[q * T + k] = e;
      sum += e;
    }
    for (let k = 0; k < T; k++) rows[q * T + k] /= sum;
  }
  return rows;
}

function trapzUnit(curve) {
  // integrate over [0,1] with n-1 equal steps
  const n = curve.length;
  if (n < 2) return 0;
  let s = 0;
  for (let i = 0; i < n - 1; i++) s += (curve[i] + curve[i + 1]) / 2;
  return s / (n - 1);
}

function spearman(a, b) {
  const rank = (arr) => {
    const idx = arr.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
    const rk = new Array(arr.length);
    idx.forEach(([, i], r) => (rk[i] = r));
    return rk;
  };
  const ra = rank(a);
  const rb = rank(b);
  const n = a.length;
  let d2 = 0;
  for (let i = 0; i < n; i++) d2 += (ra[i] - rb[i]) ** 2;
  return 1 - (6 * d2) / (n * (n * n - 1));
}

// --------------------------------------------------------------------------- //
// pack builders
// --------------------------------------------------------------------------- //

function writePng(dir, cx, cy, base, seed) {
  const rng = mulberry32(seed);
  const W = 224;
  const H2 = 224;
  const rgba = Buffer.allocUnsafe(W * H2 * 4);
  for (let y = 0; y < H2; y++) {
    for (let x = 0; x < W; x++) {
      const dx = (x / W) * GRID - cx;
      const dy = (y / H2) * GRID - cy;
      const g = Math.exp(-(dx * dx + dy * dy) / 18);
      const n = rng() * 0.12;
      const r = Math.min(255, (base[0] * (0.55 + n) + g * 90) | 0);
      const gg = Math.min(255, (base[1] * (0.6 + n) + g * 70) | 0);
      const b = Math.min(255, (base[2] * (0.5 + n) + g * 40) | 0);
      const o = (y * W + x) * 4;
      rgba[o] = r;
      rgba[o + 1] = gg;
      rgba[o + 2] = b;
      rgba[o + 3] = 255;
    }
  }
  const png = encodePng(W, H2, rgba);
  writeFileSync(join(dir, "image.png"), png);
  return { filename: "image.png", bytes: png.length };
}

function attributionAssets(rng, salient) {
  // per-layer token attributions [L,T] (chefer, rollout), gradcam [14,14], ig [T]
  const perLayer = (growth) => {
    const arr = new Float32Array(L * T);
    for (let l = 0; l < L; l++) {
      const depth = l / (L - 1);
      // CLS score
      arr[l * T + 0] = 0.1 + depth * 0.2;
      for (let i = 1; i < T; i++) {
        const s = salient[i - 1];
        arr[l * T + i] = s * (0.2 + depth * growth) + rng() * 0.03;
      }
      // normalize each layer row to [0,1]
      let mx = 0;
      for (let i = 0; i < T; i++) mx = Math.max(mx, arr[l * T + i]);
      if (mx > 0) for (let i = 0; i < T; i++) arr[l * T + i] /= mx;
    }
    return arr;
  };
  const chefer = perLayer(1.4);
  const rollout = perLayer(1.0);
  const gradcam = new Float32Array(GRID * GRID);
  for (let i = 0; i < GRID * GRID; i++) gradcam[i] = salient[i];
  const ig = new Float32Array(T);
  ig[0] = 0.05;
  for (let i = 1; i < T; i++) ig[i] = salient[i - 1] * (0.8 + rng() * 0.2);
  return { chefer, rollout, gradcam, ig };
}

function faithfulnessJson(attrs, salient) {
  // deletion: start high, drop; insertion: start low, rise. Chefer beats random.
  const rng = mulberry32(7);
  const N = 21;
  const curveFor = (auc, mode) => {
    const c = [];
    for (let i = 0; i < N; i++) {
      const x = i / (N - 1);
      const base = mode === "deletion" ? 0.95 * (1 - x) ** (1 / auc) : 0.05 + 0.9 * x ** auc;
      c.push(Math.max(0, Math.min(1, base + (rng() - 0.5) * 0.02)));
    }
    return c;
  };
  const methods = ["chefer", "rollout", "gradcam", "ig"];
  const del_auc = { chefer: 0.55, rollout: 0.9, gradcam: 0.8, ig: 0.75, random: 1.3 };
  const ins_auc = { chefer: 1.7, rollout: 1.2, gradcam: 1.3, ig: 1.35, random: 0.7 };
  const deletion_curves = {};
  const insertion_curves = {};
  const deletion_auc = {};
  const insertion_auc = {};
  for (const m of [...methods, "random"]) {
    deletion_curves[m] = curveFor(del_auc[m], "deletion");
    insertion_curves[m] = curveFor(ins_auc[m], "insertion");
    deletion_auc[m] = trapzUnit(deletion_curves[m]);
    insertion_auc[m] = trapzUnit(insertion_curves[m]);
  }
  // agreement: Spearman between per-token vectors (use last chefer layer etc.)
  const vecs = {
    chefer: Array.from(attrs.chefer.slice((L - 1) * T + 1, L * T)),
    rollout: Array.from(attrs.rollout.slice((L - 1) * T + 1, L * T)),
    gradcam: Array.from(attrs.gradcam),
    ig: Array.from(attrs.ig.slice(1)),
  };
  const agreement = {};
  for (const a of methods) {
    agreement[a] = {};
    for (const b of methods) agreement[a][b] = a === b ? 1 : round3(spearman(vecs[a], vecs[b]));
  }
  return {
    steps: N - 1,
    deletion_curves,
    insertion_curves,
    deletion_auc,
    insertion_auc,
    agreement,
  };
}

function round3(x) {
  return Math.round(x * 1000) / 1000;
}

function graphJson(attnLayers) {
  const K = 8;
  const layers = [];
  for (let l = 0; l < L; l++) {
    const a = attnLayers[l];
    const nodes = [];
    // trivial 2-community split: CLS-hub vs the rest, for the mock
    for (let i = 0; i < T; i++) {
      nodes.push({ idx: i, kind: i === 0 ? "cls_token" : "patch_token", community: i === 0 ? 0 : 1 });
    }
    const edges = [];
    for (let i = 0; i < T; i++) {
      // top-k keys for destination i
      const row = [];
      for (let k = 0; k < T; k++) row.push([a[i * T + k], k]);
      row.sort((p, q) => q[0] - p[0]);
      for (let t = 0; t < K; t++) {
        const [w, j] = row[t];
        edges.push([j, i, round3(w)]);
      }
    }
    layers.push({ layer: l, nodes, edges });
  }
  return {
    num_layers: L,
    num_tokens: T,
    k: K,
    grid: GRID,
    cls_index: 0,
    seed: 0,
    edge_semantics:
      "key->query; per destination token top-k head-averaged attention; weight rounded to 3 decimals",
    residual: {
      kind: "identity",
      materialized: false,
      weight: 1.0,
      count: T * (L - 1),
      description:
        "Unrolled residual-stream edges are implicit; synthesize (t,i)->(t+1,i) identity edges.",
    },
    layers,
  };
}

function conceptsJson(salient) {
  const layer = 9;
  const topk = 8;
  const nConcepts = 4096;
  const feature_ids = [];
  const activations = [];
  const rng = mulberry32(99);
  // assign a small stable set of feature ids; salient tokens share "forest" concepts
  for (let i = 0; i < T; i++) {
    const ids = [];
    const acts = [];
    const s = i === 0 ? 0.5 : salient[i - 1];
    for (let t = 0; t < topk; t++) {
      // salient patches map onto a shared concept cluster (ids 100..107)
      const fid = s > 0.4 ? 100 + t : ((Math.floor(rng() * nConcepts)) % nConcepts);
      ids.push(fid);
      acts.push(round3(s * (1 - t * 0.08) + rng() * 0.05));
    }
    feature_ids.push(ids);
    activations.push(acts);
  }
  return {
    layer,
    topk,
    dictionary_id: "eurosat-deit-s-l9-sae",
    provider_kind: "sae_topk",
    n_concepts: nConcepts,
    num_tokens: T,
    feature_ids,
    activations,
  };
}

function gaussiansBin(rng, salient, base, attnLayers) {
  // [S, T, 12] float16, channel order == GAUSS_CHANNELS
  const C = GAUSS_CHANNELS.length;
  const out = new Float32Array(S * T * C);
  const r0 = 0.5 / GRID;
  // precompute halo (column-sum attention received) per layer, normalized later
  for (let t = 0; t < S; t++) {
    const attnLayer = t === 0 ? null : attnLayers[t - 1];
    for (let i = 0; i < T; i++) {
      const o = (t * T + i) * C;
      let x, y;
      if (i === 0) {
        x = 0.0;
        y = 0.0;
      } else {
        const [row, col] = patchRowCol(i);
        x = (col + 0.5) / GRID;
        y = (row + 0.5) / GRID;
      }
      const s = i === 0 ? 0.3 : salient[i - 1];
      // color
      const cr = Math.min(1, base[0] / 255 + s * 0.2);
      const cg = Math.min(1, base[1] / 255 + s * 0.15);
      const cb = Math.min(1, base[2] / 255);
      const depth = t / (S - 1);
      let halo = 0;
      if (attnLayer && i > 0) {
        for (let q = 0; q < T; q++) halo += attnLayer[q * T + i];
        halo = Math.min(1, halo / 8);
      }
      out[o + 0] = x;
      out[o + 1] = y;
      out[o + 2] = r0 * (1 + s * 0.5); // rx
      out[o + 3] = r0 / (1 + s * 0.3); // ry
      out[o + 4] = (rng() - 0.5) * 0.6; // theta
      out[o + 5] = cr;
      out[o + 6] = cg;
      out[o + 7] = cb;
      out[o + 8] = i === 0 ? 0.4 : Math.min(1, 0.2 + s * depth * 1.2); // opacity
      out[o + 9] = t === 0 ? 0 : Math.min(1, s * depth * 1.1); // glow
      out[o + 10] = halo;
      out[o + 11] = s; // activation_raw
    }
  }
  return float16Buffer(Array.from(out));
}

// --------------------------------------------------------------------------- //
// manifest assembly
// --------------------------------------------------------------------------- //

function buildFullPack(dir, opts) {
  mkdirSync(dir, { recursive: true });
  const rng = mulberry32(opts.seed);
  const salient = saliencyField(mulberry32(opts.seed + 1), opts.cx, opts.cy, 22);
  const assets = {};

  // attention.bin (per_row_uint8)
  const attnLayers = [];
  const attnFlat = new Float32Array(L * H * T * T);
  for (let l = 0; l < L; l++) {
    const rows = attentionForLayer(mulberry32(opts.seed + 10 + l), l, salient);
    attnLayers.push(rows);
    for (let h = 0; h < H; h++) {
      // heads are slight perturbations of the head-averaged map
      for (let idx = 0; idx < T * T; idx++) {
        attnFlat[((l * H + h) * T * T) + idx] = rows[idx];
      }
    }
  }
  const { data, scales } = quantizePerRow(attnFlat, T);
  const attnBuf = Buffer.concat([data, scales]);
  writeFileSync(join(dir, "attention.bin"), attnBuf);
  assets["attention.bin"] = {
    dtype: "uint8",
    shape: [L, H, T, T],
    encoding: "per_row_uint8",
    bytes: attnBuf.length,
    quant: {
      scheme: "per_row_uint8",
      row_axis: -1,
      scale_dtype: "float32",
      data_offset: 0,
      data_bytes: data.length,
      scale_offset: data.length,
      scale_count: scales.length / 4,
    },
  };

  // tokens.bin (fp16 [S,T,D])
  const tokVals = new Float32Array(S * T * D);
  for (let t = 0; t < S; t++)
    for (let i = 0; i < T; i++)
      for (let d = 0; d < D; d++)
        tokVals[(t * T + i) * D + d] = (rng() - 0.5) * (0.5 + (i === 0 ? 0.5 : salient[i - 1]));
  const tokBuf = float16Buffer(Array.from(tokVals));
  writeFileSync(join(dir, "tokens.bin"), tokBuf);
  assets["tokens.bin"] = { dtype: "float16", shape: [S, T, D], encoding: "raw", bytes: tokBuf.length };

  // attributions
  const attrs = attributionAssets(mulberry32(opts.seed + 3), salient);
  const cheferBuf = float32Buffer(Array.from(attrs.chefer));
  const rolloutBuf = float32Buffer(Array.from(attrs.rollout));
  const gradcamBuf = float32Buffer(Array.from(attrs.gradcam));
  const igBuf = float32Buffer(Array.from(attrs.ig));
  writeFileSync(join(dir, "attr_chefer.bin"), cheferBuf);
  writeFileSync(join(dir, "attr_rollout.bin"), rolloutBuf);
  writeFileSync(join(dir, "attr_gradcam.bin"), gradcamBuf);
  writeFileSync(join(dir, "attr_ig.bin"), igBuf);
  assets["attr_chefer.bin"] = { dtype: "float32", shape: [L, T], encoding: "raw", bytes: cheferBuf.length };
  assets["attr_rollout.bin"] = { dtype: "float32", shape: [L, T], encoding: "raw", bytes: rolloutBuf.length };
  assets["attr_gradcam.bin"] = { dtype: "float32", shape: [GRID, GRID], encoding: "raw", bytes: gradcamBuf.length };
  assets["attr_ig.bin"] = { dtype: "float32", shape: [T], encoding: "raw", bytes: igBuf.length };

  const attrIndex = {
    chefer: { asset: "attr_chefer.bin", kind: "per_layer_tokens" },
    rollout: { asset: "attr_rollout.bin", kind: "per_layer_tokens" },
    gradcam: { asset: "attr_gradcam.bin", kind: "token_grid" },
    ig: { asset: "attr_ig.bin", kind: "tokens" },
  };
  writeJson(dir, "attributions.json", attrIndex, assets);

  // faithfulness
  writeJson(dir, "faithfulness.json", faithfulnessJson(attrs, salient), assets);

  // image
  const img = writePng(dir, opts.cx, opts.cy, opts.color, opts.seed + 5);
  assets[img.filename] = { dtype: "png", shape: [224, 224], encoding: "png", bytes: img.bytes };

  // gaussians.bin
  const gBuf = gaussiansBin(mulberry32(opts.seed + 6), salient, opts.color, attnLayers);
  writeFileSync(join(dir, "gaussians.bin"), gBuf);
  assets["gaussians.bin"] = {
    dtype: "float16",
    shape: [S, T, GAUSS_CHANNELS.length],
    encoding: "raw",
    bytes: gBuf.length,
    meta: {
      channels: GAUSS_CHANNELS,
      layout: `S=${S},N=${T},C=${GAUSS_CHANNELS.length} float16 C-order`,
      n_steps: S,
      n_tokens: T,
      grid: GRID,
      cls_index: 0,
      cls_position: [0.0, 0.0],
      ecc_max: 2.5,
      attribution: "chefer",
    },
  };

  // graph.json
  writeJson(dir, "graph.json", graphJson(attnLayers), assets, {
    k: 8,
    num_layers: L,
    residual: "implicit; see graph.json residual flag",
  });

  // concepts.json (only when requested)
  if (opts.concepts) {
    writeJson(dir, "concepts.json", conceptsJson(salient), assets, {
      layer: 9,
      dictionary_id: "eurosat-deit-s-l9-sae",
      provider_kind: "sae_topk",
      additive: "top-k SAE features per token; see §9",
    });
  }

  writeManifest(dir, opts, assets, img.filename);
}

function buildLightPack(dir, opts) {
  // Image Space only needs: manifest, image, attributions, faithfulness.
  mkdirSync(dir, { recursive: true });
  const salient = saliencyField(mulberry32(opts.seed + 1), opts.cx, opts.cy, 22);
  const assets = {};
  const attrs = attributionAssets(mulberry32(opts.seed + 3), salient);
  const cheferBuf = float32Buffer(Array.from(attrs.chefer));
  const rolloutBuf = float32Buffer(Array.from(attrs.rollout));
  const gradcamBuf = float32Buffer(Array.from(attrs.gradcam));
  const igBuf = float32Buffer(Array.from(attrs.ig));
  writeFileSync(join(dir, "attr_chefer.bin"), cheferBuf);
  writeFileSync(join(dir, "attr_rollout.bin"), rolloutBuf);
  writeFileSync(join(dir, "attr_gradcam.bin"), gradcamBuf);
  writeFileSync(join(dir, "attr_ig.bin"), igBuf);
  assets["attr_chefer.bin"] = { dtype: "float32", shape: [L, T], encoding: "raw", bytes: cheferBuf.length };
  assets["attr_rollout.bin"] = { dtype: "float32", shape: [L, T], encoding: "raw", bytes: rolloutBuf.length };
  assets["attr_gradcam.bin"] = { dtype: "float32", shape: [GRID, GRID], encoding: "raw", bytes: gradcamBuf.length };
  assets["attr_ig.bin"] = { dtype: "float32", shape: [T], encoding: "raw", bytes: igBuf.length };
  writeJson(dir, "attributions.json", {
    chefer: { asset: "attr_chefer.bin", kind: "per_layer_tokens" },
    rollout: { asset: "attr_rollout.bin", kind: "per_layer_tokens" },
    gradcam: { asset: "attr_gradcam.bin", kind: "token_grid" },
    ig: { asset: "attr_ig.bin", kind: "tokens" },
  }, assets);
  writeJson(dir, "faithfulness.json", faithfulnessJson(attrs, salient), assets);
  const img = writePng(dir, opts.cx, opts.cy, opts.color, opts.seed + 5);
  assets[img.filename] = { dtype: "png", shape: [224, 224], encoding: "png", bytes: img.bytes };
  writeManifest(dir, opts, assets, img.filename);
}

function writeJson(dir, name, payload, assets, meta) {
  const data = Buffer.from(JSON.stringify(payload), "utf-8");
  writeFileSync(join(dir, name), data);
  assets[name] = { dtype: "json", shape: [], encoding: "json", bytes: data.length, ...(meta ? { meta } : {}) };
}

function writeManifest(dir, opts, assets, imageFile) {
  const manifest = {
    pack_version: "1.0.0",
    model: {
      arch: "deit_small_patch16_224",
      hf_repo: opts.hf_repo,
      num_layers: L,
      num_heads: H,
      num_tokens: T,
      embed_dim: D,
      patch_size: 16,
    },
    dataset: {
      name: opts.dataset.name,
      display_name: opts.dataset.display_name,
      num_classes: opts.dataset.class_names.length,
      class_names: opts.dataset.class_names,
    },
    image: { id: opts.imageId, width: 224, height: 224, source: "gallery" },
    prediction: opts.prediction,
    assets,
    timings: {
      predict_ms: 138.4,
      attention_ms: 84.1,
      chefer_ms: 301.7,
      gradcam_ms: 47.9,
      ig_ms: 2110.5,
      faithfulness_ms: 1788.2,
    },
  };
  writeFileSync(join(dir, "manifest.json"), Buffer.from(JSON.stringify(manifest, null, 2), "utf-8"));
}

// --------------------------------------------------------------------------- //
// datasets + gallery (mock DB)
// --------------------------------------------------------------------------- //

const EUROSAT_CLASSES = [
  "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
  "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
];
const PET_CLASSES = ["Abyssinian", "Beagle", "Bengal", "Boxer", "Persian", "Ragdoll", "Samoyed", "Sphynx"];

function probs(nClasses, top, conf) {
  const p = new Array(nClasses).fill((1 - conf) / (nClasses - 1));
  p[top] = conf;
  return p.map((v) => Math.round(v * 1e4) / 1e4);
}

function main() {
  rmSync(MOCK, { recursive: true, force: true });
  mkdirSync(join(MOCK, "packs"), { recursive: true });

  const gallery = { eurosat: [], oxford_pet: [] };

  // --- EuroSAT: one FULL reference pack + two light packs -------------------
  const euro = {
    name: "eurosat",
    display_name: "EuroSAT — land use from Sentinel-2",
    class_names: EUROSAT_CLASSES,
  };
  buildFullPack(join(MOCK, "packs", "eurosat", "eurosat_forest_00123"), {
    seed: 1, cx: 7, cy: 6, color: [46, 120, 60],
    imageId: "eurosat_forest_00123", dataset: euro, hf_repo: "vitreous/deit-small-eurosat",
    concepts: true,
    prediction: { label: "Forest", class_index: 1, confidence: 0.9731, probabilities: probs(10, 1, 0.9731) },
  });
  buildLightPack(join(MOCK, "packs", "eurosat", "eurosat_residential_00045"), {
    seed: 2, cx: 5, cy: 9, color: [140, 120, 110],
    imageId: "eurosat_residential_00045", dataset: euro, hf_repo: "vitreous/deit-small-eurosat",
    prediction: { label: "Residential", class_index: 7, confidence: 0.9123, probabilities: probs(10, 7, 0.9123) },
  });
  buildLightPack(join(MOCK, "packs", "eurosat", "eurosat_river_00210"), {
    seed: 3, cx: 9, cy: 4, color: [40, 90, 130],
    imageId: "eurosat_river_00210", dataset: euro, hf_repo: "vitreous/deit-small-eurosat",
    prediction: { label: "River", class_index: 8, confidence: 0.8477, probabilities: probs(10, 8, 0.8477) },
  });
  gallery.eurosat = [
    galleryRow("eurosat", "eurosat-deit-s", "eurosat_forest_00123", "Forest", "Forest", 0.9731, ["reference", "full-pack"]),
    galleryRow("eurosat", "eurosat-deit-s", "eurosat_residential_00045", "Residential", "Residential", 0.9123, ["light"]),
    galleryRow("eurosat", "eurosat-deit-s", "eurosat_river_00210", "River", "River", 0.8477, ["light"]),
  ];

  // --- Oxford Pet: one light pack, NO concepts (graceful-absence proof) ------
  const pet = {
    name: "oxford_pet",
    display_name: "Oxford-IIIT Pet — fine-grained breeds",
    class_names: PET_CLASSES,
  };
  buildLightPack(join(MOCK, "packs", "oxford_pet", "oxford_pet_beagle_0007"), {
    seed: 11, cx: 7, cy: 7, color: [150, 120, 90],
    imageId: "oxford_pet_beagle_0007", dataset: pet, hf_repo: "vitreous/deit-small-oxfordpet",
    prediction: { label: "Beagle", class_index: 1, confidence: 0.8894, probabilities: probs(8, 1, 0.8894) },
  });
  gallery.oxford_pet = [
    galleryRow("oxford_pet", "pet-deit-s", "oxford_pet_beagle_0007", "Beagle", "Beagle", 0.8894, ["light", "no-concepts"]),
  ];

  const datasetsDoc = {
    generated_by: "scripts/gen-mock-pack.mjs",
    storage: { bucket: "packs" },
    datasets: [
      { id: "eurosat", name: "eurosat", display_name: euro.display_name, num_classes: 10, class_names: EUROSAT_CLASSES, model_id: "eurosat-deit-s", arch: "deit_small_patch16_224" },
      { id: "oxford_pet", name: "oxford_pet", display_name: pet.display_name, num_classes: 8, class_names: PET_CLASSES, model_id: "pet-deit-s", arch: "deit_small_patch16_224" },
    ],
    gallery,
    projections: [
      { id: "proj-eurosat-l12-umap", dataset_id: "eurosat", model_id: "eurosat-deit-s", layer: 12, method: "umap", url: null, reducer_url: null, note: "coords omitted from M5 mock; M7 renders" },
    ],
  };
  writeFileSync(join(MOCK, "datasets.json"), Buffer.from(JSON.stringify(datasetsDoc, null, 2), "utf-8"));
  console.log("mock pack fixture written to", MOCK);
}

function galleryRow(datasetId, modelId, imageId, classLabel, predLabel, confidence, tags) {
  return {
    id: imageId,
    dataset_id: datasetId,
    model_id: modelId,
    class_label: classLabel,
    pred_label: predLabel,
    confidence,
    pack_prefix: `${datasetId}/${imageId}/`,
    thumb_url: `/mock/packs/${datasetId}/${imageId}/image.png`,
    tags,
  };
}

main();
