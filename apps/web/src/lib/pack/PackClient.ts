/**
 * PackClient — the web data layer's typed reader for one Explanation Pack (§5).
 *
 * Given a `packUrl` (the pack directory's public base URL, WITH trailing slash;
 * built by the db layer from `StorageAdapter.get_url` + `pack_prefix`), it:
 *   1. fetches manifest.json first (small first paint, ~100 KB),
 *   2. lazily fetches individual binary assets via HTTP Range requests using the
 *      manifest asset index (bytes + quant offsets), falling back to a full GET
 *      when the host ignores Range (see range.ts),
 *   3. decodes each into typed arrays keyed off the frozen `@vitreous/schema`
 *      types and the frozen binary layouts.
 *
 * All decode logic is the exact inverse of packages/core's PackWriter, so a
 * pack produced by Kaggle batch, the live CPU service, or the mock generator
 * reads identically.
 */
import type { AssetEntry, PackManifest } from "@vitreous/schema";
import { decodeFloat16 } from "./fp16";
import { dequantizePerRow } from "./dequant";
import { fetchRange } from "./range";
import type {
  AttributionsIndex,
  ConceptsJson,
  FaithfulnessJson,
  GraphJson,
  LoadedAttention,
  LoadedAttribution,
  LoadedGaussians,
  LoadedTokens,
} from "./types";

export interface PackClientOptions {
  /** Injectable fetch (tests / SSR). Defaults to global fetch. */
  fetchImpl?: typeof fetch;
}

function product(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export class PackClient {
  readonly packUrl: string;
  private readonly fetchImpl: typeof fetch;
  private manifestCache: PackManifest | null = null;
  private attrIndexCache: AttributionsIndex | null = null;

  constructor(packUrl: string, opts: PackClientOptions = {}) {
    // Normalize to a trailing slash so asset names append cleanly.
    this.packUrl = packUrl.endsWith("/") ? packUrl : `${packUrl}/`;
    // Bind the default global fetch to its realm: the browser's `fetch` throws
    // "Illegal invocation" if called with `this` set to anything but the global
    // object, and we invoke it as `this.fetchImpl(...)`. (Node/undici's fetch is
    // lenient, so this only bites in the browser.)
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  assetUrl(name: string): string {
    return `${this.packUrl}${name}`;
  }

  // -- manifest ------------------------------------------------------------- //

  async loadManifest(): Promise<PackManifest> {
    if (this.manifestCache) return this.manifestCache;
    const res = await this.fetchImpl(this.assetUrl("manifest.json"), {
      redirect: "follow",
    });
    if (!res.ok) {
      throw new Error(`manifest.json: ${res.status} ${res.statusText}`);
    }
    const manifest = (await res.json()) as PackManifest;
    this.manifestCache = manifest;
    return manifest;
  }

  private entry(manifest: PackManifest, name: string): AssetEntry {
    const e = manifest.assets[name];
    if (!e) throw new Error(`no asset ${name} in pack ${this.packUrl}`);
    return e;
  }

  has(manifest: PackManifest, name: string): boolean {
    return name in manifest.assets;
  }

  // -- raw binary assets ---------------------------------------------------- //

  /** Range-fetch the raw bytes of an asset (exact `bytes` from the manifest). */
  private async fetchAsset(entry: AssetEntry, name: string): Promise<ArrayBuffer> {
    const { buffer } = await fetchRange(
      this.assetUrl(name),
      0,
      entry.bytes,
      this.fetchImpl,
    );
    return buffer;
  }

  async loadJson<T>(name: string): Promise<T> {
    const res = await this.fetchImpl(this.assetUrl(name), { redirect: "follow" });
    if (!res.ok) throw new Error(`${name}: ${res.status} ${res.statusText}`);
    return (await res.json()) as T;
  }

  // -- typed loaders -------------------------------------------------------- //

  /** Dequantized attention [L,H,T,T] float32 (§5). */
  async loadAttention(manifest?: PackManifest): Promise<LoadedAttention> {
    const m = manifest ?? (await this.loadManifest());
    const entry = this.entry(m, "attention.bin");
    if (entry.encoding !== "per_row_uint8" || !entry.quant) {
      throw new Error("attention.bin is not per_row_uint8 quantized");
    }
    const blob = await this.fetchAsset(entry, "attention.bin");
    const [layers, heads, tokens] = entry.shape;
    const data = dequantizePerRow(blob, entry.quant, entry.shape[entry.shape.length - 1]);
    return { data, layers, heads, tokens };
  }

  /** Token embeddings [L+1,T,D] fp16 -> float32 (§5). */
  async loadTokens(manifest?: PackManifest): Promise<LoadedTokens> {
    const m = manifest ?? (await this.loadManifest());
    const entry = this.entry(m, "tokens.bin");
    const blob = await this.fetchAsset(entry, "tokens.bin");
    const data = decodeFloat16(blob, 0, product(entry.shape));
    const [steps, tokens, dim] = entry.shape;
    return { data, steps, tokens, dim };
  }

  /** Gaussian Feature Field [L+1,T,C] fp16 -> float32 + channel names (§7). */
  async loadGaussians(manifest?: PackManifest): Promise<LoadedGaussians> {
    const m = manifest ?? (await this.loadManifest());
    const entry = this.entry(m, "gaussians.bin");
    const blob = await this.fetchAsset(entry, "gaussians.bin");
    const data = decodeFloat16(blob, 0, product(entry.shape));
    const channels = ((entry.meta?.channels as string[] | undefined) ?? []).slice();
    const [steps, tokens, channelCount] = entry.shape;
    return { data, channels, steps, tokens, channelCount };
  }

  /** attributions.json index (cached). */
  async loadAttributionsIndex(manifest?: PackManifest): Promise<AttributionsIndex> {
    if (this.attrIndexCache) return this.attrIndexCache;
    const idx = await this.loadJson<AttributionsIndex>("attributions.json");
    this.attrIndexCache = idx;
    return idx;
  }

  /**
   * Load one attribution method's asset, decoded to a flat float32 array with
   * its declared shape/kind (§6). Raw fp32 assets: chefer/rollout [L,T],
   * gradcam [14,14], ig [T].
   */
  async loadAttribution(method: string, manifest?: PackManifest): Promise<LoadedAttribution> {
    const m = manifest ?? (await this.loadManifest());
    const index = await this.loadAttributionsIndex(m);
    const info = index[method];
    if (!info) throw new Error(`no attribution method ${method} in pack`);
    const entry = this.entry(m, info.asset);
    const blob = await this.fetchAsset(entry, info.asset);
    let data: Float32Array;
    if (entry.dtype === "float32") {
      data = new Float32Array(blob.slice(0, product(entry.shape) * 4));
    } else if (entry.dtype === "float16") {
      data = decodeFloat16(blob, 0, product(entry.shape));
    } else {
      throw new Error(`unsupported attribution dtype ${entry.dtype}`);
    }
    return {
      method,
      kind: info.kind,
      data,
      shape: entry.shape.slice(),
      pixelAsset: info.pixel_asset,
    };
  }

  /** graph.json (§8). */
  async loadGraph(): Promise<GraphJson> {
    return this.loadJson<GraphJson>("graph.json");
  }

  /** faithfulness.json (§10). */
  async loadFaithfulness(): Promise<FaithfulnessJson> {
    return this.loadJson<FaithfulnessJson>("faithfulness.json");
  }

  /**
   * concepts.json (§9) — may be ABSENT. Gates on the manifest asset index and
   * returns null gracefully when the pack carries no concept tier.
   */
  async loadConcepts(manifest?: PackManifest): Promise<ConceptsJson | null> {
    const m = manifest ?? (await this.loadManifest());
    if (!this.has(m, "concepts.json")) return null;
    return this.loadJson<ConceptsJson>("concepts.json");
  }

  /** Public URL of the display image asset (image.webp or image.png). */
  imageUrl(manifest: PackManifest): string | null {
    for (const name of Object.keys(manifest.assets)) {
      if (/^image\.(webp|png|jpg|jpeg)$/i.test(name)) return this.assetUrl(name);
    }
    return null;
  }
}
