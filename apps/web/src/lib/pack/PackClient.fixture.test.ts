/**
 * Fixture round-trip regression test (M5). Constructs a PackClient with a
 * filesystem-backed `fetchImpl` that serves the committed mock fixture under
 * public/mock/packs/... exactly like a static host would (200 for JSON, 206 +
 * sliced bytes for Range requests). It then drives the EXACT selectImage chain
 * (loadManifest -> loadConcepts -> buildPackIndex) plus resolve()/attribution/
 * gaussians so any decode or schema mismatch surfaces here rather than only at
 * runtime in the browser.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { PackClient } from "./PackClient";
import { buildPackIndex } from "../state/packIndex";
import { resolve } from "../state/resolver";

const PACK_DIR = fileURLToPath(
  new URL(
    "../../../public/mock/packs/eurosat/eurosat_forest_00123/",
    import.meta.url,
  ),
);

/** A Response-like backed by an on-disk file, honoring Range like a real CDN. */
function fileResponse(path: string, range?: string): Response {
  let bytes: Buffer;
  try {
    bytes = readFileSync(path);
  } catch {
    return {
      ok: false,
      status: 404,
      statusText: "Not Found",
      async arrayBuffer() {
        return new ArrayBuffer(0);
      },
      async json() {
        throw new Error("404");
      },
    } as unknown as Response;
  }

  if (range) {
    const m = /bytes=(\d+)-(\d+)/.exec(range);
    if (m) {
      const start = Number(m[1]);
      const end = Number(m[2]); // inclusive
      const slice = bytes.subarray(start, end + 1);
      const ab = slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
      return {
        ok: false, // 206 is not in the 200-299 "ok" for our range fallback logic path
        status: 206,
        statusText: "Partial Content",
        async arrayBuffer() {
          return ab;
        },
        async json() {
          throw new Error("range on json");
        },
      } as unknown as Response;
    }
  }

  const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    async arrayBuffer() {
      return ab;
    },
    async json() {
      return JSON.parse(bytes.toString("utf8"));
    },
  } as unknown as Response;
}

const fetchImpl = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === "string" ? input : input.toString();
  // packUrl below is the on-disk directory; asset names append directly.
  const range = (init?.headers as Record<string, string> | undefined)?.["Range"];
  return fileResponse(url, range);
}) as unknown as typeof fetch;

describe("PackClient over the committed eurosat_forest fixture", () => {
  const client = new PackClient(PACK_DIR, { fetchImpl });

  it("runs the selectImage chain: loadManifest -> loadConcepts -> buildPackIndex", async () => {
    const manifest = await client.loadManifest();
    expect(manifest.image.id).toBe("eurosat_forest_00123");

    const concepts = await client.loadConcepts(manifest);
    expect(concepts).not.toBeNull();

    const packIndex = buildPackIndex(manifest, concepts);
    expect(packIndex.numTokens).toBe(197);
    expect(packIndex.numLayers).toBe(12);
    expect(packIndex.conceptLayer).toBe(9);

    // resolve a patch through the index (the §11 sync backbone)
    const r = resolve({ kind: "patch", row: 3, col: 4 }, packIndex, 11);
    expect(r.token).not.toBeNull();
    expect(r.node).toMatch(/^L\d+_T\d+$/);
  });

  it("decodes every binary asset the workbench reads (range-fetched)", async () => {
    const manifest = await client.loadManifest();

    const attn = await client.loadAttention(manifest);
    expect(attn.layers).toBe(12);
    expect(attn.data.length).toBe(12 * 6 * 197 * 197);
    expect(Number.isFinite(attn.data[0])).toBe(true);

    const gauss = await client.loadGaussians(manifest);
    expect(gauss.channels.length).toBe(12);
    expect(gauss.data.length).toBe(13 * 197 * 12);
    expect(Number.isFinite(gauss.data[gauss.data.length - 1])).toBe(true);

    for (const method of ["chefer", "rollout", "gradcam", "ig"]) {
      const a = await client.loadAttribution(method, manifest);
      const expected = a.shape.reduce((p, s) => p * s, 1);
      expect(a.data.length).toBe(expected);
      expect(Number.isFinite(a.data[0])).toBe(true);
    }

    const faith = await client.loadFaithfulness();
    expect(faith.deletion_auc).toBeTruthy();
    const graph = await client.loadGraph();
    expect(graph.layers.length).toBeGreaterThan(0);
  });
});
