/**
 * Regression test for the M5 "pack fails to load in mock mode" bug.
 *
 * The store constructs `new PackClient(url)` with NO fetchImpl and later calls
 * it as `this.fetchImpl(...)`. In the browser the global `fetch` throws
 * "Illegal invocation" when invoked with `this` bound to anything but the realm
 * global — so the whole selectImage chain threw and the UI showed status
 * "error" / "awaiting pack". Node's fetch is lenient, so to catch this without a
 * browser we install a STRICT global fetch that emulates the browser's binding
 * check: it throws unless called with `this === globalThis` (or undefined).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import type { GalleryImageRow } from "../db/types";

const PUBLIC = fileURLToPath(new URL("../../../public", import.meta.url));

function fileResponse(pathname: string, range?: string): Response {
  const file = `${PUBLIC}${pathname}`;
  let bytes: Buffer;
  try {
    bytes = readFileSync(file);
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
      const slice = bytes.subarray(Number(m[1]), Number(m[2]) + 1);
      const ab = slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
      return {
        ok: false,
        status: 206,
        statusText: "Partial Content",
        async arrayBuffer() {
          return ab;
        },
        async json() {
          throw new Error("range json");
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

beforeAll(() => {
  process.env.NEXT_PUBLIC_VITREOUS_MOCK = "1";
  // Strict, browser-like fetch: reject an illegal `this` (regression guard).
  function strictFetch(this: unknown, input: RequestInfo | URL, init?: RequestInit) {
    if (this !== undefined && this !== globalThis) {
      throw new TypeError("Failed to execute 'fetch' on 'Window': Illegal invocation");
    }
    const url = typeof input === "string" ? input : input.toString();
    const pathname = url.startsWith("http") ? new URL(url).pathname : url;
    const range = (init?.headers as Record<string, string> | undefined)?.["Range"];
    return Promise.resolve(fileResponse(pathname, range));
  }
  (globalThis as unknown as { fetch: typeof fetch }).fetch = strictFetch as unknown as typeof fetch;
});

describe("useWorkbench.selectImage (mock mode, strict browser-like fetch)", () => {
  it("loads the first eurosat pack without throwing Illegal invocation", async () => {
    const { useWorkbench } = await import("./store");
    const image: GalleryImageRow = {
      id: "eurosat_forest_00123",
      dataset_id: "eurosat",
      model_id: "eurosat-deit-s",
      class_label: "Forest",
      pred_label: "Forest",
      confidence: 0.9731,
      pack_prefix: "eurosat/eurosat_forest_00123/",
      thumb_url: "/mock/packs/eurosat/eurosat_forest_00123/image.png",
      tags: [],
    };
    await useWorkbench.getState().selectImage("eurosat", image);
    const st = useWorkbench.getState();
    expect(st.error).toBeNull();
    expect(st.manifest).not.toBeNull();
    expect(st.packIndex?.numTokens).toBe(197);
    expect(st.packIndex?.numLayers).toBe(12);
    expect(st.packIndex?.conceptLayer).toBe(9);
  });
});
