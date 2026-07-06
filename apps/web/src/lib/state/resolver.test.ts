import { describe, expect, it } from "vitest";
import type { PackManifest } from "@vitreous/schema";
import type { ConceptsJson } from "../pack/types";
import { buildPackIndex } from "./packIndex";
import { resolve, resolvedPatches } from "./resolver";

function manifest(): PackManifest {
  return {
    pack_version: "1.0.0",
    model: { arch: "deit_small_patch16_224", hf_repo: "x", num_layers: 12, num_heads: 6, num_tokens: 197, embed_dim: 384, patch_size: 16 },
    dataset: { name: "eurosat", num_classes: 10, class_names: [] },
    image: { id: "img_a", width: 224, height: 224, source: "gallery" },
    prediction: { label: "Forest", class_index: 1, confidence: 0.9, probabilities: [] },
    assets: {},
    timings: {},
  };
}

// concepts.json where token 5 and token 200(>T ignored) fire feature 100;
// token 5 also fires 101; token 6 fires 100.
function concepts(): ConceptsJson {
  const feature_ids: number[][] = Array.from({ length: 197 }, () => [0]);
  const activations: number[][] = Array.from({ length: 197 }, () => [0]);
  feature_ids[5] = [100, 101];
  feature_ids[6] = [100];
  activations[5] = [0.9, 0.7];
  activations[6] = [0.8];
  return { layer: 9, topk: 2, dictionary_id: "dict-1", provider_kind: "sae_topk", n_concepts: 4096, num_tokens: 197, feature_ids, activations };
}

describe("buildPackIndex", () => {
  it("derives grid from token count", () => {
    const idx = buildPackIndex(manifest());
    expect(idx.grid).toBe(14);
    expect(idx.numTokens).toBe(197);
    expect(idx.numLayers).toBe(12);
    expect(idx.conceptLayer).toBeNull();
  });

  it("builds token↔concept maps when concepts present", () => {
    const idx = buildPackIndex(manifest(), concepts());
    expect(idx.conceptLayer).toBe(9);
    expect(idx.tokenConcepts[5]).toEqual(["100", "101"]);
    expect(idx.conceptTokens.get("100")).toEqual([5, 6]);
    expect(idx.conceptTokens.get("101")).toEqual([5]);
  });
});

describe("resolve", () => {
  const idx = buildPackIndex(manifest(), concepts());

  it("links a token to patch, gaussian, node, and concepts", () => {
    // token 6 -> patch divmod(5,14) = (0,5)
    const r = resolve({ kind: "token", layer: 3, idx: 6 }, idx);
    expect(r.patch).toEqual({ row: 0, col: 5 });
    expect(r.gaussian).toEqual({ layer: 3, idx: 6 });
    expect(r.node).toBe("L3_T6");
    expect(r.isCls).toBe(false);
    expect(r.point).toBeNull();
    expect(r.concepts).toEqual(["100"]);
    // linked set contains each kind
    const kinds = new Set(r.refs.map((x) => x.kind));
    expect(kinds).toEqual(new Set(["token", "gaussian", "node", "patch", "concept"]));
  });

  it("maps CLS (idx 0) to an embedding point, no patch", () => {
    const r = resolve({ kind: "token", layer: 6, idx: 0 }, idx);
    expect(r.isCls).toBe(true);
    expect(r.patch).toBeNull();
    expect(r.point).toEqual({ imageId: "img_a", layer: 6 });
    expect(r.refs.some((x) => x.kind === "point")).toBe(true);
  });

  it("patch -> token resolves at the given layer", () => {
    // patch (0,5) -> token 6
    const r = resolve({ kind: "patch", row: 0, col: 5 }, idx, 9);
    expect(r.idx).toBe(6);
    expect(r.token).toEqual({ layer: 9, idx: 6 });
    expect(r.node).toBe("L9_T6");
  });

  it("patch round-trips through token consistently", () => {
    for (const [row, col] of [[0, 0], [3, 7], [13, 13], [7, 2]]) {
      const r = resolve({ kind: "patch", row, col }, idx, 4);
      expect(r.patch).toEqual({ row, col });
    }
  });

  it("node id resolves back to the same token", () => {
    const r = resolve({ kind: "node", id: "L5_T6" }, idx);
    expect(r.token).toEqual({ layer: 5, idx: 6 });
    expect(r.patch).toEqual({ row: 0, col: 5 });
  });

  it("gaussian resolves identically to its token (idx equality)", () => {
    const rt = resolve({ kind: "token", layer: 2, idx: 42 }, idx);
    const rg = resolve({ kind: "gaussian", layer: 2, idx: 42 }, idx);
    expect(rg.token).toEqual(rt.token);
    expect(rg.patch).toEqual(rt.patch);
    expect(rg.node).toEqual(rt.node);
  });

  it("concept fans out to every firing token at the probe layer", () => {
    const r = resolve({ kind: "concept", id: "100" }, idx);
    expect(r.layer).toBe(9);
    expect(r.concepts).toEqual(["100"]);
    const tokenIdxs = r.refs.filter((x) => x.kind === "token").map((x) => (x as { idx: number }).idx).sort((a, b) => a - b);
    expect(tokenIdxs).toEqual([5, 6]);
    // each firing token contributes a patch too
    expect(r.refs.filter((x) => x.kind === "patch").length).toBe(2);
  });

  it("malformed node id resolves to empty", () => {
    const r = resolve({ kind: "node", id: "not-a-node" }, idx);
    expect(r.token).toBeNull();
    expect(r.refs).toEqual([]);
  });

  it("resolvedPatches returns the lit cells for any ref", () => {
    const cells = resolvedPatches({ kind: "token", layer: 1, idx: 6 }, idx, 1);
    expect(cells.has("0:5")).toBe(true);
    expect(resolvedPatches(null, idx).size).toBe(0);
  });
});
