# SGP — the SomGraphProvider project

**Design contract & implementation plan · v0.1 (draft) · 2026-07-12**

SGP bridges the two halves of this repo: it renders **UMT-ViT's learned 3-D
SOM** — the strongest *proven* piece of the experiment
([`UMT-VIT-REPORT.md`](./UMT-VIT-REPORT.md) §5.2) — inside **ViTreous's
synchronized, replayable workbench** — the strongest piece of the product
([`ARCHITECTURE.md`](./ARCHITECTURE.md) §8, §11). Both design docs already
reserve this slot: ViTreous's `GraphProvider` protocol was written to admit
non-ViT models, and UMT-ViT's architecture lists a "ViTreous
pack/GraphProvider adapter" as its first future extension (§11).

The one-line pitch: **today's brain view is honest but derived** (Louvain
communities, force-simulated layout); **the SOM is a native graph** — real
learned topology with ground-truth coordinates. SGP upgrades the graph
element from a visualization artifact to a rendering of something the model
actually learned, and adds the view UMT-ViT is missing: a single image
flowing through the learned map.

---

## 1. Requirements

| Dimension | Decision |
|---|---|
| Goal | Explore a UMT-ViT run's SOM as a first-class ViTreous graph: native lattice layout, U-matrix edge weights, per-image BMU activation replay, bidirectional neuron ↔ image-patch sync |
| Non-goal (v1) | Replacing the ViT token graph; UMT-ViT classification quality; training-epoch replay inside the workbench (stays in `/umtvit`) |
| Pack format | **`pack_version` stays `1.0.0`.** All SGP assets are additive (new asset names + `meta`), exactly like the M3 concepts tier — a pack without them is still valid, a reader that ignores them still works |
| Package boundaries | `packages/core` and `experiments/umtvit` **never import each other** (both contracts forbid it). The integration point is the Kaggle notebook, which installs both and passes plain numpy arrays across the boundary |
| Import discipline | The core builder is numpy-only (M0 rule: no torch at import). The web decoder reuses the existing fp16/raw loaders |
| Honesty rules | Every rendered quantity is measured: coordinates = the literal neuron lattice; edges = literal lattice adjacency; edge weights = measured weight-space distances (U-matrix); activation = actual BMU assignments; dead neurons shown, never hidden. The Z axis is labeled *"learned hierarchy (encoder depth)"*, never physical depth |
| Compute | Everything client-side and tiny: the full SGP asset set for one image is **< 30 KB** (see §4). No backend changes required for the mock path |

## 2. What the user sees (the target experience)

1. Pick a UMT-ViT gallery image (or the bundled shapes demo) in the workbench.
2. The brain pane shows the **SOM lattice** (default: 8×8×8 = 512 neurons)
   as an Obsidian-style graph whose *positions are the real grid* (soft
   isometric projection), edges lit by weight-space similarity — cluster
   boundaries (high U-matrix ridges) read as dark gaps, exactly like the
   `/umtvit` U-matrix panel but as a living graph.
3. Press play: the transport clock `t` now scrubs **encoder depth `z = 0…Z-1`**.
   At each depth, the neurons that the image's voxels map to (their BMUs)
   light up; the activation EMA + traveling-pulse language from BrainView is
   reused unchanged. Watching the hot region migrate across the map as `z`
   advances *is* the inference replay — on native geometry.
4. Hover a neuron → the image patches whose voxels land on it highlight in
   Image Space (and vice versa: hover a patch → its BMU trail lights across
   depths). Same `EntityRef` → resolver → subscriber pattern as every other
   view (§11); no view talks to another view directly.
5. Expert mode: hit-count sizing, dead-neuron markers, U-matrix ridge
   emphasis, per-Z small-multiple strata as an alternative to the isometric
   view.

## 3. Dataflow

```
Kaggle notebook (installs umtvit + vitreous — the ONLY place both exist)
  ├─ umtvit: trained model + Soft3DSOM
  ├─ per probe image: volume V[H',W',Z,C] ─▶ BMU map  argmin_k ‖v−w_k‖  → [Z,H',W'] uint16
  ├─ SOM statics: weights [K,C] · grid (Gz,Gy,Gx) · hits [K] · U-matrix [K]
  ▼  (plain numpy across the boundary)
vitreous.som.build_som_graph_asset(...)          packages/core, numpy-only
  ├─ som.json        nodes/edges/communities + provenance (§4)
  ├─ som_bmu.bin     [Z,H',W'] uint16 per-image assignment map
  └─ manifest.json   additive asset entries (pack_version 1.0.0)
  ▼
Supabase Storage / public mock fixture
  ▼
apps/web  PackClient.loadSom() ─▶ SomBrainView (BrainView's activation/pulse
          language on fixed lattice layout) ⇄ store/resolver (kind:"neuron")
```

## 4. Data contract (the heart of SGP)

### 4.1 `som.json` (additive pack asset)

```jsonc
{
  "provider": "som",                      // discriminator (vs. the ViT token graph)
  "grid": [8, 8, 8],                      // (Gz, Gy, Gx); K = Gz·Gy·Gx
  "num_neurons": 512,
  "depth_steps": 8,                       // Z of the BMU map = encoder depth
  "depth_semantics": "learned hierarchy (encoder depth), not physical depth",
  "volume_grid": [16, 16],                // (H', W') of the BMU map
  "adjacency": "6-connected",             // edge rule (v1: faces only; 26-conn optional)
  "nodes": [                              // one per neuron, index k = z·Gy·Gx + y·Gx + x
    { "idx": 0, "grid": [0,0,0], "hits": 41, "umatrix": 0.183, "community": 2, "dead": false }
  ],
  "edges": [ [k_a, k_b, weight] ],        // lattice-adjacent pairs; weight = similarity
  "edge_semantics": "w = 1/(1+‖w_a − w_b‖); lattice adjacency only — every edge is a real grid neighbor",
  "communities": { "method": "kmeans_weights", "k": 12, "seed": 0 },
  "provenance": { "run": "...", "dataset": "ham10000", "epoch": 30 }
}
```

Design notes:
- **`hits`** = training-set BMU win counts (the existing `hits_final` export)
  → node sizing; `dead` = `hits == 0` → shown, not hidden (honesty).
- **`umatrix`** = mean weight-space distance to lattice neighbors (already
  computed per epoch for `/umtvit`; SGP embeds the *final* one per node).
- **`community`** = seeded k-means over neuron weight vectors (v1). The
  U-matrix-watershed alternative is recorded as an S6 option; either way the
  method + seed are stamped in the JSON so it is reproducible.
- Sizes: 512 nodes + ~1.4k face edges ≈ **25 KB** pretty-printed.

### 4.2 `som_bmu.bin` (additive, per-image)

`[Z, H', W']` **uint16** raw C-order — the BMU index of every voxel of *this
image's* latent volume, per depth slice. Default `8×16×16×2 B = 4 KB`. This
single asset powers both directions of the sync:

- neuron → patches: all `(h,w)` at depth `z` with `bmu == k`;
- patch → neurons: the depth-trail `bmu[:, h', w']` (voxel grid `(H',W')`
  maps to the image plane the same way ViT patches do — `H'×W'` is a
  uniform grid over the input, so the existing `tokenToPatch` convention
  generalizes: SGP records the grid in `volume_grid` and the frontend maps
  patch (row,col) at 14×14 → nearest voxel cell at 16×16 by normalized
  coordinates, documented in `som.json.meta`).

Per-depth activation histograms (the analog of tokens.bin norms that
BrainView EMAs) are **derived client-side** from this map — no extra asset.

### 4.3 Optional assets (flag-gated at export)

- `som_weights.bin` `[K, C]` fp16 (~64 KB @ 512×64) — enables client-side
  "what does this neuron look like in weight space" extensions later; not
  needed by v1 rendering.
- `som_formation.json` — the per-epoch U-matrix stack already exported for
  `/umtvit`; carried only if the S6 epoch-replay mode lands.

### 4.4 TypeScript mirror

`apps/web/src/lib/pack/types.ts` gains `SomJson` / `LoadedSomBmu`;
`packages/schema` is **untouched** (the manifest schema already admits any
asset name; `som.json`'s own shape is validated by a defensive parser à la
`lib/umtvit.ts`, with field-naming errors, not a JSON Schema — same policy
as `graph.json` today).

## 5. Package placement & boundary rules

| Piece | Lives in | Depends on | Rationale |
|---|---|---|---|
| `vitreous/som.py` — `build_som_graph_asset(weights, grid, hits, bmu_maps)` + `som_umatrix()`, `som_communities()` | `packages/core` | numpy only | Follows `graph.py`'s pattern; M0-testable with a synthetic SOM; **no umtvit import** |
| `SomGraphProvider` (implements the existing `GraphProvider` Protocol for API completeness: `nodes/edges/communities` over a SOM-state object) | `packages/core` | numpy only | Keeps the promise that the frontend abstraction admits new model families |
| Export cell (compute BMU maps from a trained run, call the builder, write the pack) | `experiments/umtvit/notebooks/kaggle_umtvit.ipynb` | umtvit **and** vitreous (notebook-only) | The notebook is the sanctioned integration point; zero package-level coupling |
| `loadSom()` / `loadSomBmu()` | `apps/web` `PackClient` | existing raw/uint16 decode | Same additive pattern as `loadConcepts()` (returns `null` when absent) |
| `SomBrainView` | `apps/web` views | BrainView's `force.ts`/`activation.ts` utilities | Reuses the pulse/EMA/label language; layout is **fixed** (no force sim needed — see §6) |
| `EntityRef` extension | `apps/web` state | — | `{ kind: "neuron"; idx: number }` + resolver rules (§7) |

## 6. Rendering plan (SomBrainView)

- **Layout = the lattice, not a simulation.** Soft isometric projection of
  the `(Gz,Gy,Gx)` grid to 2-D (deterministic, pure function → unit-testable
  like `force.ts`). A *very* low-alpha jitter drift may reuse the existing
  "breathing" idiom, but resting positions are the true grid — this is the
  entire honesty upgrade over the force layout, so no force integration on
  node positions.
- **Edges**: lattice neighbors only, alpha ∝ similarity weight → U-matrix
  ridges appear as visually dark cluster boundaries for free.
- **Activation**: per frame, depth `z = layerForT(t, Z)`; target activation
  of neuron `k` = its share of BMU hits at slice `z` (from `som_bmu.bin`),
  EMA-blended exactly like BrainView; top-N pulses on edges whose endpoints
  are both hot. Verdict blending (green) is **omitted in v1** — UMT-ViT has
  no per-class evidence signal; honesty rule says don't fake one.
- **Transport**: the store's existing clock is reused; `packIndex.numSteps`
  for a SOM pack is `Z` (from `som.json.depth_steps`). The transport UI
  label switches from "layer" to "depth" via the provider discriminator.
- **Expert toggle**: per-Z strata small-multiples (the `/umtvit` SomPanel
  look) as the second mode, mirroring GraphView's layer/unrolled toggle.

## 7. Selection & sync (resolver rules)

New `EntityRef`: `{ kind: "neuron"; idx: number }` (`refKey` → `neuron:{idx}`).
Resolver additions (pure, O(1) with one precomputed reverse map per depth):

- `neuron → patches`: at the current depth `z`, all voxel cells with
  `bmu == idx`, mapped to image-plane patches (§4.2) → Image Space
  highlight; Gaussian field highlights the same patch set.
- `patch → neuron`: the patch's voxel cell's BMU at depth `z` (plus its
  full depth trail for the trajectory affordance).
- `neuron → embedding/concepts`: none in v1 (documented as not-applicable
  rather than approximated).

The reverse maps live in an SGP-extended `PackIndex` (built once per pack
load, like `tokenConcepts`).

## 8. Milestones

Repo-convention roadmap (cf. ViTreous M-, UMT-ViT U-rows). Each row lands as
reviewed commits with tests; S0–S3 are pure Python/TS with synthetic
fixtures (no GPU anywhere before S5).

| S | Deliverable | Key acceptance test |
|---|---|---|
| S0 | This contract reviewed/frozen; `som.json` + `som_bmu.bin` shapes fixed; naming (`SGP`) recorded in DECISION-LOG | Owner sign-off; asset names + shapes referenced from one place |
| **S1 ✅** | **`vitreous/som.py`** (U-matrix, seeded k-means communities, BMU maps, `build_som_graph_asset`, `SomGraphProvider`) + `tests/test_som.py` (16 tests) | **Done:** synthetic SOM — frozen neuron order, exact face-edge count, deterministic canonical communities, BMU exact-recovery, JSON-serializable asset, PackWriter round-trip of both assets. numpy-only (no torch/umtvit import) |
| **S5a ✅** | **`experiments/umtvit/notebooks/kaggle_umtvit_sgp.ipynb`** (+ `_build_sgp_nb.py` builder): trains/resumes UMT-ViT on HAM10000, builds SGP assets via the tested core, renders the lattice/U-matrix/BMU-replay inline, exports `som.json`+`som_bmu.bin` packs **and** a self-contained `sgp_ham10000.json` web bundle | **Done + executed by the owner on Kaggle (SGP run 2, T4, 30 epochs)** — bundle exported and loaded in `/sgp`. Findings folded back (see [`SGP-RUNS.md`](./SGP-RUNS.md)): bf16-FFT fix at the source + regression test, `SOM_GRID` default 6×6×6, eval suite wired in, dataset-wide hit pass |
| **S2 ✅** | Web decode as the standalone `/sgp` surface: `lib/sgp.ts` (typed schema + defensive field-naming parser + BMU derivations: per-depth activations, migration curve, neuron↔voxel maps) + demo fixture generated through the SAME Python core (`apps/web/scripts/gen-sgp-demo.py` → `public/sgp/demo.json`) | **Done:** 12 parser/derivation tests + 4 Python-fixture round-trip contract tests (drift alarm); parse-reject names the exact field |
| **S3 ✅** | `/sgp` Explorer: `SomLatticeView` (three.js Gaussian-splat lattice at REAL grid coordinates, U-matrix-weighted additive edges, auto-orbit + drag, hover pick, EMA'd depth-scrub activation, dead-neuron rings) + `BmuReplayPanel` (probe image with community overlay + migration bars) + `UMatrixPanel` (per-Z strata, U-matrix/hits toggle) under one workbench-style selection model | **Done:** 11 pure lattice-math tests (projection, bracketing, activation lerp, EMA, sizing); verified end-to-end headless (SwiftShader) — lattice renders, depth scrub moves the hot region, hover syncs lattice ↔ image overlay ↔ strata. Workbench (`/`) untouched |
| S4 | Resolver/state integration: `kind:"neuron"`, reverse maps in PackIndex, Image-Space patch highlight both directions | resolver unit tests (neuron↔patch at every depth); refKey/refsEqual cases |
| S5 | Notebook export cell (U6-notebook section): trained HAM10000 run → SGP pack; owner runs on Kaggle; demo bundle committed to the mock fixture; screenshots | Notebook nbformat-valid; exported pack passes `PackReader` + web parser; workbench session over a real run recorded |
| S6 (stretch) | Epoch-formation replay mode (`som_formation.json`), U-matrix-watershed communities, `som_weights.bin` extensions | Mode toggle without store changes; watershed vs k-means comparison note |

Sequencing: S0 → S1 → S2 → S3 → S4 serial; S5 after S3 (S4 not required for
the first visual); S6 open-ended.

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| The map is visually muddy on real HAM10000 features (probe 0.77 ≠ crisp clusters) | Med | Hit-count sizing + U-matrix edge shading carry structure even when communities are soft; the shapes demo fixture guarantees a legible default; worst case is still an honest picture |
| Depth axis underwhelms (Z-ordering was a *negative* result) | Med | SGP visualizes BMU *migration* across depth, which exists regardless of scale ordering; the depth label copy never claims a hierarchy emerged |
| Scope creep into the ViT graph path | Low | Provider discriminator + additive assets; S3 acceptance includes "ViT packs render exactly as before" |
| Boundary erosion (umtvit ↔ core imports) | Low | Integration only in the notebook; S1 import-purity test enforces numpy-only |
| Pack-format drift | Low | `pack_version` frozen; additive-asset policy identical to the proven concepts tier |

## 10. Explicit non-goals (v1)

Classification-quality claims for UMT-ViT · epoch replay in the workbench
(S6) · a JSON Schema for `som.json` (defensive parser, like `graph.json`) ·
Supabase schema changes (SGP packs are ordinary packs under the existing
`pack_prefix` model) · any change to the ViT token graph or BrainView
defaults.
