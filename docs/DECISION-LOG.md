# ViTreous — Decision Log & Orchestration Plan

Record of the requirements interview (2026-07-06) and the working agreement
for how the system gets built. Companion to
[`ARCHITECTURE.md`](./ARCHITECTURE.md) and [`RESEARCH.md`](./RESEARCH.md).

## Working agreement (set by the repo owner)

- **Claude Fable 5** acts as researcher, architect, and orchestrator only.
- **Claude Opus** agents write all implementation code, one roadmap
  milestone (M0–M9, ARCHITECTURE.md §16) per work order, against the
  architecture doc as binding contract.
- Plan precedes code: no milestone starts before its design section exists.
- All work lands on branch `claude/explainable-vit-research-qadg2i`;
  the orchestrator reviews agent commits before pushing.
- Legacy code (`hatchvision/`, `webapp/`, `notebooks/`, `scripts/`,
  `tests/`) is historical reference — read, never modified.

## Interview record

### Round 1 — load-bearing decisions

| Question | Answer | Consequence |
|---|---|---|
| Primary purpose | **Portfolio / demo piece** | Optimize polish, public deployability, zero-setup exploration; correctness still required (it demonstrates XAI skill) |
| Where does GPU inference run | **Hybrid: precompute + light live** | Explanation Packs precomputed per gallery image; small live path for uploads |
| Models in v1 | **Standard ViT (timm/HF)** | ViT-S/16; all published attribution methods apply directly; GraphProvider keeps model-agnosticism |
| Explainability depth | **Attention + attribution + faithfulness + concept tier (SAE/clustering)** | The deepest offered tier; roadmap stages it so attention+attribution ship first |

### Round 2 — visualization & live path

| Question | Answer | Consequence |
|---|---|---|
| v1 datasets | **EuroSAT** | Primary demo dataset (satellite land-use; visually distinctive, cheap to precompute) |
| Flagship view | **Gaussian Feature Field** | Gets the animation/interaction polish budget; anchors the demo narrative |
| Live upload path | Serverless GPU (Modal) — **superseded in round 4 by the $0 constraint** | Final: free CPU service (HF Spaces) running the same `vitreous` package |
| Graph node semantics/scale | **Both modes, toggled**: per-layer (~250 nodes) AND full unrolled (~2.4k nodes) | One WebGL renderer (Sigma.js/graphology) serving both modes |

### Round 3 — venues, storage, repo, scope

| Question | Answer | Consequence |
|---|---|---|
| Training/precompute venue | **Kaggle trains, precompute shared** | Kaggle notebooks fine-tune; the pack generator lives in `packages/core` and runs in batch and live identically |
| Artifact storage | HF Hub — **superseded in round 4** | Final: Supabase Storage + Postgres primary (HF dataset repo as overflow valve behind `StorageAdapter`) |
| Repo strategy | **Monorepo, legacy kept aside** | New `apps/`, `packages/`, `kaggle/`, `docs/` at root; legacy untouched |
| v1 definition | **Everything at once + easy dataset switching through the entire pipeline** | Gallery, uploads, faithfulness, concepts all in v1; dataset swap proven end-to-end with a second dataset |

### Round 4 — final parameters

| Question | Answer | Consequence |
|---|---|---|
| Swap-proof second dataset | **Oxford-IIIT Pet** | Maximal contrast with EuroSAT (fine-grained photographic vs. satellite) proves the adapter layer |
| Model size | **ViT-S/16** (DeiT-S weights via timm) | 197 tokens @224px, 12 layers, 6 heads; unrolled graph ~2.4k nodes; CPU-serveable |
| Budget | **$0.** GPU = Kaggle only; storage = Supabase; frontend = Vercel; other needs = free services | Overrides Modal (R2) and HF-Hub-primary (R3). Live path = HF Spaces free CPU; honest cold-start UI |
| Visual design | **Dark scientific instrument** | Near-black canvas, luminous WebGL marks, monospaced readouts |

## Standing defaults (set by orchestrator, overridable)

- Interaction targets: hover propagation < 1 frame; replay 60 fps
  (Gaussian field), ≥ 30 fps (unrolled graph); first gallery paint ~100 KB.
- Pack budget 3–6 MB/image (uint8 attention, fp16 tokens, zstd).
- Frontend libs: three.js (Gaussian field), Sigma.js + graphology (graph),
  regl-scatterplot (embeddings), Zustand (selection store), D3 (overlays).
- Live service: FastAPI, single worker, SSE staged progress, uploads
  ephemeral, 10 MB cap.
- SAE: layer 9, 4096 features, k=32; k-means fallback behind a quality gate.

## Milestone status

| M | Scope (see ARCHITECTURE.md §16) | Status |
|---|---|---|
| M0 | Monorepo scaffold, pack schema + codegen, web shell, live stub, CI | **Complete** (Opus agent, 2026-07-06; 30+1 tests pass, tsc/lint/build clean; reviewed & pushed) |
| M1 | Dataset adapters (EuroSAT, Oxford Pet, imagefolder), ViT-S/16 loader, Instrumenter | **Complete** (Opus agent, 2026-07-06; 61 tests pass incl. hook-purity + exact trace shapes; reviewed & pushed) |
| M2 | XAI suite (rollout, Chefer ICCV-2021 grad-weighted formulation, Grad-CAM, IG), faithfulness eval, PackWriter/PackReader, **pack format v1 frozen** | **Complete** (Opus agent, 2026-07-06; 87 tests pass, tsc clean; reviewed & pushed) |
| M3 | `vitreous.gaussians` (Gaussian Feature Field) + `vitreous.graph` (ViTTokenGraphProvider, seeded Louvain communities) + `vitreous.projections` (PCA/UMAP/t-SNE + persisted reducers); pack gains `gaussians.bin` + `graph.json` additively | **Complete** (Opus agent, 2026-07-06; 120 tests pass, tsc clean; `pack_version` unchanged at 1.0.0) |
| M4–M9 | — | Pending. Pack format v1 is FROZEN (`pack_version` 1.0.0); M4+ may only *add* assets to the open asset index, never change existing layouts. |

Technical notes carried forward:
- Attention is captured by *recomputing* softmax from each block's own qkv
  while the fused forward runs untouched — this is what keeps logits
  bit-identical with hooks attached. Keep recompute; do not un-fuse.
- `Trace` = detached CPU torch tensors: attention `[12,6,197,197]` (true
  softmax rows — safe for per-row uint8 quantization), tokens `[13,197,384]`
  (block inputs t=0..11 + final norm t=12), logits, timings.
- M2's Chefer relevance needs a grad-enabled capture variant (drop
  `no_grad`, don't detach on that path); hook structure already supports it.
  → Done: `Instrumenter.capture_with_grad()` (temporary unfused-forward swap,
  restored after; plain forward bit-identical before/after).
- Pack v1 binary layouts (frozen): attention.bin = uint8 data block +
  trailing per-row fp32 scales (offsets in `AssetEntry.quant`); tokens.bin
  raw fp16; attributions fp32 raw; IG pixel map as 8-bit PNG. zstd remains a
  schema-valid encoding for a later size pass — no format change needed.
- Chefer formulation: grad-weighted attention rollout (ICCV 2021 generic
  variant), recorded in pack meta as `grad_weighted_rollout_iccv2021`.
- Canonical grid mapping for M3+: token 0 = CLS; patches `[1:]` reshape to
  14×14 via `divmod(i-1, 14)`; `xai.eval.to_patch_vector` is the canonical
  [196] reducer; `xai._common.embed_tokens`/`run_from_tokens` are the
  sanctioned timm entry points — do not re-derive internals.
- M3 additive schema change (still `pack_version` 1.0.0): `AssetEntry` gained an
  optional free-form `meta` object (schema/Pydantic/TS all mirrored). Used to
  record the `gaussians.bin` channel order on its asset entry; existing v1
  assets omit it, so all M0–M2 packs stay valid. `meta` never describes the
  frozen binary layout (dtype/shape/encoding/quant still own that).
- M3 `gaussians.bin` = `[13][197][12]` fp16, C-order. Channel order (frozen for
  M6 renderer): `(x, y, rx, ry, theta, r, g, b, opacity, glow, halo,
  activation_raw)`. CLS (token 0) sits at reserved off-grid anchor (0.0, 0.0);
  eccentricity bounded to rx/ry ≤ 2.5 (area-preserving); step t=0 has neutral
  (isotropic, zero-glow/halo) attention. Ranges [0,1] except theta ∈ [-π, π].
- M3 `graph.json`: per-layer compact nodes `{idx, kind, community}` + edges
  `[src, dst, weight-3dp]` (key→query, top-k=8 per destination on head-averaged
  attention). Unrolled residual edges are IMPLICIT — a `residual` flag tells the
  frontend to synthesize `(t,i)→(t+1,i)` identity edges (`num_tokens*(L-1)`),
  never materialized. Louvain seeded via `networkx.louvain_communities` (no
  extra dep). Providers: `ViTTokenGraphProvider` implements the M0 Protocol.
- M3 projections are DATASET-level, NOT per-pack: `build_projection_artifacts`
  writes coords `.bin` (fp16) + JSON sidecars + joblib reducers to their own
  dir. PCA/UMAP support `.transform` (trajectories, upload-into-landscape);
  t-SNE does not (static landscape only). UMAP is optional (`umap-learn`);
  code degrades to PCA+t-SNE via `umap_available()` when absent.

Update this table as milestones land.
