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
| M2–M9 | — | Pending. Schema freezes at end of M2 (`pack.schema.json`/`manifest.py`/`pack.ts` co-edited, round-trip enforced; M1 added optional `ModelInfo.patch_size`). |

Technical notes carried forward:
- Attention is captured by *recomputing* softmax from each block's own qkv
  while the fused forward runs untouched — this is what keeps logits
  bit-identical with hooks attached. Keep recompute; do not un-fuse.
- `Trace` = detached CPU torch tensors: attention `[12,6,197,197]` (true
  softmax rows — safe for per-row uint8 quantization), tokens `[13,197,384]`
  (block inputs t=0..11 + final norm t=12), logits, timings.
- M2's Chefer relevance needs a grad-enabled capture variant (drop
  `no_grad`, don't detach on that path); hook structure already supports it.

Update this table as milestones land.
