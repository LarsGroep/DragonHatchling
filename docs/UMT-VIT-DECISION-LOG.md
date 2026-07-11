# UMT-ViT — Decision Log & Orchestration Plan

Record of the planning session (2026-07-10) and the working agreement for how
the UMT-ViT experiment gets built. Companion to
[`UMT-VIT-ARCHITECTURE.md`](./UMT-VIT-ARCHITECTURE.md) and
[`UMT-VIT-RESEARCH.md`](./UMT-VIT-RESEARCH.md). This experiment is separate
from ViTreous ([`DECISION-LOG.md`](./DECISION-LOG.md)) and follows the same
orchestration model.

## Working agreement (set by the repo owner)

- **Claude Fable 5** acts as researcher, architect, and orchestrator only.
- **Claude Opus** agents write all implementation code, one roadmap
  milestone (U0–U7, UMT-VIT-ARCHITECTURE.md §9) per work order, against the
  architecture doc as binding contract.
- Plan precedes code: no milestone starts before its design section exists.
- All work lands on branch `claude/umt-vit-opus-orchestration-zpd03a`;
  the orchestrator reviews agent commits before pushing.
- ViTreous code (`apps/`, `packages/`, `kaggle/`, `supabase/`) and legacy
  code (`hatchvision/`, `webapp/`, `notebooks/`, `scripts/`, `tests/`) are
  read-only reference for this experiment — read, never modified. All new
  code goes in `experiments/umtvit/`.

## Provenance of the design

Two user-supplied inputs, both preserved in the research record:

1. **The UMT-ViT proposal** — DSCATNet-inspired dual-scale cross-attention
   front end; spatial uplifting of all encoder layers into an
   `H'×W'×Z×C` voxel volume; Hebbian 3-D SOM; self-supervised objectives
   (contrastive, geodesic, SOM quantization, smoothness); universal
   config-driven Kaggle notebook.
2. **The companion mathematical analysis** — the formal framework adopted in
   ARCHITECTURE §3 (tokenization, cross-attention, uplifting, SOM update,
   NT-Xent, geodesic graph loss, TV smoothness, combined objective) plus the
   load-bearing caveat that transformer depth does not order itself by
   spatial scale, resolved by the layer-scale ordering regularizer
   (ARCHITECTURE §3.7) and the Z-axis probes (§6.4).

## Load-bearing decisions (orchestrator, from Phase-1 research)

| Question | Decision | Basis |
|---|---|---|
| Cross-scale fusion | CLS-bridged (CrossViT) default; full-pair (DSCATNet) config option | CrossViT ablation: cheaper and stronger at scale (RESEARCH §2) |
| Is the SOM literally Hebbian? | Differentiable soft-SOM (DESOM/DPSOM) default; Kohonen-EMA as ablatable variant; no Hebbian claims in user-facing copy | DESOM outperforms discrete/EMA variants; same honesty ruling as BDH (RESEARCH §4) |
| Geodesic loss | Ablation-gated, weight 0 by default | Non-standard, degenerate optimum, expensive (RESEARCH §6) |
| Z-axis semantics | Imposed via `L_order` regularizer; emergence measured, never assumed | Raghu et al. NeurIPS 2021 (RESEARCH §3) |
| Primary dataset | HAM10000 (+ EuroSAT swap proof, shapes CI) | DSCATNet's domain → direct context; repo has prior ISIC/EuroSAT experience |
| Code location | Self-contained `experiments/umtvit/`, own dataset config | Keeps ViTreous frozen; universality requirement forbids leaning on `packages/core` |
| v1 scope | Pretraining + evaluation + ablations + swap proof; no ViTreous UI integration | Integration deferred until the representation is shown to be worth visualizing |

## Standing defaults (set by orchestrator, overridable)

- Model: image 128 · fine patch 8 / coarse 16 · dim 256 · L=8 encoder layers
  · volume 16×16×8×64 (fp16) · SOM grid 8×8×8.
- Losses: λ = (1.0 NT-Xent, 0.5 SOM, 0.1 smooth, 0.1 order, 0.0 geodesic);
  τ=0.2 (NT-Xent); σ annealed exponentially (DESOM schedule).
- Training: AdamW, cosine LR + warmup, batch 128, AMP (bf16 where available),
  gradient checkpointing on the encoder, resumable checkpoints.
- Testing: every module CPU-testable; shapes dataset is the CI workhorse;
  no test may require a GPU or a download.
- Reporting: every run emits the §6 metric set + §8 artifacts; ablation
  tables generated, not hand-assembled.

## Open questions for the owner (non-blocking, defaults apply)

1. Pretrained patch-embed/encoder init: allowed in the headline run, or
   ablation-only? (Default: from-scratch headline, pretrained as ablation.)
2. HAM10000 Kaggle source dataset to pin in the notebook config.
3. Any appetite for the DINO-style negatives-free fallback in v1 if NT-Xent
   underperforms on 10k images? (Default: fallback lands only if U6 metrics
   demand it.)

## Milestone status

| U | Scope (see UMT-VIT-ARCHITECTURE.md §9) | Status |
|---|---|---|
| U0 | Package scaffold, config schema, shapes generator, pytest wiring | **Complete** (Opus agent, 2026-07-10; 33 tests pass; reviewed & pushed) |
| U1 | Universal data pipeline + 3 dataset configs + augmentation registry | **Complete** (Opus agent, 2026-07-10; 62 tests pass incl. grouped-split leakage; image_size unified to dataset.image_size; reviewed & pushed) |
| U2 | Dual-scale backbone (embed, cross-attention ×2 modes, fusion, encoder) | **Complete** (Opus agent, 2026-07-11; 82 tests; + **U2b fix**: cls_bridged was inert at cross_rounds=1 — reordered cross→self-attn, liveness+gradient proofs, notebook+package both fixed; runs 1–2 stand as the no-cross-attention baseline) |
| U3 | Spatial uplifting + Soft3DSOM (+ EMA variant) + L_som | **Complete** (Opus agent, 2026-07-11; 104 tests; both SOM modes converge on synthetic blobs, TE 0.013 vs 0.735 random; data-init dead 0.89→0.0) |
| U4 | Loss suite + trainer (AMP, checkpointing, schedules, resume) | **Complete** (Opus agent, 2026-07-11; 123 tests; resume bit-exact incl. optimizer state; geodesic gate verified) |
| U5 | Evaluation suite + ablation runner | Pending |
| U6 | Kaggle notebook (HAM10000) + artifact export | Pending |
| U7 | EuroSAT swap proof + ablation matrix + final experiment report | Pending |

Update this table as milestones land.

Out-of-band delivery **V1** (owner request, 2026-07-11): the **UMT-ViT
Explorer** — a standalone `/umtvit` route in the deployed `apps/web` Vercel
app (one nav link added; ViTreous views untouched). The notebook exports a
compact `umtvit_web.json` (v1 contract, ≤4 MB) per run; the page ships a
shapes demo fixture and accepts drag-drop of real run bundles, fully
client-side. Panels: latent-cube Z-scrubber, per-epoch SOM U-matrix replay +
hit maps, embedding-formation replay, training curves, metrics row, Z-axis
honesty panel with the monotone-centroid check. Verified: tsc clean,
110 vitest pass (15 new), next build clean, notebook re-executed end-to-end.

First owner GPU run (HAM10000, 2026-07-11): linear probe **0.768** / k-NN
**0.730** (chance 0.143), QE 0.243, TE **0.008**, trustworthiness 0.759 —
strong label-free signal, near-perfect SOM topology preservation. Flag:
**dead-neuron fraction 0.977** (SOM underused). Tuning guidance for the next
run, in priority order: slower neighborhood anneal (`sigma_end: 1.0`),
smaller SOM grid (`[6,6,6]`), or raise the `som` loss weight; revisit at U5
ablations.

Out-of-band delivery (owner request, 2026-07-10): the complete experiment
also ships as a **self-contained universal notebook** —
`experiments/umtvit/notebooks/kaggle_umtvit.ipynb` (config-cell dataset
swapping via presets, all visualisations + three animations, evaluation,
artifact export; verified clean 33-cell CPU execution). It previews U6 and
serves as the executable reference for U1–U5; the package modules remain the
production form. Schema deltas between the notebook's CONFIG dict and
`umtvit/config.py` are expected (the architecture doc is binding); noted for
U1 hardening: unify `image_size` (derive `model.image_size` from `dataset`),
and fold the notebook's `path_column`/`path_suffix`/`n_per_class` dataset
fields plus loss-schedule knobs (`sigma_start/end`, `order_fmax`) into the
schema as those milestones land.
