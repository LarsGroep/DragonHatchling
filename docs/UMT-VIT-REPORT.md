# UMT-ViT — Final Experiment Report

**Universal Multi-Scale Topographic Vision Transformer · v1.0 · 2026-07-11**

This is the capstone report for the UMT-ViT experiment (milestone U7). It is
written to stand on its own: a reader who has not followed the build session
should be able to learn what UMT-ViT is, what was built, what the experiment
measured, and what it did and did not establish. Every quantitative claim traces
to a specific source — an owner-executed GPU run recorded in
[`UMT-VIT-NOTEBOOK-FEEDBACK.md`](./UMT-VIT-NOTEBOOK-FEEDBACK.md), the CPU-scale
ablation matrix produced for this milestone, or a test in
[`experiments/umtvit/tests/`](../experiments/umtvit/tests/). Design contract:
[`UMT-VIT-ARCHITECTURE.md`](./UMT-VIT-ARCHITECTURE.md); research grounding and
honesty rules: [`UMT-VIT-RESEARCH.md`](./UMT-VIT-RESEARCH.md); milestone log:
[`UMT-VIT-DECISION-LOG.md`](./UMT-VIT-DECISION-LOG.md).

---

## 1. Executive summary

**What UMT-ViT is.** A self-supervised representation-learning experiment: a
dual-scale cross-attention Vision Transformer whose *every* encoder layer is
uplifted into a slice of a 3-D latent voxel volume (`H'×W'×Z×C`, where Z is
transformer depth), which a differentiable 3-D Self-Organizing Map (DESOM/DPSOM
lineage) continuously reorganizes into a topology-preserving, directly
inspectable manifold — trained without labels, on any image dataset, through
configuration alone. The Z-axis is documented as a *learned hierarchy of
representations*, never physical depth (ARCHITECTURE §3.4, RESEARCH §3).

**What was built.** A self-contained `experiments/umtvit/` Python package
(config → data → models → losses → engine → eval), three Kaggle notebooks, a
Vercel web explorer, and one-string dataset swapping via YAML configs. The test
suite is green at **139 tests** (136 pre-U7 + 3 swap-proof tests added here),
all CPU-only, no downloads.

**Three headline findings** (all from the owner's HAM10000 GPU runs, §5):

1. **Strong label-free representation.** A frozen-feature linear probe reaches
   **0.768 / 0.774 / 0.770** across the three runs (chance 0.143, 7 classes),
   k-NN **0.730 / 0.743 / 0.738** — the SSL objectives learned genuinely
   discriminative dermoscopy features with no labels in training.
2. **SOM collapse was a real failure, then structurally fixed.** Dead-neuron
   fraction traced **0.977 → 0.991 (full collapse) → 0.194 (healthy)** across
   the runs; topographic error **0.008 → 0.973 → 0.084**. The fix (data-driven
   init, dead-neuron revival, grid-derived σ) cost nothing on the probe.
3. **Two honest negatives.** The dual-scale cross-attention bridge, once made
   live, moved the probe by **≈ 0** at this scale (cross-attention value
   unproven); and the Z-axis **did not order by spatial scale** despite the
   ordering regularizer — the experiment's documented negative result.

The EuroSAT swap is proven **structurally** (§7): the same code paths build both
runs and only config values differ; a full EuroSAT-layout dataset runs
end-to-end through the eurosat config route in a test with zero code changes. The
actual EuroSAT Kaggle run is a one-string owner action.

---

## 2. What was built (package U0–U5, notebooks, explorer)

The experiment is delivered as `experiments/umtvit/`, built milestone by
milestone (ARCHITECTURE §9). Pointers:

| Layer | Module(s) | What it provides |
|---|---|---|
| Config (U0) | `umtvit/config.py` | Single source of truth: typed dataclass schema, YAML load/validate/round-trip; every knob for a run lives here. |
| Data (U1) | `umtvit/data/` | `UniversalDataset` (one class for all loaders), `loaders.py` (imagefolder / csv / shapes), grouped leakage-free splits, augmentation registry (`dermoscopy_default`, `satellite_default`, `natural_default`, `none`), two-view contrastive wrapper. |
| Backbone (U2) | `umtvit/models/` | Dual-scale patch embed, cross-attention (`cls_bridged` default + `full_pair`), fusion, `L`-layer encoder returning **all** layer outputs. |
| Latent geometry (U3) | `models/uplifting.py`, `models/som3d.py` | Per-layer spatial uplifting → `H'×W'×Z×C` volume; `Soft3DSOM` (gradient + `kohonen_ema` variants, data-init, revival). |
| Losses + trainer (U4) | `umtvit/losses/`, `engine/trainer.py` | NT-Xent, SOM quantization, TV smoothness (axes-configurable), layer-scale ordering + monotone-centroid penalty, gated geodesic; AMP, grad-checkpoint, cosine LR, σ/τ schedules, bit-exact resume. |
| Evaluation + ablations (U5) | `umtvit/eval/`, `engine/ablation.py` | Linear probe, k-NN, SOM metrics (QE/TE/dead), trustworthiness/continuity, Z-axis spectral-centroid probe, run report; `AblationRunner` over the canonical axes. |

**Notebooks** (`experiments/umtvit/notebooks/`): the canonical
`kaggle_umtvit.ipynb` (config-cell dataset swapping, all visualizations +
animations, evaluation, artifact export — the executable reference and the form
the owner ran on Kaggle), the package-driven `kaggle_umtvit_package.ipynb`, and
`notebooks/README.md`.

**Web explorer** (out-of-band V1): a standalone `/umtvit` route in the deployed
`apps/web` Vercel app — latent-cube Z-scrubber, per-epoch SOM U-matrix replay +
hit maps, embedding-formation replay, training curves, metrics row, and a Z-axis
honesty panel with the monotone-centroid check. The notebook exports a compact
`umtvit_web.json` (v1 contract, ≤ 4 MB) per run; the page ships a shapes demo
fixture and drag-drops real run bundles fully client-side.

**One-string dataset swapping.** A new dataset is a new YAML file
(`configs/*.yaml`); model and training code never read dataset specifics from
anywhere else. `configs/ham10000.yaml` and `configs/eurosat.yaml` differ only in
values — see §7 for the executable proof.

---

## 3. The model at a glance (ARCHITECTURE §3)

- **Input** → two augmented views under a per-dataset augmentation policy.
- **Dual-scale tokenization**: fine (patch 8) + coarse (patch 16) streams, each
  with a CLS token and learned positional embeddings.
- **Cross-scale attention**: CLS-bridged (CrossViT-style) by default, full-pair
  (DSCATNet-style) optional. Each round is `cross-attention → per-stream
  self-attn` so the re-injected CLS reaches the patch tokens before fusion drops
  the CLS (the U2b liveness fix, §5.3).
- **Fusion + encoder**: both streams resampled to a shared `H'×W'` grid, summed,
  projected; a pre-norm ViT of `L = 8` layers, **every** layer output kept.
- **Spatial uplifting**: layer `l` → volume slice `V[:,:,l,:]`, stacked into
  `V ∈ R^{H'×W'×L×C}` (HAM10000: `16×16×8×64`).
- **Two heads off the volume**: pooled readout → 2-layer MLP → contrastive
  embedding `z` (NT-Xent); and the voxel features → 3-D SOM (`8×8×8`).
- **Objective**: `L = λ₁ L_ntxent + λ₂ L_som + λ₃ L_smooth + λ₄ L_order + λ₅
  L_geo`, defaults `λ = (1.0, 0.5, 0.1, 0.1, 0.0)`; geodesic gated off.

Standing model scale (HAM10000 / EuroSAT): dim 256, depth 8, SOM `8×8×8`,
≈ 6.55 M parameters.

---

## 4. Experimental record — overview

The primary dataset is **HAM10000** dermoscopy (DSCATNet's domain). The owner
executed three full GPU runs on Kaggle (T4, ≈ 71–75 min each, 30 epochs, 6.55 M
params), recorded verbatim in
[`UMT-VIT-NOTEBOOK-FEEDBACK.md`](./UMT-VIT-NOTEBOOK-FEEDBACK.md). This report
consolidates them; the raw per-run tuning notes stay in that file.

All probe / k-NN numbers below are **frozen-feature SSL yardsticks** — they
measure how linearly separable the label-free representation is and are **not**
comparable to supervised end-to-end accuracies such as DSCATNet's 97.8% on
HAM10000 (RESEARCH §2, report caveat in `eval/report.py`).

---

## 5. The three HAM10000 GPU runs

### 5.1 Results table (source: NOTEBOOK-FEEDBACK Runs 1–3)

| Metric | Run 1 | Run 2 | Run 3 | Reading |
|---|---|---|---|---|
| Linear probe (chance 0.143) | 0.768 | 0.774 | 0.770 | **stable, strong** |
| k-NN (k=5, cosine) | 0.730 | 0.743 | 0.738 | stable |
| SOM quantization error | 0.243 | 0.205 | 0.233 | healthy |
| SOM topographic error | 0.008 | **0.973** | 0.084 | collapse in Run 2, fixed in Run 3 |
| SOM dead-neuron fraction | 0.977 | **0.991** | **0.194** | collapse → fixed |
| Trustworthiness (k=7) | 0.759 | 0.766 | 0.742 | flat |
| Cross-scale bridge | inert | inert | **live** | see §5.3 |

Config deltas: Run 1 was the first full run (SOM `8×8×8`, σ_end 0.5). Run 2
applied Run-1 SOM guidance (`som_grid [6,6,6]`, wider σ anneal). Run 3 ran the
fully-fixed notebook (SOM structural fixes, Z-free smoothness + monotone
penalty, live cross-attention, GPU/AMP hardening).

### 5.2 The SOM-collapse arc and its structural fix

The dead-neuron fraction tells a three-act story:

- **Run 1 (0.977 dead).** Only ~12 of 512 neurons ever won a BMU assignment: the
  voxels collapsed onto a corner of the SOM. QE/TE looked good (TE 0.008)
  *partly because* so few neurons competed. Flagged as under-utilization.
- **Run 2 (0.991 dead, TE 0.973 — full collapse).** Applying the tuning
  guidance naively (`som_grid [6,6,6]` with a σ_start ≈ 5.7 wider than the whole
  grid radius) pulled every neuron toward the global voxel mean from step one;
  TE ≈ 1.0 from epoch 2. **Diagnosis:** schedule tweaks alone cannot fix this —
  a neighborhood wider than the grid guarantees collapse. Structural fixes
  required.
- **Run 3 (0.194 dead, TE 0.084 — healthy).** Three structural fixes landed:
  **data-driven weight init** (seed each neuron from a real voxel, not Gaussian
  noise), **dead-neuron revival** (re-seed zero-win neurons each epoch), and
  **grid-derived σ** (`σ_start = max(grid)/2`, keeping the initial neighborhood
  inside the lattice). The epoch log shows revival doing its job: dead fraction
  fell 0.99 → 0.46 through training, ~211 neurons re-seeded at epoch 2 tapering
  to 0 by ~epoch 20. The SOM is now a genuinely *used*, topology-preserving map
  (~174/216 live). Crucially, **representation quality held**: probe 0.770 vs
  0.768/0.774 — fixing the SOM cost nothing on the features.

This is the experiment's clearest positive engineering result: a diagnosed
failure mode with a structural, config-flagged remedy that worked.

### 5.3 The cls_bridged inertness discovery

A post-Run-2 audit (U2b) found the CLS-bridged cross-attention was **inert**.
Each round ran `per-stream self-attn → cross-attention`; the cross step writes
only to each stream's CLS token, and fusion then *drops* the CLS. With the
default `cross_rounds: 1`, the bridged information never reached the patch
tokens — the cross-attention parameters received a strictly zero gradient, and
the model degenerated to two independent streams plus sum fusion. **Runs 1–2
therefore trained with no cross-scale exchange** and stand as a valid
*no-cross-attention baseline*. The fix reorders each round to `cross-attention →
per-stream self-attn` (an ordering change, no new parameters), so the self-attn
spreads the updated CLS into the patch tokens before fusion.

**Finding.** Run 3 is the first with the live bridge: probe **0.770** vs the
inert-bridge **0.768 / 0.774** — within noise (Δprobe ≈ 0). At this scale the
dual-scale cross-attention is not (yet) earning its parameters on the probe
metric. This is an honest negative, queued for the full-scale ablation matrix
(no_cross_attention vs cls_bridged vs full_pair); Run 3 supplies the live-bridge
data point.

---

## 6. The Z-axis question, answered with measurements

The architecture's U7 acceptance criterion is that the report *"answers the
Z-axis question with measurements."* The Z-axis question (RESEARCH §3, after
Raghu et al. NeurIPS 2021): **does transformer depth, uplifted into voxel depth,
order itself into a spatial-scale hierarchy — texture-shallow → shape-deep?**
The research record flagged from the outset that it would not emerge for free
and that a negative result is a legitimate finding.

**Measurement (Run 3, the fair per-channel probe).** Per-slice spectral
centroids by depth `z = 0…7`:

```
0.138, 0.143, 0.142, 0.140, 0.135, 0.144, 0.145, 0.145
```

These are **essentially flat** (~0.14 across all 8 depths) — no monotone
decrease, i.e. no scale ordering. The legacy channel-mean measure from Run 3
(`0.138, 0.119, 0.106, 0.113, 0.106, 0.122, 0.133, 0.144`) is non-monotone; Run
1's channel-mean measure (`0.125, 0.194, 0.118, 0.102, 0.126, 0.162, 0.155,
0.107`) spiked at z=1 and wandered. Across every run and every measure, the
sequence never realized the shallow-sharp → deep-smooth hierarchy the Z-axis was
meant to carry.

**Verdict (stated per the honesty framing, RESEARCH §3 / NOTEBOOK-FEEDBACK
Run 3):** *imposing depth-scale ordering on a residual ViT with these
regularizers, at this strength, does not induce monotone scale ordering.* This
is the experiment's documented negative result, not a failure — it was named as
one of the two genuine research questions.

**Why (diagnosed, from NOTEBOOK-FEEDBACK Run 1 issue 2).**
1. The ordering regularizer is *one-sided*: it penalizes power *above* a
   depth-decreasing cutoff (forbids deep slices from being sharp) but never
   *requires* shallow slices to be sharp — low frequency is free at every depth.
2. The smoothness loss was an *antagonist*: TV along the Z-axis actively pulls
   adjacent depth slices together. Run 3 excluded Z from `L_smooth`
   (`smooth_axes: [h, w]`) and added a monotone-centroid penalty (λ 0.05), yet
   the centroids stayed flat — the fix removed the antagonist but the bias at
   this strength still did not create a hierarchy.
3. Residual-stream uniformity (Raghu et al.): ViT representations are inherently
   uniform across depth, working against any imposed depth differentiation.

**Escalation levers — listed and explicitly untested at scale** (from Run 1/3
options, escalating; none has been run to convergence on GPU):
- raise `order_monotone` from 0.05 → 0.2–0.3;
- raise `loss.order` from 0.1 → 0.3–0.5;
- convex cutoff schedule `f(l) = f_max·(1 − l/L)^γ`, γ ≈ 2, so the constraint
  bites earlier in depth;
- a two-sided per-depth **band-pass** target (require shallow slices to carry
  high frequency, not merely permit it).

Whether any lever induces ordering is future work; this report records the
measured negative as it stands.

---

## 7. Swap proof — EuroSAT via config only (ARCHITECTURE §9 U7)

EuroSAT (satellite imagery) is the maximal-domain-contrast swap target. It
cannot be downloaded in this environment, so the swap is proven **structurally
and executably** rather than by a live EuroSAT training run.

**Evidence 1 — the two YAMLs.** `configs/ham10000.yaml` and
`configs/eurosat.yaml` differ only in *values*: dataset name, loader (`csv` vs
`imagefolder`), image size (128 vs 64), augmentation policy (`dermoscopy_default`
vs `satellite_default`), and the derived volume grid (`16×16` vs `8×8`). They
share the standing backbone (dim 256, depth 8, SOM `8×8×8`), the loss weights,
and the whole dataclass schema.

**Evidence 2 — the swap-proof test**
(`experiments/umtvit/tests/test_swap_proof.py`, 3 tests). It:

- loads both shipped YAMLs and builds `UMTViT` + `Soft3DSOM` + `Trainer` for
  **both** from the identical constructors, asserting the objects are of the
  identical classes and that each model's forward output shape follows its own
  config (HAM `volume 16×16×8×64`, EuroSAT `volume 8×8×8×64`; both pooled 512,
  proj 128) — every structural difference traces to a config value;
- builds a synthetic **EuroSAT-layout imagefolder** (`root/<class>/*.jpg`, 3
  classes × 12 images at 64 px), then runs it **end to end** through
  `UniversalDataset` with the `satellite_default` policy → a short `Trainer`
  run → `run_evaluation` → `render_report`, using the `eurosat.yaml` dataset
  block **verbatim** (loader, augmentation, image size); only compute knobs
  (model scale, step budget) are shrunk for CPU. The probe/k-NN run (labeled
  layout) and every SOM metric is finite. This is executable evidence that
  EuroSAT-shaped data + the eurosat config route works with **zero code
  changes**;
- asserts the two configs expose the identical schema (a swap is a value edit,
  never a code/schema change).

**The remaining action is the owner's.** A real EuroSAT Kaggle run is a
one-string change in either notebook (select the `eurosat` preset / point at the
EuroSAT config) and press Run — no code diff. The structural proof above is what
the U7 acceptance criterion ("zero code diffs between HAM10000 and EuroSAT
runs") asks for; the empirical EuroSAT numbers are an owner follow-up.

---

## 8. Ablation matrix (CPU / shapes scale)

**This is smoke-scale evidence — read §9's scale caveat first.** The matrix was
produced with `engine.AblationRunner` over the canonical `ABLATIONS` axes plus a
`baseline` (the default `cls_bridged`, `cross_rounds = 1` recipe), on the
generated **shapes** dataset (3 classes, 120 images), a ~0.1 M-parameter model
(dim 64, depth 2, volume `4×4×2×16`, SOM `3×3×3`), **3 seeds × 250 steps** per
variant. Total wall time 744 s (~12 min) on CPU. Cells are **mean ± std over the
3 seeds** (population std). It is seeded and reproducible, but the test split is
only ~12 images, so the probe/k-NN columns carry large variance and are
directional at best.

| variant | probe_acc | knn_acc | quant_err | topo_err | dead_frac | trust | continuity |
|---|---|---|---|---|---|---|---|
| baseline | 0.606 ± 0.187 | 0.455 ± 0.148 | 0.345 ± 0.012 | 0.421 ± 0.027 | 0.222 ± 0.030 | 0.543 ± 0.025 | 0.505 ± 0.029 |
| no_cross_attention | 0.697 ± 0.113 | 0.636 ± 0.000 | 0.356 ± 0.019 | 0.423 ± 0.186 | 0.198 ± 0.035 | 0.572 ± 0.042 | 0.525 ± 0.025 |
| full_pair | 0.606 ± 0.086 | 0.545 ± 0.129 | 0.337 ± 0.018 | 0.438 ± 0.108 | 0.284 ± 0.046 | 0.580 ± 0.011 | 0.539 ± 0.030 |
| no_som | 0.576 ± 0.043 | 0.424 ± 0.043 | **2.238 ± 0.754** | 0.253 ± 0.073 | 0.432 ± 0.097 | 0.661 ± 0.086 | 0.590 ± 0.089 |
| no_order | 0.545 ± 0.074 | 0.364 ± 0.074 | 0.336 ± 0.021 | 0.465 ± 0.100 | 0.173 ± 0.092 | 0.505 ± 0.043 | 0.455 ± 0.026 |
| smooth_hwz | 0.606 ± 0.113 | 0.303 ± 0.086 | 0.354 ± 0.017 | 0.455 ± 0.096 | 0.210 ± 0.035 | 0.531 ± 0.014 | 0.471 ± 0.024 |
| kohonen_ema | 0.455 ± 0.000 | 0.394 ± 0.113 | 0.312 ± 0.015 | **0.003 ± 0.002** | 0.284 ± 0.017 | 0.525 ± 0.020 | 0.465 ± 0.052 |

lower is better for `quant_err`, `topo_err`, `dead_frac`; higher is better for
the rest. `probe`/`knn` chance ≈ 0.33 (3 classes).

**Interpretation** (directional; consistent with the GPU runs where they
overlap):

- **`no_som` breaks quantization, as designed.** Dropping the SOM loss sends
  quantization error to **2.238** (vs ~0.34 elsewhere) with the highest dead
  fraction — with no term pulling voxels toward neurons, the volume drifts off
  the map. This confirms `L_som` is what makes the SOM a *used* quantizer, the
  same lesson as the HAM10000 SOM-collapse work (§5.2).
- **`kohonen_ema` gives the best topology.** The classical Kohonen/Hebbian EMA
  update reaches topographic error **0.003** (vs ~0.42 for the gradient SOM) and
  the lowest quantization error — exactly what the literal neighborhood rule is
  built to optimize, and consistent with the U5 smoke note. It trades a little
  probe accuracy for map quality; which matters depends on whether the SOM or the
  features are the product.
- **Cross-attention does not help here.** `no_cross_attention` (cross bridge
  removed) *ties or beats* both `baseline` and `full_pair` on the probe
  (0.697 vs 0.606 / 0.606), echoing the U5 smoke note and, more importantly, the
  live-bridge HAM10000 result (§5.3, Δprobe ≈ 0). At small scale the dual-scale
  cross-attention is not earning its parameters. **This must be settled at full
  scale — it is the headline question for the pending GPU ablation matrix.**
- **`no_order` / `smooth_hwz`** (the Z-axis-ordering axes) move the probe within
  noise and do not cleanly improve any manifold metric — congruent with §6's
  finding that the ordering machinery, at these strengths, is not decisively
  shaping the representation.

The probe standard deviations (±0.09–0.19) are as large as the between-variant
gaps, so **no ranking here is conclusive**. The value of this matrix is
methodological (the axes run, seeded and reproducible) and directional (it points
the same way as the GPU runs on the two open questions). The authoritative matrix
is the full-scale GPU run — see §9.

---

## 9. Limitations and future work

**Scale.** The ablation matrix (§8) is **CPU / shapes scale**: a 3-class
synthetic dataset, a ~0.1 M-parameter model, hundreds of steps. It is systematic
and seeded but small; treat it as a *smoke-scale directional signal*, not a
verdict. The full-scale GPU ablation matrix (one notebook run with
`RUN_ABLATIONS=True` at the HAM10000 preset) is a pending owner follow-up and is
the authority that settles the cross-attention and SOM-variant questions.

**Two open findings unresolved at scale.**
- *Cross-attention value* (§5.3): live-bridge Run 3 moved the probe by ≈ 0. Not
  yet earning its parameters at this scale; needs the full-scale
  no_cross / cls_bridged / full_pair comparison.
- *Z-axis ordering* (§6): the documented negative result. The escalation levers
  are listed and explicitly untested at scale.

**Geodesic loss.** Ablation-gated and weighted 0 by default (RESEARCH §6): a
non-standard loss with a known degenerate optimum. It has **never been trained
at scale**; the trainer verifies only that gating it off is zero-overhead. A
persistent-homology batch loss (Moor et al.) is recorded as the principled
upgrade path if geodesic ablations ever show promise.

**Other future extensions** (ARCHITECTURE §11): a ViTreous pack/GraphProvider
adapter to explore the latent cube in the four-view workbench;
DINO/negatives-free objective if NT-Xent underperforms on ~10k images;
HF/torchvision built-in loaders; 3-D medical volumes (true depth alongside
learned depth, kept strictly distinguished); fine-tuning heads for
segmentation / retrieval / anomaly detection; SOM-guided active learning.

**Small-data SSL caveat** (RESEARCH §5). HAM10000 is ~10k images; SSL from
scratch is data-hungry. The probe at ~0.77 is a genuinely strong label-free
signal, but the pretrained-init escape hatch remains an untested ablation.

---

## 10. Provenance of every quantitative claim

- HAM10000 probe/k-NN/SOM/trustworthiness/centroid numbers → NOTEBOOK-FEEDBACK
  Runs 1–3 (owner GPU runs).
- SOM-collapse arc (0.977 → 0.991 → 0.194) and structural fix → NOTEBOOK-FEEDBACK
  Run 2 diagnosis + Run 3 "what the fixes achieved".
- cls_bridged inertness → NOTEBOOK-FEEDBACK Run 2 post-run fix (U2b) + Run 3
  finding 1.
- Z-axis centroids and verdict → NOTEBOOK-FEEDBACK Run 3 finding 2 + Run 1
  issue 2.
- Ablation matrix (§8) → the CPU-scale run produced for this milestone
  (seeded, reproducible via `engine.AblationRunner`).
- Swap proof (§7) → `experiments/umtvit/tests/test_swap_proof.py` + the two
  shipped YAMLs.
- Test count → `python3 -m pytest tests -q` in `experiments/umtvit`.
