# UMT-ViT — Universal Multi-Scale Topographic Vision Transformer

**Architecture & Implementation Plan · v1.0 · 2026-07-10**

UMT-ViT is a self-supervised representation-learning experiment: a dual-scale
cross-attention Vision Transformer whose encoder layers are **uplifted into a
3-D latent voxel volume** that a differentiable 3-D Self-Organizing Map
continuously reorganizes into a topology-preserving, directly inspectable
manifold — trained without labels, on any image dataset, through
configuration alone.

This document is the design contract for the experiment. It is a **separate
experiment** from ViTreous ([`ARCHITECTURE.md`](./ARCHITECTURE.md)): it
shares the repo, the orchestration model, and the honesty rules, and touches
none of ViTreous's code. Research grounding and the critical evaluation of
the underlying proposal live in
[`UMT-VIT-RESEARCH.md`](./UMT-VIT-RESEARCH.md); orchestration agreement and
milestone status in [`UMT-VIT-DECISION-LOG.md`](./UMT-VIT-DECISION-LOG.md).

---

## 1. Requirements

| Dimension | Decision |
|---|---|
| Purpose | Research experiment: does an explicitly geometric (voxel + SOM) latent space give interpretable, transferable representations? Negative results are reportable results |
| Compute | **Kaggle single GPU** (T4/P100, ~30 h/week). Everything must fit: mixed precision, gradient checkpointing, configurable depth/resolution |
| Supervision | Self-supervised pretraining; labels optional (linear probe / fine-tune only) |
| Universality | **Dataset swap = config only.** No class names, resolutions, directory layouts, or metadata columns hardcoded anywhere in model/training code |
| Datasets (v1) | HAM10000 dermoscopy (primary — DSCATNet's domain, enables direct comparison) + EuroSAT (swap proof; maximal domain contrast) + generated shapes (CI smoke) |
| Deliverables | `experiments/umtvit` Python package + one Kaggle notebook + exported latent-cube/SOM artifacts + evaluation report |
| Repo | New code in `experiments/umtvit/` only. `apps/`, `packages/`, `kaggle/`, `hatchvision/`, `webapp/` untouched |
| Honesty | Z-axis labeled "learned hierarchy", never physical depth; "Hebbian" out of user-facing copy; speculative components (geodesic loss) ablation-gated, off by default |

## 2. High-level dataflow

```
dataset config ─▶ UniversalDataModule ─▶ two augmented views (I_a, I_b)
                                              │
                     ┌────────────────────────┤ (shared weights)
                     ▼                        ▼
            DualScalePatchEmbed        (same for I_b)
              fine Xf · coarse Xc
                     ▼
            CrossScaleAttention  (CLS-bridged; full-pair optional)
                     ▼
            FeatureFusion ─▶ TransformerEncoder (L layers, all layers kept)
                     ▼
            SpatialUplifting: layer l ─▶ slice V[:,:,l,:]
                     ▼
            Latent volume  V ∈ R^{H'×W'×L×C}
                     ▼                          ▼
            Soft3DSOM (BMU, neighborhood)   pooled z + projection head
                     ▼                          ▼
   L_som + L_smooth (+ L_geo, gated)      L_ntxent(z_a, z_b)
   + L_order (layer-scale regularizer)
```

No global average pooling before the latent volume; no classifier during
pretraining.

## 3. Mathematical framework

### 3.1 Dual-scale tokenization
Image `I ∈ R^{H×W×3}` → fine patches `P_f` (size `p_f`, N_f tokens) and
coarse patches `P_c` (size `p_c`, N_c tokens); linear embeddings
`x_i = W p_i + b` per stream, plus per-stream CLS tokens and learned
positional embeddings. Defaults: 128×128 input, `p_f = 8` (256 tokens),
`p_c = 16` (64 tokens), `C = 256`.

### 3.2 Cross-scale attention
Default (CrossViT-style, linear in tokens): each stream's CLS token attends
as query over the *other* stream's patch tokens, then is re-injected into its
own stream. Config-optional full-pair variant (DSCATNet-style):
`Q = X_f W_Q`, `K = X_c W_K`, `V = X_c W_V`,
`Y_f = X_f + softmax(QKᵀ/√d)V` (and symmetrically for the coarse stream).

### 3.3 Feature fusion + encoder
Fused token sequence (fine grid resampled to a common `H'×W'` grid and summed
with upsampled coarse tokens) enters a pre-norm ViT encoder of `L` layers
(default `L = 8`). **Every layer's output is retained.**

### 3.4 Spatial uplifting (the architectural contribution)
For layer `l`, reshape its patch tokens to the `H'×W'` grid, giving
`F_l(x,y)`; a per-layer linear projection produces slice
`V(x, y, z=l) = W_l F_l(x, y)`, stacked into
`V ∈ R^{H'×W'×L×C}` (default `16×16×8×64` after channel projection —
131k voxel-features per image). The Z-axis is transformer depth: a learned
hierarchy of representations, **not** physical depth.

### 3.5 Differentiable 3-D SOM
Neurons `w_k` on a 3-D grid `G` (default `8×8×8`). Per voxel `v_i`:
soft assignment `q_{ik} ∝ exp(−‖v_i − w_k‖²/τ_som)` (DPSOM-style; hard
argmin BMU only for evaluation), neighborhood
`h(k, k*) = exp(−d_G(k,k*)²/2σ²)` with σ annealed exponentially over
training (DESOM schedule). Loss

`L_som = Σ_i Σ_k q_{ik*-detached} · h(k, BMU(i)) · ‖v_i − w_k‖²`

trained by gradient descent on both `w` and the encoder. Config-selectable
classical alternative: `som.update = "kohonen_ema"` applies the literal
Kohonen/Hebbian EMA update to `w` outside the autograd graph (kept for the
biological-variant ablation).

### 3.6 Self-supervised objectives
- **Contrastive:** pooled volume → 2-layer MLP head → `z`; NT-Xent
  `L_ntxent = −log [exp(sim(z_a,z_b)/τ) / Σ_j exp(sim(z_a,z_j)/τ)]`.
- **Smoothness:** total variation over the volume's 3-D neighbor graph `E`:
  `L_smooth = Σ_{(i,j)∈E} ‖v_i − v_j‖²` (normalized per edge).
- **Geodesic (ablation-gated, default weight 0):** mini-batch k-NN graph over
  voxel features, edge weights `‖v_i − v_j‖`; `D_g(a,b)` = Dijkstra shortest
  path with gradients through the (detached-path) edge lengths;
  `L_geo = D_g(z_a, z_b)`. Speculative — see RESEARCH §6 for why it is gated.

### 3.7 Layer-scale ordering regularizer (added inductive bias)
ViT depth does not order itself by spatial scale (Raghu et al. — RESEARCH
§3). To make the Z-axis meaningful we *impose* a bias: per slice `l`, a
spatial-frequency penalty whose cutoff decreases with `l` (slice `l`'s
spatial power spectrum above cutoff `f(l) = f_max · (1 − l/L)` is penalized),
so shallow slices may carry high-frequency detail and deep slices are pushed
toward smooth, coarse structure:
`L_order = Σ_l ‖HighPass_{f(l)}(V[:,:,l,:])‖²`.
Whether genuine scale ordering emerges is measured (U5 probes), not assumed.

### 3.8 Total objective
`L = λ₁ L_ntxent + λ₂ L_som + λ₃ L_smooth + λ₄ L_order + λ₅ L_geo`
Defaults: `λ₁=1.0, λ₂=0.5, λ₃=0.1, λ₄=0.1, λ₅=0.0`. All λ's and schedules
(σ, τ_som, τ) live in the run config; every term individually switchable for
ablations.

## 4. Universal dataset configuration

Declarative config, mirroring the proven repo pattern (ViTreous adapters,
legacy loaders) but self-contained — no dependency on `packages/core`:

```yaml
dataset:
  name: ham10000
  loader: csv            # csv | imagefolder | shapes (auto-detect fallback)
  image_dir: /kaggle/input/ham10000/images
  metadata_csv: /kaggle/input/ham10000/HAM10000_metadata.csv
  label_column: dx        # optional — absent ⇒ fully unlabeled mode
  group_column: lesion_id # optional — leakage-free grouped splits
  image_size: 128
  channels: 3
  splits: {train: 0.8, val: 0.1, test: 0.1, seed: 1}
  augmentation: dermoscopy_default   # named policy from the registry
```

- Loaders: `imagefolder` (train/<class>/*), `csv` (image path + optional
  columns), `shapes` (generated, zero-download CI smoke — reuses the idea of
  `scripts/make_shapes_dataset.py`).
- Augmentation registry: named, composable policies (crop, flip, rotation,
  affine, elastic, color/brightness/contrast jitter, Gaussian noise); each
  policy declares which ops it applies so medical configs can exclude
  hue-destroying transforms. Two-view contrastive sampling wraps any policy.
- Labels, when present, are used **only** by the evaluation suite.
- New dataset = new YAML file. Model and training code never read dataset
  specifics from anywhere else. (HF/torchvision built-ins are a future
  extension, not v1.)

## 5. Package layout

```
experiments/umtvit/
  umtvit/
    config.py        dataclass schema + YAML load/validate (single source of truth)
    data/            loaders (imagefolder, csv, shapes) · augmentation registry
                     · two-view wrapper · grouped splits
    models/          patch_embed.py · cross_attention.py · fusion.py
                     · encoder.py · uplifting.py · som3d.py · heads.py
    losses/          ntxent.py · som.py · smoothness.py · ordering.py
                     · geodesic.py (gated)
    engine/          trainer (AMP, grad-checkpoint, cosine LR, σ/τ schedules,
                     resume) · ablation runner
    eval/            linear_probe.py · knn.py · som_metrics.py (quantization
                     + topographic error) · manifold.py (trustworthiness,
                     continuity) · zaxis_probe.py (per-slice frequency/
                     receptive-field analysis)
    export/          latent cube (fp16 .bin + JSON sidecar) · SOM grid maps
                     · training curves · run report (markdown)
  configs/           ham10000.yaml · eurosat.yaml · shapes.yaml · model/*.yaml
  notebooks/         kaggle_umtvit.ipynb  (config cell at top — the only knob)
  tests/             unit + smoke tests (CPU, shapes dataset)
```

## 6. Evaluation protocol (labels enter here only)

1. **Linear probe / k-NN** on frozen pooled features (val/test) — the
   standard SSL yardsticks; HAM10000 numbers set against DSCATNet's
   supervised 97.8% for context (not parity — different training signal).
2. **SOM quality:** quantization error, topographic error, dead-neuron rate.
3. **Manifold quality:** trustworthiness & continuity between input space
   and latent volume / SOM space.
4. **Z-axis probes:** per-slice spatial-frequency spectra and effective
   receptive fields — the measured answer to "did scale ordering emerge?"
5. **Ablations** (the science): each loss term on/off, SOM gradient vs
   kohonen_ema, CLS-bridged vs full-pair cross-attention, ± uplifting
   (baseline = plain SimCLR on the same backbone), ± pretrained init.

## 7. Computational budget & scalability

Defaults sized for a Kaggle T4 (16 GB): ~15–20 M parameters; volume
`16×16×8×64` fp16 ≈ 260 KB/image; SOM 512 neurons × 64 dims. Batch 128 with
AMP + gradient checkpointing on the encoder. Knobs that scale the same code
down (CI: 32×32 shapes, L=2, volume 4×4×2×16) or up (P100, image 224,
L=12): all in config. Sparse SOM updates (top-k voxels per image) and
capped-size geodesic graphs keep the gated components bounded. SOM soft
assignments computed in chunks to bound the `voxels × neurons` matrix.

## 8. Interpretability artifacts

Every run exports: the latent cube per probe image (scrollable depth slices),
SOM component planes + hit maps + U-matrix (3-D: rendered as per-Z-layer
grids), per-slice frequency probes, and neighborhood-organization metrics
over training time. Formats are self-describing (JSON sidecar + fp16 bin,
same discipline as ViTreous packs). A ViTreous `GraphProvider`/pack adapter
so the cube can be explored in the existing four-view workbench is a
**future extension** (§11) — explicitly out of v1 scope.

## 9. Implementation roadmap (Opus work packages)

Fable 5 orchestrates; **Opus agents implement** each milestone against this
document. Every milestone lands as reviewed commits on
`claude/umt-vit-opus-orchestration-zpd03a` with tests (CPU-runnable; the
shapes dataset is the CI workhorse).

| U | Deliverable | Key acceptance test |
|---|---|---|
| U0 | `experiments/umtvit` scaffold: package, config schema + YAML validation, shapes generator, pytest wiring | config round-trips; invalid configs rejected with clear errors; shapes dataset yields correct tensors |
| U1 | Universal data pipeline: imagefolder + csv loaders, grouped splits, augmentation registry, two-view contrastive wrapper; `ham10000.yaml`, `eurosat.yaml`, `shapes.yaml` | same code loads all three configs; grouped split leaks no `lesion_id` across splits; unlabeled mode works |
| U2 | Backbone: dual-scale embed, CLS-bridged + full-pair cross-attention, fusion, encoder returning all L layer outputs | shape tests at 3 resolutions; param count within budget; both fusion modes forward cleanly |
| U3 | Spatial uplifting → latent volume + Soft3DSOM (+ kohonen_ema variant) + `L_som` | volume shape `H'×W'×L×C` from config; SOM on synthetic 3-cluster data: quantization error falls, topographic error < chance; EMA and gradient variants both converge |
| U4 | Losses (NT-Xent, smoothness, ordering; geodesic gated) + Trainer (AMP, checkpointing, schedules, resume) | 200-step shapes smoke run: total loss decreases, no NaN with AMP, resume is bit-consistent; geodesic off ⇒ zero overhead |
| U5 | Evaluation suite (probe, k-NN, SOM metrics, trustworthiness/continuity, Z-axis probes) + ablation runner | shapes run: linear probe ≫ chance; all metrics emitted to run report; ablation runner produces comparison table |
| U6 | Kaggle notebook end-to-end on HAM10000 + export module (latent cubes, SOM maps, curves, report) | notebook runs top-to-bottom (nbformat-valid, no hardcoded paths outside config cell); artifacts self-describing and re-loadable |
| U7 | Swap proof (EuroSAT via config only) + baseline/ablation matrix + final experiment report in `docs/` | zero code diffs between HAM10000 and EuroSAT runs; report answers the Z-axis question with measurements |

Sequencing: U0→U1→U2→U3→U4 serial (each consumes the previous layer's
interfaces); U5 may start once U3 lands; U6/U7 serial after U4+U5.

## 10. Risk analysis

| Risk | Severity | Mitigation |
|---|---|---|
| Z-axis ordering fails to emerge | Med (it's the research question) | `L_order` inductive bias; per-slice probes; negative result is a documented finding, not a failure |
| SOM collapse / dead neurons | Med | soft assignments, σ annealing, dead-neuron metric in every run report; DESOM-standard schedules |
| Geodesic loss unstable/expensive | High if enabled | gated off by default; capped batch graphs; persistent-homology loss recorded as principled fallback |
| SSL data-hunger on ~10k images | Med | pretrained-init flag (ablatable); DINO-style objective as fallback; probe against supervised reference honestly |
| Volumetric memory blow-up | Med | fp16 volume, channel-projected C=64, chunked SOM assignment, grad checkpointing; budget test in U3 |
| Augmentation mismatch on medical data | Med | per-dataset named policies; dermoscopy policy excludes hue-destroying ops |
| Kaggle GPU quota | Low | ≤ 4 h pretrain at default scale; resumable trainer; CI never needs GPU |
| Scope creep toward ViTreous integration | Low | pack adapter explicitly future work (§8, §11) |

## 11. Future extensions

ViTreous pack/GraphProvider adapter (explore the cube in the four-view
workbench) · persistent-homology topology loss · DINO/negatives-free
objective · HF/torchvision built-in dataset loaders · 3-D medical volumes
(true depth alongside learned depth — kept strictly distinguished) ·
fine-tuning heads for segmentation/retrieval/anomaly detection · SOM-guided
active learning.

## 12. References

DSCATNet, PLOS ONE 2024 · Chen, Fan & Panda, *CrossViT*, ICCV 2021 · Raghu et
al., *Do ViTs See Like CNNs?*, NeurIPS 2021 · Kohonen 1982/1990 · Fortuin et
al., *SOM-VAE*, ICLR 2019 · Forest et al., *DESOM*, 2019/2021 · Manduchi et
al., *DPSOM* · Huijben et al., *SOM-CPC*, ICML 2023 · Chen et al., *SimCLR*,
ICML 2020 · Caron et al., *DINO*, ICCV 2021 · Moor et al., *Topological
Autoencoders*, ICML 2020 · Tenenbaum et al., *Isomap*, Science 2000 ·
Tschandl et al., *HAM10000*, 2018 · Helber et al., *EuroSAT*, 2019.
Annotated list with links and the critical proposal evaluation:
[`UMT-VIT-RESEARCH.md`](./UMT-VIT-RESEARCH.md).
