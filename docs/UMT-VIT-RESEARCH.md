# UMT-ViT — Research Record (Phase 1)

Companion to [`UMT-VIT-ARCHITECTURE.md`](./UMT-VIT-ARCHITECTURE.md). This file
preserves the literature review and the critical evaluation of the
**Universal Multi-Scale Topographic Vision Transformer (UMT-ViT)** research
proposal that the architecture decisions rest on (2026-07-10, planned by
Claude Fable 5 acting as research/orchestration model; implementation
delegated to Claude Opus agents). UMT-ViT is a **separate experiment** from
ViTreous ([`RESEARCH.md`](./RESEARCH.md) / [`ARCHITECTURE.md`](./ARCHITECTURE.md));
they share the repo and the orchestration model, nothing else.

---

## 1. The proposal under evaluation

User-supplied research proposal: extend the Dual-Scale Cross-Attention
Transformer (DSCATNet) beyond supervised classification into a universal,
self-supervised spatial representation learner. Keep the dual-scale
cross-attention front end; **delete** global average pooling + classifier;
instead project every encoder layer into a slice of a 3-D latent voxel volume
(`H' × W' × Z × C`, Z = transformer depth), self-organize that volume with a
Hebbian 3-D SOM, and train with label-free objectives (contrastive
consistency, SOM quantization, geodesic distance, smoothness). Implementation
target: a dataset-agnostic Kaggle notebook where swapping datasets is
configuration only.

A supplied companion analysis contributes the mathematical framework adopted
in the architecture doc (tokenization → cross-attention → uplifting → SOM →
losses, see ARCHITECTURE §3) and one load-bearing caveat evaluated in §3
below: transformer depth does **not** automatically order itself into a
texture → shape hierarchy, so the Z-axis needs an explicit inductive bias to
be interpretable.

## 2. Multi-scale cross-attention ViTs — what is validated

**DSCATNet** ([*Dual scale light weight cross attention transformer for skin
lesion classification*, PLOS ONE 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312598))
— dual-scale patches (8×8 and 16×16), cross-attention between the two token
streams, feature-fusion module; 97.80% accuracy / 0.9584 kappa on HAM10000,
~94–96% on PAD-UFES. Validates the proposal's front end **in the target
domain** (dermoscopy) at lightweight scale.

**CrossViT** ([Chen, Fan & Panda, ICCV 2021, arXiv:2103.14899](https://arxiv.org/abs/2103.14899))
— the stronger, ImageNet-scale evidence for the same idea: dual-branch
small/large-patch encoders with cross-attention token fusion, where using
each branch's CLS token as the cross-attention *query agent* is linear-time
in tokens and outperformed full pairwise cross-attention in their ablation.

**Decision:** adopt the dual-scale cross-attention backbone as proposed; use
CrossViT's CLS-bridged cross-attention as the default fusion (cheaper, better
ablation evidence) with full token-pair cross-attention (DSCATNet-style) as a
config option.

## 3. Does ViT depth form a spatial-scale hierarchy? (the Z-axis question)

[Raghu et al., *Do Vision Transformers See Like Convolutional Neural
Networks?*, NeurIPS 2021](https://arxiv.org/abs/2108.08810) show ViT layer
representations are **relatively uniform across depth** (strong lower↔higher
layer CKA similarity; self-attention aggregates global information from the
earliest layers). This directly undercuts a naive reading of the proposal's
Stage 5, where stacking layers 1..L as voxel depth slices is implied to yield
a texture→structure→shape axis.

**Consequences adopted (matching the companion analysis):**
1. The Z-axis is *documented* as "learned hierarchy of representations", never
   as physical/anatomical depth or an assumed semantic scale axis.
2. An explicit **layer-scale ordering regularizer** is added (ARCHITECTURE
   §3.7) so consecutive depth slices are pushed toward progressively coarser
   effective receptive fields — an inductive bias, not an emergent property.
3. Whether the ordering actually emerges is a **measured experimental
   outcome** (per-slice frequency/receptive-field probes in the evaluation
   suite), reported honestly either way. This is one of the experiment's
   genuine research questions, not a premise.

## 4. Deep self-organizing maps — what is validated

- **SOM-VAE** ([Fortuin et al., ICLR 2019](https://arxiv.org/abs/1806.02199))
  — SOM over a discrete VAE latent; proved joint deep+SOM training works.
- **DESOM** ([Forest et al., ESANN 2019 / Neural Computing & Applications
  2021](https://link.springer.com/article/10.1007/s00521-021-06331-w),
  [code](https://github.com/FlorentF9/DESOM)) — SOM layer in a *continuous*
  latent space, Gaussian neighborhood with exponential radius decay, trained
  jointly with the encoder by gradient descent; no pretraining; consistently
  outperforms SOM-VAE. The SOM quantization term is exactly the proposal's
  `L_SOM = Σ‖v_i − w_BMU(i)‖²` with neighborhood weighting.
- **DPSOM** ([Manduchi et al., arXiv:1910.01590](https://arxiv.org/pdf/1910.01590))
  — probabilistic (soft-assignment) variant; soft assignments avoid the
  non-differentiable argmin BMU.
- **SOM-CPC** ([Huijben et al., ICML 2023, arXiv:2205.15875](https://arxiv.org/pdf/2205.15875))
  — **contrastive learning + SOM jointly**, structured 2-D maps of high-rate
  time series. The closest published relative of the proposal's
  contrastive+SOM combination; validates that the two objectives cooperate.

**Decision:** the "Hebbian SOM" ships as a **differentiable DESOM-style SOM
layer in continuous latent space** (Gaussian neighborhood, annealed σ, soft
BMU assignment à la DPSOM) extended to a 3-D neuron grid. The classical
Kohonen EMA update (the literal Hebbian rule) is kept as a config-selectable
alternative (`som.update: "gradient" | "kohonen_ema"`) so the biological
variant remains testable, but gradient training is the validated default.
"Hebbian" language stays in docs/comments as lineage, out of any UI copy —
same rule as the BDH evaluation (RESEARCH.md §4).

## 5. Self-supervised objectives — what is validated

- **NT-Xent / SimCLR** ([Chen et al., ICML 2020](https://arxiv.org/abs/2002.05709))
  — the proposal's contrastive consistency loss verbatim; augmentation choice
  dominates performance, which the universal-config design must expose
  per-dataset (medical imagery tolerates different augmentations than
  natural images — hue jitter that is harmless on pets can destroy
  dermoscopic class evidence).
- **DINO** (Caron et al., ICCV 2021) — evidence that ViTs train well
  self-supervised at small-ish scale; kept as a fallback objective if
  NT-Xent underperforms on small datasets (negatives-free).
- Small-data caveat: SSL from scratch on ~10k-image datasets (HAM10000 size)
  is data-hungry; the config supports optional ImageNet-pretrained patch
  embeddings / encoder init as a documented, ablatable escape hatch.

**Decision:** NT-Xent primary; projection head on the pooled latent volume;
per-dataset augmentation policy in the dataset config; pretrained-init flag.

## 6. Geodesic & topology-preserving losses — critical assessment

- **Isomap** (Tenenbaum et al., Science 2000) — geodesic-preserving embedding
  is classical, but non-parametric and offline.
- **Topological Autoencoders** ([Moor et al., ICML 2020, arXiv:1906.00722](https://arxiv.org/pdf/1906.00722))
  — differentiable topology preservation via persistent homology on
  *mini-batches*; the accepted way to make "preserve structure" a loss.
- The proposal's `L_geo` (shortest path through a voxel k-NN graph between
  two views' embeddings) is **not established practice**: path search is
  non-differentiable in the path choice, O(V log V) per pair per step, and
  has a known degenerate optimum — the network can shorten graph edges
  globally until geodesic ≈ Euclidean, at which point the term duplicates
  the contrastive loss.

**Decision:** geodesic loss is **ablation-gated, off by default**
(`loss.geodesic.weight: 0.0`), implemented as a mini-batch k-NN-graph
approximation with gradients flowing only through edge lengths on the
(detached) shortest path — honest about its status as the proposal's most
speculative component. The always-on topology terms are the SOM neighborhood
loss + the smoothness (total-variation) regularizer, both validated. A
persistent-homology batch loss (Moor et al.) is recorded as the principled
upgrade path if geodesic ablations show promise.

## 7. Critical evaluation summary of the proposal

**Experimentally supported (per cited literature):** dual-scale
cross-attention improves representations (DSCATNet, CrossViT); joint
deep-network + SOM training self-organizes latents without labels (DESOM,
DPSOM); contrastive + SOM objectives cooperate (SOM-CPC); NT-Xent
augmentation-consistency training works (SimCLR); smoothness/TV
regularization is standard.

**Speculative / to be tested by this experiment:** that stacking encoder
layers as voxel depth yields a *useful* 3-D latent geometry (novel — the
experiment's central claim); that the Z-axis becomes scale-ordered (only
with the added regularizer, measured not assumed — §3); the geodesic loss
(§6); any "biologically inspired / Hebbian" framing beyond loose lineage
(same ruling as for BDH); SSL-from-scratch quality on ~10k-image datasets
(§5 caveat).

**Rejected:** claiming Z = anatomical/physical depth; Hebbian language in
user-facing copy; geodesic loss on the critical path; hard (non-soft) BMU
assignment inside the training graph.

## 8. Method-selection summary

| Component | Chosen | Status |
|---|---|---|
| Backbone | dual-scale patches + CLS-bridged cross-attention (full-pair optional) | validated (CrossViT ICCV 2021; DSCATNet PLOS ONE 2024) |
| Latent geometry | per-layer uplifting → `H'×W'×Z×C` voxel volume | **novel — the experiment** |
| Z-axis semantics | layer-scale ordering regularizer + per-slice probes | research question, measured |
| Self-organization | differentiable 3-D DESOM-style SOM, soft BMU, annealed σ | validated at 2-D (DESOM/DPSOM/SOM-CPC); 3-D extension is ours |
| SSL objective | NT-Xent on pooled volume + projection head | validated (SimCLR) |
| Topology terms | SOM neighborhood + TV smoothness (on) · geodesic (ablation-gated) | validated · speculative |
| Universality | declarative dataset config, swap = config only | proven pattern in this repo (ViTreous §4, legacy loaders) |
| Evaluation | linear probe, k-NN, SOM quantization/topographic error, trustworthiness/continuity, ablations | standard practice |

## 9. Reference list

- Anonymous authors, *Dual scale light weight cross attention transformer for
  skin lesion classification* (DSCATNet), PLOS ONE 2024 —
  journals.plos.org/plosone/article?id=10.1371/journal.pone.0312598
- Chen, Fan, Panda, *CrossViT: Cross-Attention Multi-Scale Vision Transformer
  for Image Classification*, ICCV 2021 — arxiv.org/abs/2103.14899
- Raghu, Unterthiner, Kornblith, Zhang, Dosovitskiy, *Do Vision Transformers
  See Like Convolutional Neural Networks?*, NeurIPS 2021 — arxiv.org/abs/2108.08810
- Kohonen, *Self-Organized Formation of Topologically Correct Feature Maps*,
  Biological Cybernetics 1982; *The Self-Organizing Map*, Proc. IEEE 1990
- Fortuin, Hüser, Locatello, Strathmann, Rätsch, *SOM-VAE*, ICLR 2019 —
  arxiv.org/abs/1806.02199
- Forest, Lebbah, Azzag, Lacaille, *Deep Embedded SOM*, ESANN 2019 / Neural
  Computing & Applications 2021 — github.com/FlorentF9/DESOM
- Manduchi, Hüser, Rätsch, Fortuin, *DPSOM: Deep Probabilistic Clustering
  with Self-Organizing Maps* — arxiv.org/pdf/1910.01590
- Huijben, Nijdam, Overeem, van Gilst, van Sloun, *SOM-CPC*, ICML 2023 —
  arxiv.org/pdf/2205.15875
- Chen, Kornblith, Norouzi, Hinton, *SimCLR*, ICML 2020 — arxiv.org/abs/2002.05709
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*
  (DINO), ICCV 2021
- Moor, Horn, Rieck, Borgwardt, *Topological Autoencoders*, ICML 2020 —
  arxiv.org/pdf/1906.00722
- Tenenbaum, de Silva, Langford, *A Global Geometric Framework for Nonlinear
  Dimensionality Reduction* (Isomap), Science 2000
- Tschandl, Rosendahl, Kittler, *The HAM10000 dataset*, Scientific Data 2018
  · Helber et al., *EuroSAT*, JSTARS 2019

Key citations validated against live sources on 2026-07-10.
