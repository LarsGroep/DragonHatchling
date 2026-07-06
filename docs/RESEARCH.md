# ViTreous — Research Record (Phase 1)

Companion to [`ARCHITECTURE.md`](./ARCHITECTURE.md). This file preserves the
full literature review and the critical BDH evaluation that the architecture
decisions rest on, so the reasoning survives beyond the design session
(2026-07-06, planned by Claude Fable 5 acting as research/orchestration
model; implementation delegated to Claude Opus agents).

---

## 1. ViT interpretability & attribution — what is validated

**Attention-based methods.** Attention Rollout and Attention Flow
([Abnar & Zuidema, ACL 2020](https://arxiv.org/abs/2005.00928)) aggregate
attention across layers via matrix products / max-flow. Cheap and intuitive,
but attention weights alone are repeatedly shown to be unfaithful as
explanations ([Jain & Wallace, *Attention is not Explanation*, NAACL
2019](https://arxiv.org/abs/1902.10186); Serrano & Smith 2019).
**Decision:** ship rollout as one lens among several, never the sole story.

**Gradient×attention relevance.** [Chefer, Gur & Wolf, *Transformer
Interpretability Beyond Attention Visualization*, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)
(and the generic-attention follow-up, ICCV 2021;
[official code](https://github.com/hila-chefer/Transformer-Explainability))
propagates relevance (Deep-Taylor-based, class-specific) through attention
layers and consistently beats raw rollout on perturbation benchmarks.
**Decision:** primary token-attribution method.

**CAM variants.** Grad-CAM (Selvaraju et al., ICCV 2017) transfers to ViTs
by treating the last block's token grid as a feature map. Coarse but
class-discriminative and familiar. **Decision:** baseline lens.

**Path integration.** Integrated Gradients (Sundararajan et al., ICML 2017):
axiomatic, model-agnostic, ~20–50 passes per image. **Decision:** included,
computed per-image (precompute or on-demand), never per-hover.

**Faithfulness evaluation.** Deletion/insertion curves, pointing game, and
sanity checks (Adebayo et al., NeurIPS 2018) are the accepted way to score
attribution. **Decision:** the workbench shows multiple lenses side-by-side
*plus their disagreement* (Spearman method-agreement matrix) — disagreement
is treated as signal, not noise.

**Concept / circuit level.** Sparse autoencoders and transcoders (Anthropic's
*Towards/Scaling Monosemanticity* line, 2023–24; applied to ViTs/CLIP in
2024–25 work) recover far more monosemantic features than raw neurons;
CRAFT/ACE-style concept extraction is the non-SAE alternative.
**Decision:** k-sparse SAE over token activations is the v1 concept tier,
with a k-means fallback behind a quality gate; graph-node abstraction admits
concept nodes natively.

## 2. Visual analytics prior art

- [AttentionViz (Yeh et al., IEEE VIS 2023)](https://arxiv.org/abs/2305.03210) —
  joint query–key embeddings across many inputs; validates embedding views
  of attention; found ViT heads grouping patches by hue/brightness.
- exBERT, BertViz, Dodrio, VL-InterpreT — token-level attention inspection;
  demonstrate both the value and the clutter risk of linked views.
- CNN Explainer, GAN Lab, Diffusion Explainer (Polo Chau's group) — the gold
  standard for animated in-browser model pedagogy. Operational lesson
  adopted wholesale: **precompute per-example explanation artifacts, render
  client-side**; live servers per interaction do not scale on small budgets.
- Summit (Hohman et al., VIS 2019), NeuroCartography (Park et al., VIS 2021)
  — attribution/embedding graphs over CNN features: the closest published
  relatives of the Interaction Graph view; both rely on aggregation and
  community structure rather than raw neurons.
- Coordinated-multiple-views with brushing-and-linking is standard vis
  practice; the four-space design is well grounded. The known failure mode
  is interaction latency, solved by the pack format + O(1) entity resolver.

## 3. Gaussian splats as feature primitives

- [Kerbl et al., *3D Gaussian Splatting*, SIGGRAPH 2023] spawned 2D and
  semantic descendants:
  [GaussianImage, ECCV 2024](https://github.com/Xinjie-Q/GaussianImage)
  (2D Gaussians as an image codec: 8 params/Gaussian, ~2000 FPS decode) and
  LangSplat / Feature-3DGS (language/feature fields attached to Gaussians).
- No published work uses Gaussian feature fields as an *interpretability
  bridge* between pixel space and token space — the flagship view is novel
  as a visualization affordance.
- Honestly assessed: it is a rendering/UX layer, **not** an attribution
  method. Risks: (a) per-image splat fitting costs GPU time → v1 derives
  Gaussian parameters deterministically from patch geometry + measured model
  quantities (fitting is a future extension); (b) it must not invent
  structure the model doesn't use → the honesty rule in ARCHITECTURE.md §7
  and an explicit visual-encoding legend in the UI.

## 4. BDH — critical evaluation (user-supplied sources)

Sources reviewed:

1. Paper: [arXiv:2509.26507, *The Dragon Hatchling*, Kosowski et al. /
   Pathway](https://arxiv.org/abs/2509.26507)
   ([HF papers page](https://huggingface.co/papers/2509.26507)).
2. Official code: [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh) —
   `bdh.py` + `train.py`; README states the public code is the baseline
   variant from the paper, and that the headline Sudoku-Extreme 97.4% result
   comes from Pathway's *internal* implementation, not this repo.
3. Community visualization work:
   [r/MachineLearning — "Visualizing emergent structure in the Dragon
   Hatchling"](https://www.reddit.com/r/MachineLearning/comments/1perpzl/p_visualizing_emergent_structure_in_the_dragon/)
   (Reddit blocks automated fetching; noted as evidence of community
   interest in graph-visualizing BDH internals — the same instinct behind
   this repo's legacy `hatchvision` prototype).
4. This repo's own adaptation:
   `hatchvision/models/backbones/bdh.py` (vision adaptation of BDH-GPU:
   patch tokens → high-dimensional ReLU-positive neuron lift → linear
   attention with positive kernels → optional weight-shared universal
   layer).

**Experimentally supported (at LM scale, per paper + repo):** BDH-GPU trains
and roughly matches GPT-2-class Transformers at 10M–1B params on language/
translation; activations are genuinely sparse (~5% nonzero) and positive;
linear attention with ReLU kernels works; trained models exhibit modular,
heavy-tailed ("scale-free") interaction structure.

**Speculative / not established:** the neuroscience framing ("missing link
between the Transformer and models of the brain"); monosemanticity of
individual synapses/neurons as a general property; any causal reading of
Hebbian co-activation statistics; transfer of the interpretability claims to
vision (this repo's adaptation is experimental and makes no accuracy
claims — see its own module docstring).

**Implementation lessons from the abandoned prototype in this repo:**
the Hebbian concept graph became the only view; node identity was hardcoded
to Hebbian units; exports were static with no temporal dimension; force
layouts broke past a few hundred nodes.

**Adopted into ViTreous** (technically justified):
- observation-only forward hooks with a bit-identity training test,
- sparse-positive unit spaces as good graph-node material *when a model
  provides them* (future `BDHUnitGraphProvider`),
- precomputed explanation bundles consumed by a thin frontend.

**Rejected:**
- neurons/units as the core node abstraction (replaced by the
  `GraphProvider` interface),
- Hebbian co-activation as default edge semantics (correlational; the
  validated default is top-k attention edges),
- neuroscience claims in UI copy.

## 5. Method-selection summary

| Layer | Chosen | Status |
|---|---|---|
| Token attribution | Chefer relevance (primary), rollout / Grad-CAM / IG (lenses) | validated literature |
| Faithfulness | deletion+insertion AUC, method agreement | validated literature |
| Concepts | k-sparse SAE (fallback: k-means), exemplar-grounded | validated at LM scale; ViT application is active research → quality gate |
| Graph semantics | top-k attention edges over tokens; Louvain communities | attention edges validated as *description of computation*; presented as such |
| Gaussian field | deterministic derivation from patch geometry + measured quantities | novel viz affordance; labeled a lens |
| Projections | UMAP (default) / PCA / t-SNE, persisted reducers | standard practice |

## 6. Full reference list

- Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020 — arxiv.org/abs/2005.00928
- Chefer, Gur, Wolf, *Transformer Interpretability Beyond Attention Visualization*, CVPR 2021 — openaccess.thecvf.com (pp. 782–791); code: github.com/hila-chefer/Transformer-Explainability
- Chefer, Gur, Wolf, *Generic Attention-model Explainability*, ICCV 2021
- Sundararajan, Taly, Yan, *Axiomatic Attribution for Deep Networks* (IG), ICML 2017
- Selvaraju et al., *Grad-CAM*, ICCV 2017
- Jain & Wallace, *Attention is not Explanation*, NAACL 2019
- Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018
- Yeh, Chen, Wu, Chen, Viégas, Wattenberg, *AttentionViz*, IEEE TVCG 30, 2024 (VIS 2023) — arxiv.org/abs/2305.03210, attentionviz.com
- Hohman et al., *Summit*, IEEE VIS 2019 · Park et al., *NeuroCartography*, IEEE VIS 2021
- Wang et al., *CNN Explainer*, IEEE VIS 2020
- Kerbl et al., *3D Gaussian Splatting*, SIGGRAPH 2023
- Zhang et al., *GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting*, ECCV 2024 — github.com/Xinjie-Q/GaussianImage
- Qin et al., *LangSplat*, CVPR 2024
- Bricken et al. / Templeton et al., *Towards / Scaling Monosemanticity*, Anthropic 2023–24
- Kosowski et al., *The Dragon Hatchling*, arXiv:2509.26507 — github.com/pathwaycom/bdh, huggingface.co/papers/2509.26507
- Community BDH visualization thread: reddit.com/r/MachineLearning/comments/1perpzl
- Helber et al., *EuroSAT*, JSTARS 2019 · Parkhi et al., *Oxford-IIIT Pet*, CVPR 2012
