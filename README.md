# DragonHatchling · hatchvision

A reusable PyTorch image-classification framework with pluggable encoder
backbones — including an experimental **Baby Dragon Hatchling (BDH)**
backbone — an optional **Hebbian feature memory** for explainability, and a
pipeline that turns what a network learned into an interactive concept graph.

```
dataset loader ──▶ backbone ──▶ classifier head ──▶ training
   (swap me)        (swap me)                          │
                        │ forward hooks (observe only) │
                        ▼                              ▼
              Hebbian feature memory            Grad-CAM · SHAP
                        │
          concept clustering (units that fire together)
                        │
              IVGraph JSON ──▶ webapp/ (Vercel viewer)
```

## Quick start

```bash
pip install -r requirements.txt          # + `pip install shap` for SHAP

# train, record Hebbian statistics, export the concept graph
python scripts/train.py --dataset cifar10 --backbone simple_cnn \
    --epochs 3 --hebbian --export-graph webapp/sample-graph.json

# explore the graph
cd webapp && python3 -m http.server 8000   # or: npx vercel deploy
```

Or walk the whole pipeline interactively in
[`notebooks/explainability_demo.ipynb`](notebooks/explainability_demo.ipynb):
training → Grad-CAM → SHAP → Hebbian concept clusters → exemplar images →
IVGraph export.

## Swapping datasets

**Changing datasets means changing the dataset loader — nothing else.**
Models, transforms, heads, Hebbian tracking, Grad-CAM, and the export all
derive from the loader's `DatasetSpec` (classes, image size, channels,
normalization).

```python
data = build_loader("cifar10")                              # built-in
data = build_loader("fashion_mnist")                        # grayscale? handled
data = build_loader("imagefolder", root="/my/data")         # any train/val tree
```

Built-ins: `cifar10`, `cifar100`, `fashion_mnist`, and `imagefolder` (any
directory of `train/<class>/*`, `val/<class>/*`). For anything else, subclass
`DatasetLoader` (two methods + a spec) and register it:

```python
@register_loader("galaxies")
class GalaxyLoader(DatasetLoader):
    ...
```

## Swapping backbones

Backbones implement one small interface — `forward(x) -> [B, feature_dim]`,
plus `cam_layer()` (for Grad-CAM) and `hebbian_layers()` (what the Hebbian
memory observes) — so encoders are fully interchangeable:

```python
model = create_model("simple_cnn", data.spec)   # fast demo CNN
model = create_model("resnet18",  data.spec)    # torchvision, CIFAR-aware stem
model = create_model("bdh",       data.spec)    # experimental BDH encoder
```

New encoders register with `@register_backbone("name")`.

### The Baby Dragon Hatchling backbone (experimental)

`hatchvision/models/backbones/bdh.py` adapts the BDH architecture
(["The Dragon Hatchling"](https://arxiv.org/abs/2509.26507)) to images:
patch tokens are lifted into a high-dimensional **sparse, positive neuron
space** (ReLU), mixed with **linear attention** (positive query/key kernels),
through an optionally **weight-shared universal layer**. Those sparse,
positive "neurons" are exactly what Hebbian co-activation analysis wants,
which makes this backbone the most interpretable one in the registry. If the
official `bdh` package is installed it is detected
(`OFFICIAL_BDH_AVAILABLE`); the vision adaptation here is self-contained.

## Hebbian feature memory

`HebbianFeatureMemory` hooks the backbone's advertised layers and maintains
an EMA of neuron co-activation ("fire together, wire together") plus
class-conditional firing rates. It is **pure observation**: everything is
detached, so training with or without it is bit-identical — enforced by
`tests/test_framework.py::test_hebbian_memory_does_not_affect_training`.

```python
memory = HebbianFeatureMemory(model, num_classes=data.spec.num_classes)
Trainer(model, TrainConfig(epochs=3), hebbian_memory=memory).fit(train, val)

memory.correlation("stage3")     # co-activation matrix
memory.class_affinity("stage3")  # which classes each unit fires for
```

## Explainability

- **Grad-CAM** (`hatchvision.explain.GradCAM`) — pixel-level saliency for any
  backbone via its `cam_layer()`. No dependencies.
- **SHAP** (`hatchvision.explain.ShapExplainer`) — expected-gradients pixel
  attribution; optional `shap` extra.
- **Concepts** (`cluster_concepts`, `find_exemplars`) — clusters the Hebbian
  co-activation matrix into concepts, labels them by class affinity, and
  attaches the probe images that activate them most.

## IVGraph export & web viewer

`export_ivgraph(...)` writes the unit/concept/class graph as IVGraph JSON
(schema documented in `hatchvision/export/ivgraph.py`). The static app in
[`webapp/`](webapp/README.md) renders it — force layout, cluster
highlighting, tooltips, edge filters, light/dark — and deploys to Vercel
with `npx vercel deploy` (no build step).

## Layout

```
hatchvision/
  data/        DatasetSpec, DatasetLoader interface, built-in loaders
  models/      backbone registry (simple_cnn, resnet*, bdh) + classifier head
  hebbian/     HebbianFeatureMemory (observation-only forward hooks)
  explain/     GradCAM, ShapExplainer, concept clustering & exemplars
  export/      IVGraph JSON builder
  engine/      Trainer / TrainConfig
scripts/       train.py CLI (train → evaluate → export graph)
notebooks/     explainability_demo.ipynb (end-to-end walkthrough)
webapp/        static IVGraph viewer (Vercel-ready)
tests/         smoke tests for every component
```

## Tests

```bash
python -m pytest tests/ -q
```
