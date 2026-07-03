# DragonHatchling · hatchvision

A **universal image-classification + explainability tool** built on the Baby
Dragon Hatchling (BDH) architecture: train a classifier on any image dataset,
record what its neurons do with a **Hebbian feature memory**, translate the
activation patterns into **named visual concepts**, and explore the result —
including live in-browser inference — in an interactive concept graph.

```
dataset loader ──▶ backbone ──▶ classifier head ──▶ training
   (swap me)        (swap me)                          │
                        │ forward hooks (observe only) │
                        ▼                              ▼
              Hebbian feature memory            Grad-CAM · SHAP
                        │
          concept clustering (units that fire together)
                        │
        attribute grounding ("wing color: yellow", when the
                        │       dataset has attribute annotations)
                        ▼
      graph.json + model.onnx + manifest.json  ──▶  webapp/
                                     (upload an image, watch the graph fire)
```

## The universal workflow

The same command works for every registered dataset — **swapping datasets
means swapping the loader, nothing else**. Models, transforms, Hebbian
tracking, concept naming, and the export all derive from the loader's
`DatasetSpec`.

```bash
pip install -r requirements.txt onnx    # onnx only needed for --export-bundle

# a fast, self-contained demo (no downloads):
python scripts/make_shapes_dataset.py --out data/shapes
python scripts/train.py --dataset imagefolder --root data/shapes --image-size 64 \
    --backbone simple_cnn --epochs 4 --export-bundle webapp

# the flagship experiment — CUB-200 birds (GPU recommended, see below):
python scripts/train.py --dataset cub200 --root data/cub --backbone hybrid \
    --epochs 12 --export-bundle exports/cub200

# explore the exported graph + run inference in the browser
cd webapp && python3 -m http.server 8000
```

`--export-bundle DIR` writes the three files the web app consumes:
`graph.json` (Hebbian concept graph), `model.onnx` (classifier **plus neuron
activation outputs**), and `manifest.json` (preprocessing + node mapping).

### CUB-200 on Kaggle (GPU)

[`notebooks/kaggle_cub200.ipynb`](notebooks/kaggle_cub200.ipynb) is a
self-contained Kaggle notebook: attach a CUB-200-2011 dataset (e.g.
`wenewone/cub2002011`), enable GPU, *Run All*. It trains the hybrid
BDH model, grounds every Hebbian concept in CUB's 312 attribute annotations
("wing color: yellow · bill shape: hooked"), and produces `bundle.zip` —
unzip it into `webapp/` and redeploy to get a live bird classifier whose
Hebbian graph lights up per image.

## Datasets

Built-ins: `cifar10`, `cifar100`, `fashion_mnist`, `cub200` (with attribute
annotations), and `imagefolder` (any `train/<class>/*`, `val/<class>/*`
tree). For anything else, subclass `DatasetLoader` (two methods + a spec)
and register it:

```python
@register_loader("galaxies")
class GalaxyLoader(DatasetLoader):
    ...
```

A loader can optionally expose per-image **attribute annotations**
(`attribute_names()` / `val_attribute_matrix()`); if it does, concept
grounding switches on automatically — that's the only thing that
distinguishes `cub200` from any other dataset.

## Backbones

Backbones implement one small interface — `forward(x) -> [B, feature_dim]`,
`cam_layer()` (Grad-CAM), `hebbian_layers()` (what the Hebbian memory
observes) — so encoders are fully interchangeable:

```python
model = create_model("simple_cnn", data.spec)   # fast demo CNN
model = create_model("resnet18",  data.spec)    # torchvision, CIFAR-aware stem
model = create_model("bdh",       data.spec)    # pure BDH, from scratch
model = create_model("hybrid",    data.spec)    # frozen pretrained encoder + BDH neurons
```

### Pure BDH (`bdh`)

`hatchvision/models/backbones/bdh.py` adapts the BDH architecture
(["The Dragon Hatchling"](https://arxiv.org/abs/2509.26507)) to images:
patch tokens are lifted into a high-dimensional **sparse, positive neuron
space** (ReLU), mixed with **linear attention** (positive query/key
kernels), through an optionally **weight-shared universal layer**.
Scientifically faithful, trains from scratch — expect modest accuracy on
fine-grained datasets.

### Hybrid (`hybrid`) — the practical default for CUB-200

A frozen pretrained torchvision encoder (default `resnet50`) feeds a
BDH-style sparse positive neuron lift. Only the lift + head train (fast,
even on CPU), accuracy is competitive, and the Hebbian memory observes the
same kind of sparse positive "neurons" as pure BDH — so the explainability
pipeline is identical for both. Configure with
`--encoder resnet18|resnet34|resnet50`, `--neuron-dim`, `--unfreeze-encoder`.

## Hebbian feature memory → named concepts

`HebbianFeatureMemory` hooks the backbone's advertised layers and maintains
an EMA of neuron co-activation ("fire together, wire together") plus
class-conditional firing rates. It is **pure observation**: training with or
without it is bit-identical (enforced by a test).

The explainability pipeline then:

1. **clusters** co-activating units into concepts (`cluster_concepts`),
2. attaches the probe images each concept fires on (`find_exemplars`),
3. **grounds** concepts in dataset attributes when available
   (`ground_concepts`) — measuring how much more a concept fires on images
   *with* an attribute than without, and naming it after its top attributes.

Plus pixel-level tools: **Grad-CAM** (no dependencies) and **SHAP**
(optional `shap` extra).

## Web app (`webapp/`)

A zero-build static site (Vercel-ready) that renders the concept graph —
force layout, cluster highlighting, edge filters, light/dark — and, when a
bundle is present, runs **inference fully in the browser** via onnxruntime-web
(vendored, no CDN): upload a photo, get top-5 predictions, and watch the
units/concepts/attributes that fired light up. See
[`webapp/README.md`](webapp/README.md).

## Layout

```
hatchvision/
  data/        DatasetSpec, loader registry (cifar10/100, fashion_mnist,
               cub200 + attributes, imagefolder)
  models/      backbone registry (simple_cnn, resnet*, bdh, hybrid) + head
  hebbian/     HebbianFeatureMemory (observation-only forward hooks)
  explain/     GradCAM, SHAP, concept clustering, attribute grounding
  export/      IVGraph JSON + ONNX inference bundle
  engine/      Trainer / TrainConfig
scripts/       train.py CLI · make_shapes_dataset.py (demo data)
notebooks/     kaggle_cub200.ipynb (GPU training → bundle.zip)
               explainability_demo.ipynb (interactive walkthrough)
webapp/        static graph viewer + in-browser inference (Vercel-ready)
tests/         end-to-end smoke tests for every component
```

## Tests

```bash
python -m pytest tests/ -q
```
