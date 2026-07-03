# IVGraph web viewer + in-browser inference

A zero-dependency static web app that visualizes the Hebbian concept graphs
exported by hatchvision and — when an inference bundle is deployed alongside
it — classifies uploaded images entirely in the browser and lights up the
parts of the graph that fired.

## What it shows

- **Units** (small circles) — hidden-layer neurons tracked by the Hebbian
  feature memory, colored by concept cluster and sized by mean firing rate.
- **Concepts** (ringed circles) — clusters of units that fire together.
- **Classes** (gray squares) — dataset labels, connected to the concepts
  that respond to them (dashed = class-affinity edges).
- **Attributes** (gold diamonds) — human-readable visual features
  ("wing color: yellow") a concept was grounded in, when the dataset has
  attribute annotations (CUB-200 does).
- Solid gray unit-unit edges are **co-activation** strength.

Interactions: hover for tooltips, click a node (or a concept in the sidebar)
to inspect and highlight its cluster, drag nodes, scroll to zoom, drag the
background to pan. Filter edge kinds and the co-activation threshold from
the header. Light/dark theme follows the OS preference.

## Classify an image

If `manifest.json` + `model.onnx` are present (exported by
`--export-bundle` or the Kaggle notebook), a **"Classify an image"** panel
appears: upload or drop a photo to get top-5 predictions, and the graph
switches to *live mode* — node brightness shows how strongly each unit /
concept / attribute fired for that image, and predicted classes glow.
Inference runs locally via onnxruntime-web (vendored in `vendor/`, no CDN,
nothing leaves the browser).

## Deploying your own bundle

```bash
# any dataset:
python scripts/train.py --dataset imagefolder --root data/mydata \
    --backbone hybrid --epochs 10 --export-bundle webapp

# or run notebooks/kaggle_cub200.ipynb on Kaggle and unzip bundle.zip here
```

The app auto-loads `manifest.json` → `graph.json` from its own directory
(falling back to `sample-graph.json`). You can also drag-and-drop any
IVGraph `.json` onto the page.

## Run locally

```bash
cd webapp
python3 -m http.server 8000   # any static server works
```

## Deploy to Vercel

```bash
cd webapp
git lfs pull                  # if model.onnx is tracked with Git LFS!
npx vercel deploy --prod      # no build step; it's a static site
```

If the deployed `model.onnx` is a Git LFS *pointer* instead of the real
model (a deploy that skipped `git lfs pull`), the app shows an explicit
error when you try to classify an image.
