# Hatchling · Concept Explorer (web app)

A zero-build static site that (1) **classifies an uploaded image entirely in
the browser** via onnxruntime-web, and (2) **explains the prediction** through
the model's Hebbian concept graph — which learned concepts fired, the visual
attributes they're grounded in, and where in the image the model looked.

## ▶ Demo tour — Hebbian activation regions + SHAP influence

The **▶ demo tour** button (header) runs a guided walkthrough for
presenting the core ideas; every step is also available stand-alone:

1. **Class activation regions** — search any class in the *Class activation
   regions* panel: the concepts (regions of co-firing neurons) that
   represent it glow blue in the graph, weighted by the class's Hebbian
   firing fingerprint. This is *class representation as a region in neuron
   space*.
2. **SHAP influence** — after classifying an image, the *SHAP influence*
   panel shows each concept's signed contribution to the predicted logit.
   For models whose readout is linear in the neurons (`hybrid`, `bdh`) these
   are **exact Shapley values**, computed closed-form in the browser from
   `explain.json`'s weight matrix — and the panel shows the additivity
   check: `logit = base + concepts + other paths`. Click another prediction
   row for a contrastive explanation.
3. **Image vs. class region** — overlays the image's actual firing pattern
   (orange) on the predicted class's typical region (blue) with a match
   score, so deviations from a "textbook" example stand out.
4. **Per-concept image regions** — the occlusion sweep records the neuron
   activations of every masked run, so after *Where did it look?* a dropdown
   switches the heatmap from the prediction to any firing concept ("where
   does this Hebbian region look in the photo") at no extra compute.

Steps degrade gracefully: without `explain.json` the tour covers only the
graph, without a model bundle only the region steps.

All of it is dataset-agnostic — the same `explain.json` is produced by
`scripts/train.py --export-bundle` for any registered dataset, and class
fingerprints can be rebuilt from `hebbian_state.pt` via
`scripts/rebuild_graph.py --explain-out` without retraining. (The SHAP
weight matrix needs the trained model, so it's produced at export time.)

## Layout

The default graph is **concept-centric**: ~N concept nodes (clusters of
co-activating BDH neurons), linked when they share attributes or species, and
sized by importance. It stays readable no matter how many species/attributes
the dataset has — detail is revealed on demand rather than drawn all at once.

- **Click a concept** (graph or sidebar) → it expands to show its grounded
  **attributes** (gold diamonds, e.g. *wing color: yellow*) and the **species**
  it responds to (squares); everything else dims. Click again to collapse.
- **Full graph** toggle expands every concept at once (dense; for overview).
- **Neurons** toggle shows the **full Hebbian network** at the unit level:
  every tracked neuron as a dot **colored by the concept it belongs to**,
  linked by co-activation (real Hebbian edge weights when the graph carries
  them, otherwise concept co-membership). Hover a neuron to trace its
  co-firing links and see its concept; click to focus its neighborhood.
  After you classify an image the neurons **light up by how strongly they
  fire**, so you watch the raw network activate.
- Drag to pan, scroll to zoom, drag a node to reposition, `Layout` to
  re-run the force layout and auto-fit.

## Classify + explain

When a bundle (`manifest.json` + `model.onnx`) sits next to the page, the
**Classify an image** panel is active:

1. Drop or choose a photo → **top-5 predictions** (names from the manifest,
   aligned with the model's logits).
2. **Why this prediction** — the concepts that fired most, and a **visual
   evidence** bar list aggregating their grounded attributes weighted by
   firing strength ("the model saw: blue upperparts, conical bill, …").
3. The graph enters **live mode**: concepts glow by how strongly they fired
   and the top concept auto-expands.
4. **Where did it look?** runs **occlusion saliency** — the image is diced
   into a grid, each cell blanked in turn and re-run (batched, one row at a
   time with a live progress bar), and the drop in the predicted class's
   probability is painted back as a heatmap over the photo. This is
   class-specific and faithful (no gradient approximation).

Everything runs locally; nothing leaves the browser. onnxruntime-web is
vendored under `vendor/` (no CDN — the strict setup below forbids external
requests anyway).

## Cross-origin isolation (speed)

`vercel.json` sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`. These enable
`SharedArrayBuffer`, which lets onnxruntime-web use **multiple wasm threads** —
several times faster inference and saliency. Without the headers the app still
works, single-threaded (auto-detected via `crossOriginIsolated`). All
resources are same-origin, so isolation doesn't block anything.

## Swapping datasets — no redeploy needed

**The fast path: drag `bundle.zip` onto the page.** Any training run's bundle
(`scripts/train.py --export-bundle` zipped up, or Kaggle's `bundle.zip`)
loads **directly in the browser** — unzipped client-side, model and all, and
becomes the active dataset immediately. It's persisted in IndexedDB so it
survives reloads, and the **header switcher** moves between the deployed
bundle and any stored ones (🗑 removes one). Loose files work too: select
`manifest.json + graph.json + model.onnx + explain.json` together via
**⤒ Load bundle…**. Nothing is uploaded anywhere — it stays in your browser.

That makes the swap-a-dataset loop: train → download bundle.zip → drop it on
the deployed site. Done.

## Deploying a bundle as the site default

```bash
# any dataset — one command, produces graph.json + model.onnx + manifest.json
# + explain.json + hebbian_state.pt:
python scripts/train.py --dataset imagefolder --root data/mydata \
    --backbone hybrid --epochs 10 --export-bundle webapp

# or run notebooks/kaggle_cub200.ipynb on Kaggle and unzip bundle.zip here
```

The app auto-loads `manifest.json` → `explain.json` → `graph.json` (falling
back to `sample-graph.json`, then to drag-and-drop). With no bundle it stays
a pure graph explorer; with no `explain.json` the region/SHAP panels hide.
Iterate on clustering without retraining via `scripts/rebuild_graph.py`
(uses the bundle's `hebbian_state.pt`; add `--explain-out explain.json` to
refresh the class fingerprints too).

## Run locally

```bash
cd webapp
python3 -m http.server 8000        # single-threaded (no COOP/COEP headers)
```

For threaded inference locally, serve with the isolation headers (any static
server that can set `Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp`).

## Deploy to Vercel

```bash
cd webapp
git lfs pull                  # model.onnx is tracked with Git LFS — fetch the real file
npx vercel deploy --prod      # no build step; static site + headers from vercel.json
```

If the deployed `model.onnx` is still a Git LFS *pointer* (a deploy that
skipped `git lfs pull`), the app detects it and shows an actionable error
instead of a cryptic runtime failure.
