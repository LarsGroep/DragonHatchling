# Hatchling · Concept Explorer (web app)

A zero-build static site that (1) **classifies an uploaded image entirely in
the browser** via onnxruntime-web, and (2) **explains the prediction** through
the model's Hebbian concept graph — which learned concepts fired, the visual
attributes they're grounded in, and where in the image the model looked.

## Layout

The default graph is **concept-centric**: ~N concept nodes (clusters of
co-activating BDH neurons), linked when they share attributes or species, and
sized by importance. It stays readable no matter how many species/attributes
the dataset has — detail is revealed on demand rather than drawn all at once.

- **Click a concept** (graph or sidebar) → it expands to show its grounded
  **attributes** (gold diamonds, e.g. *wing color: yellow*) and the **species**
  it responds to (squares); everything else dims. Click again to collapse.
- **Full graph** toggle expands every concept at once (dense; for overview).
- Drag to pan, scroll to zoom, drag a node to reposition, `re-layout` to
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

## Deploying your own bundle

```bash
# any dataset — one command, produces graph.json + model.onnx + manifest.json:
python scripts/train.py --dataset imagefolder --root data/mydata \
    --backbone hybrid --epochs 10 --export-bundle webapp

# or run notebooks/kaggle_cub200.ipynb on Kaggle and unzip bundle.zip here
```

The app auto-loads `manifest.json` → `graph.json` (falling back to
`sample-graph.json`, then to drag-and-drop). With no bundle it stays a pure
graph explorer. Iterate on clustering without retraining via
`scripts/rebuild_graph.py` (uses the bundle's `hebbian_state.pt`).

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
