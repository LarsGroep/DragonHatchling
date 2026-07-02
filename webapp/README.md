# IVGraph web viewer

A zero-dependency static web app that visualizes the Hebbian concept graphs
exported by hatchvision (`hatchvision.export.export_ivgraph`).

## What it shows

- **Units** (small circles) — hidden-layer channels tracked by the Hebbian
  feature memory, colored by concept cluster and sized by mean firing rate.
- **Concepts** (ringed circles) — clusters of units that fire together.
- **Classes** (gray squares) — dataset labels, connected to the concepts
  that respond to them (dashed = class-affinity edges).
- Solid gray unit-unit edges are **co-activation** strength.

Interactions: hover for tooltips, click a node (or a concept in the sidebar)
to inspect and highlight its cluster, drag nodes, scroll to zoom, drag the
background to pan. Filter edge kinds and the co-activation threshold from
the header. Light/dark theme follows the OS preference.

## Loading a graph

The app auto-loads `sample-graph.json` from its own directory. To view your
own run, either drag-and-drop the exported JSON onto the page, use the
"open JSON…" button, or replace `sample-graph.json` before deploying:

```bash
python scripts/train.py --dataset cifar10 --backbone simple_cnn \
    --epochs 3 --hebbian --export-graph webapp/sample-graph.json
```

## Run locally

```bash
cd webapp
python3 -m http.server 8000   # any static server works
```

## Deploy to Vercel

```bash
cd webapp
npx vercel deploy             # no build step; it's a static site
```
