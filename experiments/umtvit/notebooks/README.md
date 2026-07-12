# UMT-ViT notebooks

Three Kaggle-ready notebooks live here. They run the **same** self-supervised
experiment — a dual-scale cross-attention ViT lifted into a 3-D latent voxel
volume that a differentiable 3-D SOM reorganises, trained without labels — and
produce the **same** visuals, artifacts, and `umtvit_web.json` (web-explorer
bundle, schema v1). They differ only in *where the code lives* and *what they
are for*.

| Notebook | Role | Code source | Edit? |
|---|---|---|---|
| `kaggle_umtvit.ipynb` | Canonical, self-contained source — the owner's daily driver | All stages defined **inline** in the notebook | Yes — this is the working copy |
| `kaggle_umtvit_package.ipynb` | Milestone-U6 package form — same experiment via the installed package | Imports the **`umtvit`** package (config → data → model → engine → eval → export) | Yes — the config cell (`CONFIG_NAME`, `OVERRIDES`) |
| `kaggle_umtvit_ham10000_run3.ipynb` | Frozen executed-results artifact | Inline (a run of the canonical notebook) | No — never edit; it is a record |
| `kaggle_umtvit_sgp.ipynb` | **SGP** — trains/resumes UMT-ViT on HAM10000, then renders the trained 3-D SOM as a native ViTreous graph + per-image BMU replay, and exports `som.json`/`som_bmu.bin` packs + a `sgp_ham10000.json` web bundle (`docs/SGP-ARCHITECTURE.md`) | Imports **`umtvit`** + **`vitreous.som`** (the numpy-only SGP core); regenerate with `_build_sgp_nb.py` | Edit via the builder |

## `kaggle_umtvit.ipynb` — canonical self-contained notebook

The reference implementation and the owner's day-to-day notebook. Every stage —
config, data pipeline, dual-scale backbone, 3-D SOM, losses, training loop,
visualisations, evaluation, and the artifact/web-bundle exporters — is written
**inline** so the whole experiment is legible top-to-bottom in one file with no
package dependency. It carries the current conveniences (HAM10000 path
auto-detect, non-square-safe crop, CUDA-OOM guard). It runs the **entire roadmap
in one notebook**: training with **periodic per-epoch checkpoints and
`resume:"auto"`** (a timed-out Kaggle run continues on re-run), evaluation, an
optional **six-axis ablation matrix** (`RUN_ABLATIONS`), and a report + web
bundle that together form the full experimental record. This is the notebook to
prototype in; the `umtvit` package is the hardened extraction of exactly this
code.

## `kaggle_umtvit_package.ipynb` — package-driven twin (this milestone, U6)

The same experiment, but every stage is driven by the installed **`umtvit`**
package rather than inline definitions:

- **setup** — `pip install -e experiments/umtvit` (with a `sys.path` fallback so
  it works on Kaggle-with-repo *and* locally), then imports.
- **config** — `CONFIG_NAME = "shapes"` (CPU/CI default) selects
  `configs/<name>.yaml` via `umtvit.load_config`; swap to `"ham10000"` /
  `"eurosat"` with one string. An optional `OVERRIDES` dict patches any field
  before validation. HAM10000's Kaggle paths are auto-detected.
- **stages** — `UniversalDataset`, `UMTViT`, `Soft3DSOM`, `engine.Trainer`
  (with a snapshotting `on_epoch_end` callback), `eval.run_evaluation` +
  `render_report`, and an optional `engine.AblationRunner` sweep behind
  `RUN_ABLATIONS`.
- **outputs** — the identical matplotlib visuals/animations, the
  `umtvit_artifacts/<name>/` bundle, and the `umtvit_web.json` schema-v1
  drop-in for the web app's `/umtvit` explorer.

Because the heavy lifting lives in the package (which is unit-tested), this
notebook stays thin and is the form to reach for once the API has stabilised.

## `kaggle_umtvit_ham10000_run3.ipynb` — executed results artifact

A **frozen** copy of the canonical notebook executed end-to-end on HAM10000
(Run 3), kept **with its outputs** as an evidence record (probe ≈ 0.77, healthy
SOM). It is never edited and never re-run in place — treat it as a committed
result, not a template. To reproduce or extend it, run `kaggle_umtvit.ipynb`
(or the package twin) with the `ham10000` config.
