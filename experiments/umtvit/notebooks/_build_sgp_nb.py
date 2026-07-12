"""Generate ``kaggle_umtvit_sgp.ipynb`` — the SGP (SomGraphProvider) notebook.

Run:  python experiments/umtvit/notebooks/_build_sgp_nb.py

One **Run All** on Kaggle trains (or resumes) UMT-ViT on HAM10000, then builds
the **SGP** artifacts (``docs/SGP-ARCHITECTURE.md``): it turns the trained 3-D
SOM into a native ViTreous graph and computes a per-probe-image BMU activation
map, using the numpy-only ``vitreous.som`` core (unit-tested in
``packages/core/tests/test_som.py``). Results render inline (SOM lattice, hit
map, per-depth BMU community maps) and export as ``som.json`` + ``som_bmu.bin``
packs plus a self-contained ``sgp_ham10000.json`` web bundle.

The notebook is kept as plain-Python cells in one place so it reviews as code;
the builder emits nbformat-4 JSON. The heavy computation lives in tested package
functions (``umtvit`` + ``vitreous.som``), so the notebook is a thin shell.
"""

from __future__ import annotations

import json
from pathlib import Path

MD = lambda s: {  # noqa: E731
    "cell_type": "markdown",
    "metadata": {},
    "source": s.strip("\n").splitlines(keepends=True),
}
CODE = lambda s: {  # noqa: E731
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": s.strip("\n").splitlines(keepends=True),
}

cells = []

# --------------------------------------------------------------------------- #
cells.append(MD(
    """
# UMT-ViT · SGP — the SOM as a living ViTreous graph (HAM10000)

**SGP** (SomGraphProvider, `docs/SGP-ARCHITECTURE.md`) renders UMT-ViT's learned
3-D Self-Organizing Map — the experiment's *proven* piece (healthy map, dead
fraction ≈ 0.19; see `docs/UMT-VIT-REPORT.md` §5.2) — as a **native** graph
whose node coordinates are the real neuron lattice, not a force-simulated
layout, and lights it up with a single image's BMU trail across encoder depth.

**One Run All** trains (or resumes) UMT-ViT on HAM10000, then:
1. pulls the trained SOM weights + a handful of probe-image latent volumes,
2. builds the SGP assets with the numpy-only, unit-tested `vitreous.som` core,
3. renders the SOM lattice, hit map, and per-depth BMU community maps inline,
4. exports `som.json` + `som_bmu.bin` packs **and** a self-contained
   `sgp_ham10000.json` web bundle.

### Before you run (once)
1. **Add Input** → search `kmader/skin-cancer-mnist-ham10000` → Add.
2. Enable **GPU** (T4) in notebook settings (CPU works but is slow).
3. Enable **Internet** (to `pip install` the two packages from the repo), *or*
   attach this repository as a dataset and the `sys.path` fallback picks it up.

Then **Run All**. Kaggle persists `/kaggle/working`, so a timed-out run just
re-runs: training auto-resumes from the latest checkpoint.
"""
))

# --------------------------------------------------------------------------- #
cells.append(CODE(
    """
# ─── Knobs ──────────────────────────────────────────────────────────────────
EPOCHS       = 30       # full run ≈ 75 min on a T4 → healthy SOM (Report Run 3).
                        #   Lower for a quick look; the SGP graph is legible from
                        #   ~10 epochs, crisp by 30. Timed out? Just Run All again
                        #   (auto-resume continues from the last checkpoint).
N_PROBE      = 8        # probe images whose BMU trail we export/animate.
COMMUNITY_K  = 12       # k-means community count over neuron weights (som.json).
SEED         = 7
CKPT_DIR     = "/kaggle/working/sgp_ckpt"
OUT_DIR      = "/kaggle/working/sgp_out"
REPO_URL     = "https://github.com/LarsGroep/DragonHatchling"
REPO_BRANCH  = "claude/project-review-muokn9"   # branch carrying vitreous.som + this nb
"""
))

# --------------------------------------------------------------------------- #
cells.append(MD("## 1 · Install both packages (umtvit **and** vitreous)"))
cells.append(CODE(
    """
# SGP is the ONE place both packages coexist: umtvit produces the trained SOM,
# vitreous.som turns it into pack assets. They never import each other.
import importlib, subprocess, sys, os
from pathlib import Path

def _have(mod):
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False

# Prefer an already-attached repo (Kaggle 'Add Input' → the repo as a dataset);
# else clone it. Install both packages editable.
REPO = None
for base in ("/kaggle/input", "/kaggle/working"):
    for p in Path(base).glob("**/experiments/umtvit/pyproject.toml"):
        REPO = p.parents[2]; break
    if REPO: break
if REPO is None:
    REPO = Path("/kaggle/working/DragonHatchling")
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH,
                        REPO_URL, str(REPO)], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                str(REPO / "experiments" / "umtvit")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                str(REPO / "packages" / "core")], check=True)
# sys.path fallback in case editable installs are sandboxed.
sys.path.insert(0, str(REPO / "experiments" / "umtvit"))
sys.path.insert(0, str(REPO / "packages" / "core" / "src"))
print("repo:", REPO)
"""
))

# --------------------------------------------------------------------------- #
cells.append(MD(
    """
## 2 · Config — HAM10000, auto-detected paths

Loads `configs/ham10000.yaml` (the standing GPU-scale preset: dim 256, depth 8,
`16×16×8×64` latent volume, `8×8×8` SOM) and points it at whatever mount Kaggle
gave the HAM10000 dataset. `som_init="data"` + `som_revival=True` are the config
defaults — the structural fix that gives the healthy map.
"""
))
cells.append(CODE(
    """
from umtvit import load_config
import glob

cfg = load_config(str(REPO / "experiments" / "umtvit" / "configs" / "ham10000.yaml"))

# Auto-detect the HAM10000 mount (image folders + metadata CSV) under /kaggle/input.
def _find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return hits

meta = _find("HAM10000_metadata.csv")
imgdirs = sorted({str(Path(p).parent) for p in _find("ISIC_*.jpg")})
if meta:
    cfg.dataset.metadata_csv = meta[0]
if imgdirs:
    cfg.dataset.image_dir = imgdirs
cfg.train.epochs = EPOCHS
cfg.train.seed = SEED
cfg.train.checkpoint_dir = CKPT_DIR
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

print("dataset :", cfg.dataset.name, "| image_size", cfg.dataset.image_size)
print("metadata:", cfg.dataset.metadata_csv)
print("images  :", cfg.dataset.image_dir)
print("SOM grid:", cfg.model.som_grid, "| depth", cfg.model.depth,
      "| volume", (cfg.model.volume_h, cfg.model.volume_w))
"""
))

# --------------------------------------------------------------------------- #
cells.append(MD("## 3 · Build data · model · SOM"))
cells.append(CODE(
    """
import torch
from umtvit.data import UniversalDataset
from umtvit.models.model import UMTViT
from umtvit.models.som3d import Soft3DSOM
from umtvit.engine.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ds = UniversalDataset(cfg, split="train", mode="two_view")   # label-free
eval_ds  = UniversalDataset(cfg, split="test",  mode="eval")       # probes

model = UMTViT(cfg)
som   = Soft3DSOM.from_config(cfg)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M | device {device}")
"""
))

# --------------------------------------------------------------------------- #
cells.append(MD(
    """
## 4 · Train (auto-resume)

Self-supervised: no labels touch training. A per-epoch checkpoint lands in
`CKPT_DIR/_latest.pt`; if this cell re-runs (e.g. after a Kaggle timeout) it
continues from there instead of restarting.
"""
))
cells.append(CODE(
    """
latest = Path(CKPT_DIR) / "_latest.pt"

def _snapshot(epoch, metrics, trainer):
    trainer.save_checkpoint(str(latest))
    if epoch % 5 == 0 or epoch == cfg.train.epochs:
        print(f"epoch {epoch:2d} | loss {metrics.get('loss', float('nan')):.3f} "
              f"| dead {metrics.get('dead_neuron_fraction', float('nan')):.3f} "
              f"| TE {metrics.get('topographic_error', float('nan')):.3f}")

trainer = Trainer(cfg, model, som, train_ds, on_epoch_end=_snapshot)
trainer.train(resume_from=str(latest) if latest.exists() else None)
model.eval()
print("training done — final metrics:", trainer.metrics_history[-1] if trainer.metrics_history else "(none)")
"""
))

# --------------------------------------------------------------------------- #
cells.append(MD(
    """
## 5 · Build the SGP assets (the numpy-only, tested core)

Everything below is a thin call into `vitreous.som` — the same functions
verified in `packages/core/tests/test_som.py`. Volumes come from
`umtvit.eval.extract_probe_volumes` (`[N, H', W', Z, C]`); the SOM weights come
straight off the trained module. No torch maths here — just array hand-off.
"""
))
cells.append(CODE(
    """
import numpy as np
from vitreous.som import (
    build_som_graph_asset, bmu_indices, bmu_map, hit_counts,
    som_communities, som_umatrix, grid_coords,
)

grid    = tuple(int(g) for g in cfg.model.som_grid)          # (Gz, Gy, Gx)
K       = int(np.prod(grid))
weights = som.weights.detach().cpu().numpy().astype(np.float32)   # [K, C]

# Pick an explicit seeded probe set so thumbnails and BMU maps refer to the SAME
# images (extract_probe_volumes picks its own subset — we need index parity).
rng = np.random.default_rng(SEED)
n_take = min(N_PROBE, len(eval_ds))
probe_idx = sorted(rng.choice(len(eval_ds), size=n_take, replace=False).tolist())
imgs = torch.stack([eval_ds[i][0] for i in probe_idx]).to(device)
with torch.no_grad():
    vols = model(imgs)["volume"].detach().cpu().numpy().astype(np.float32)  # [N,H,W,Z,C]
N, Hc, Wc, Z, Cv = vols.shape

# BMU hit counts over ALL probe voxels (node sizing + dead flags).
all_vox = vols.reshape(-1, Cv)
hits    = hit_counts(bmu_indices(all_vox, weights), K)

# som.json — the shared SOM graph asset.
som_asset = build_som_graph_asset(
    weights, grid, hits=hits, community_k=COMMUNITY_K, seed=SEED,
    depth_steps=int(cfg.model.depth), volume_grid=(Hc, Wc),
    provenance={"dataset": cfg.dataset.name, "epochs": int(cfg.train.epochs),
                "params_millions": round(sum(p.numel() for p in model.parameters())/1e6, 2)},
)

# One BMU map per probe image → [Z, H', W'] uint16.
bmu_maps = [bmu_map(vols[i], weights) for i in range(N)]
umat     = som_umatrix(weights, grid)
comm     = som_communities(weights, COMMUNITY_K, seed=SEED)
coords   = grid_coords(grid)

print(f"SOM  : {K} neurons, {len(som_asset['edges'])} lattice edges, "
      f"{som_asset['dead_neurons']} dead ({som_asset['dead_neurons']/K:.1%})")
print(f"probe: {N} images, volume {Hc}x{Wc}x{Z}x{Cv}, "
      f"BMU maps {bmu_maps[0].shape}")
"""
))

# --------------------------------------------------------------------------- #
cells.append(MD(
    """
## 6 · Render the SOM inline (honesty rule: every pixel is measured)

- **Lattice** — the real `8×8×8` neuron grid in 3-D, coloured by community,
  node size ∝ BMU hits, dead neurons hollow. This is the layout SGP renders in
  the workbench (no force simulation — the coordinates *are* the grid).
- **U-matrix** — mean weight-space distance to lattice neighbours per Z-slice;
  dark ridges are cluster boundaries.
"""
))
cells.append(CODE(
    """
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

Gz, Gy, Gx = grid
tab = plt.get_cmap("tab20")
node_col = np.array([tab(c % 20) for c in comm])
alive = hits > 0
size = 8 + 240 * (hits / max(1, hits.max()))

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
zs, ys, xs = coords[:, 0], coords[:, 1], coords[:, 2]
ax.scatter(xs[alive], ys[alive], zs[alive], s=size[alive], c=node_col[alive],
           depthshade=True, edgecolors="k", linewidths=0.3)
ax.scatter(xs[~alive], ys[~alive], zs[~alive], s=20, facecolors="none",
           edgecolors="0.6", linewidths=0.5)
ax.set_title(f"SOM lattice — {int(alive.sum())}/{K} live neurons\\n(colour=community, size=hits)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z = depth")

# U-matrix as Z-slice small multiples.
ax2 = fig.add_subplot(1, 2, 2)
umap = umat.reshape(Gz, Gy, Gx)
cols = int(np.ceil(np.sqrt(Gz)))
rows = int(np.ceil(Gz / cols))
tile = np.ones((rows * (Gy + 1), cols * (Gx + 1))) * np.nan
for z in range(Gz):
    r, c = divmod(z, cols)
    tile[r*(Gy+1):r*(Gy+1)+Gy, c*(Gx+1):c*(Gx+1)+Gx] = umap[z]
im = ax2.imshow(tile, cmap="magma")
ax2.set_title("U-matrix by depth slice (dark = cluster boundary)")
ax2.axis("off")
fig.colorbar(im, ax=ax2, fraction=0.046)
plt.tight_layout(); plt.show()
"""
))

# --------------------------------------------------------------------------- #
cells.append(MD(
    """
## 7 · A single image flowing through the map (the SGP replay)

The view UMT-ViT was missing: pick one probe image and watch **which region of
the map its patches land on as encoder depth `z` advances** — the BMU community
of every voxel, per depth. Migration across depth is what the workbench
animates; it exists regardless of the (documented negative) scale-ordering
result, because it is just *where the image's features quantize*.
"""
))
cells.append(CODE(
    """
img_i = 0
bm = bmu_maps[img_i]                      # [Z, H', W'] neuron indices
comm_of = comm[bm]                        # [Z, H', W'] community per voxel

fig, axes = plt.subplots(1, Z, figsize=(2.0 * Z, 2.4))
for z in range(Z):
    ax = axes[z]
    ax.imshow(comm_of[z], cmap="tab20", vmin=0, vmax=19, interpolation="nearest")
    hot = np.unique(bm[z]).size
    ax.set_title(f"z={z}\\n{hot} neurons", fontsize=9)
    ax.axis("off")
fig.suptitle(f"Probe image {img_i}: BMU community map across encoder depth "
             f"(learned hierarchy, not physical depth)", y=1.06)
plt.tight_layout(); plt.show()

# The migration curve: fraction of voxels changing BMU between consecutive depths.
migration = [float((bm[z] != bm[z-1]).mean()) for z in range(1, Z)]
plt.figure(figsize=(6, 2.6))
plt.plot(range(1, Z), migration, "o-")
plt.ylim(0, 1); plt.xlabel("depth z"); plt.ylabel("voxels re-assigned")
plt.title("BMU migration across depth (this image)"); plt.grid(alpha=0.3)
plt.show()
"""
))

# --------------------------------------------------------------------------- #
cells.append(MD(
    """
## 8 · Export — pack assets + self-contained web bundle

Two outputs, both additive (pack format stays `1.0.0`):

- **Pack assets** — `som.json` (shared) + one `som_bmu.bin` (`[Z,H',W']` uint16)
  per probe, written through the real `vitreous` `PackWriter`. This is the path
  the ViTreous workbench consumes.
- **`sgp_ham10000.json`** — a compact, self-describing bundle (SOM graph + all
  probe BMU maps + thumbnails + metrics) for a drag-drop web viewer, mirroring
  UMT-ViT's `umtvit_web.json` discipline.
"""
))
cells.append(CODE(
    """
import base64, io, json
from PIL import Image
from vitreous.packs.writer import PackWriter

# --- pack assets (workbench path) ---
pack_root = Path(OUT_DIR) / "packs"
for i in range(N):
    w = PackWriter(pack_root / f"{cfg.dataset.name}" / f"probe_{i:02d}")
    w.add_json("som.json", som_asset)
    w.add_array("som_bmu.bin", bmu_maps[i], encoding="raw", dtype="uint16")
print("wrote", N, "SGP packs under", pack_root)

# --- self-contained web bundle (drag-drop path) ---
def _thumb_b64(ds_idx, px=96):
    # Reconstruct a rough RGB thumbnail from the eval dataset's resize-only view.
    x, _ = eval_ds[ds_idx]
    arr = x.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0)) if arr.shape[0] in (1, 3) else arr
    arr = (arr - arr.min()) / (float(arr.max() - arr.min()) + 1e-8)
    im = Image.fromarray((arr * 255).astype("uint8")).convert("RGB").resize((px, px))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

bundle = {
    "sgp_schema_version": 1,
    "dataset": cfg.dataset.name,
    "som": som_asset,
    "probes": [
        {
            "index": int(probe_idx[i]),
            "thumb_png_b64": _thumb_b64(probe_idx[i]),
            "bmu": bmu_maps[i].astype(int).tolist(),   # [Z][H'][W'] neuron ids
        }
        for i in range(N)
    ],
    "provenance": som_asset["provenance"],
}
bundle_path = Path(OUT_DIR) / f"sgp_{cfg.dataset.name}.json"
bundle_path.write_text(json.dumps(bundle, separators=(",", ":")))
size_mb = bundle_path.stat().st_size / 1e6
print(f"wrote {bundle_path}  ({size_mb:.2f} MB)")
print("Download it from the Kaggle output and drop it into the SGP web viewer,")
print("or upload the packs/ tree to Supabase Storage for the live workbench.")
"""
))

# --------------------------------------------------------------------------- #
cells.append(MD(
    """
## 9 · What you just built

- A trained UMT-ViT SOM on HAM10000, rendered as a **native** graph — real
  lattice coordinates, measured U-matrix edges, real BMU hit sizing.
- A per-image **BMU replay across encoder depth** — the workbench's
  activation animation, driven by genuine model geometry.
- Additive pack assets + a self-contained web bundle, pack format unchanged.

Next (SGP roadmap S3–S5, `docs/SGP-ARCHITECTURE.md`): the `SomBrainView` web
component consumes exactly these assets so the map lives inside the ViTreous
four-view workbench with full hover/scrub sync.
"""
))

# --------------------------------------------------------------------------- #

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "kaggle_umtvit_sgp.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
