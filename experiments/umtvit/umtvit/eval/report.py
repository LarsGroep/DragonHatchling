"""Evaluation orchestration + run report (ARCHITECTURE §6, §8, §9 row U5).

Ties the six §6 probes into one call and one markdown report:

- :func:`run_evaluation` runs every §6 metric over a set of eval-mode datasets
  (frozen-feature linear probe + cosine k-NN, SOM quality, manifold
  trustworthiness/continuity, Z-axis spectral-centroid ordering) and returns a
  single nested dict — the structure the notebook, the export module, and the
  ablation runner all read.
- :func:`render_report` turns that dict into the markdown run report: a results
  table per §6 family, the honest Z-axis monotonicity verdict, and the
  SSL-yardstick caveat (frozen-feature numbers are *not* comparable to
  supervised end-to-end results such as DSCATNet's 97.8%).

Labels enter the experiment only here: probe/k-NN skip gracefully (report
``None``) in fully-unlabeled mode, while the label-free metrics (SOM, manifold,
Z-axis) always run.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from umtvit.config import Config
from umtvit.eval.features import FrozenFeatures, extract_features
from umtvit.eval.knn import knn_accuracy
from umtvit.eval.linear_probe import linear_probe
from umtvit.eval.manifold import trustworthiness_continuity
from umtvit.eval.som_metrics import som_metrics
from umtvit.eval.zaxis_probe import extract_probe_volumes, zaxis_probe
from umtvit.models.som3d import Soft3DSOM

__all__ = ["run_evaluation", "render_report"]


def _pick_eval_dataset(datasets: Mapping[str, Any]):
    """The split the label-free metrics score: prefer test, then val, then train."""
    for key in ("test", "val", "train"):
        if datasets.get(key) is not None:
            return datasets[key]
    return None


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a scalar for the markdown table (``n/a`` for None/NaN)."""
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{digits}f}"


def run_evaluation(
    model: torch.nn.Module,
    som: Soft3DSOM,
    cfg: Config,
    datasets: Mapping[str, Any],
    *,
    seed: int = 0,
    feature_batch_size: int = 64,
    probe_steps: int = 300,
    knn_k: int = 5,
    manifold_k: int = 7,
    manifold_max_n: int = 400,
    som_max_imgs: int = 64,
    som_sample_voxels: int = 2048,
    zaxis_images: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run every ARCHITECTURE §6 metric over eval-mode datasets.

    Args:
        model: Frozen UMT-ViT encoder producing ``{"pooled","volume",...}``.
        som: The trained 3-D SOM to score.
        cfg: The run config (recorded in the report header).
        datasets: Mapping of split name → eval-mode dataset (``(image, label)``
            per item). ``"train"`` is the probe/k-NN reference set; the
            label-free metrics score ``"test"`` (falling back to ``"val"`` then
            ``"train"``). A missing ``"train"`` ⇒ probe/k-NN are ``None``.
        seed: Seed for every subsampled metric (probe init, voxel/manifold/
            Z-axis subsamples) — makes the whole report reproducible.
        probe_steps / knn_k / manifold_k / manifold_max_n / som_max_imgs /
        som_sample_voxels / zaxis_images: per-metric budgets (see each module).
        device: Forward-pass device; defaults to the model's device.

    Returns:
        A nested dict with keys ``linear_probe`` (dict|None), ``knn`` (dict|None),
        ``som`` (dict), ``manifold`` (dict), ``zaxis`` (dict), and ``meta``
        (dataset sizes + config summary). Every numeric leaf is finite or NaN.
    """
    train_ds = datasets.get("train")
    eval_ds = _pick_eval_dataset(datasets)

    # Frozen features: the probe/k-NN reference (train) and the scored split.
    train_feats: Optional[FrozenFeatures] = (
        extract_features(
            model, train_ds, device=device, batch_size=feature_batch_size
        )
        if train_ds is not None
        else None
    )
    eval_feats: Optional[FrozenFeatures] = (
        extract_features(
            model, eval_ds, device=device, batch_size=feature_batch_size
        )
        if eval_ds is not None
        else None
    )

    probe = None
    knn = None
    if train_feats is not None and eval_feats is not None:
        probe = linear_probe(
            train_feats, eval_feats, steps=probe_steps, seed=seed, device=device
        )
        knn = knn_accuracy(train_feats, eval_feats, k=knn_k)

    som_scores = (
        som_metrics(
            model,
            som,
            eval_ds,
            max_imgs=som_max_imgs,
            sample_voxels=som_sample_voxels,
            seed=seed,
            device=device,
        )
        if eval_ds is not None
        else {
            "quantization_error": float("nan"),
            "topographic_error": float("nan"),
            "dead_neuron_fraction": float("nan"),
        }
    )

    if eval_feats is not None and len(eval_feats) > 0:
        manifold = trustworthiness_continuity(
            eval_feats.pixels,
            eval_feats.pooled,
            k=manifold_k,
            max_n=manifold_max_n,
            seed=seed,
        )
    else:
        manifold = {
            "trustworthiness": float("nan"),
            "continuity": float("nan"),
            "k": manifold_k,
            "n": 0,
        }

    volumes = (
        extract_probe_volumes(
            model, eval_ds, n_images=zaxis_images, seed=seed, device=device
        )
        if eval_ds is not None
        else None
    )
    zaxis = zaxis_probe(volumes if volumes is not None else [])

    meta = {
        "dataset": cfg.dataset.name,
        "n_train": (len(train_feats) if train_feats is not None else 0),
        "n_eval": (len(eval_feats) if eval_feats is not None else 0),
        "depth": cfg.model.depth,
        "som_grid": list(cfg.model.som_grid),
        "labeled": bool(eval_feats is not None and eval_feats.labeled),
    }

    return {
        "linear_probe": probe,
        "knn": knn,
        "som": som_scores,
        "manifold": manifold,
        "zaxis": zaxis,
        "meta": meta,
    }


def render_report(results: Dict[str, Any]) -> str:
    """Render a :func:`run_evaluation` result dict as a markdown run report.

    Sections: frozen-feature read-out (probe + k-NN, or an unlabeled note), SOM
    quality, manifold quality, and the Z-axis ordering probe with its honest
    monotonicity verdict — closed by the SSL-yardstick caveat.
    """
    meta = results.get("meta", {})
    probe = results.get("linear_probe")
    knn = results.get("knn")
    som = results.get("som", {})
    manifold = results.get("manifold", {})
    zaxis = results.get("zaxis", {})

    lines = [
        "# UMT-ViT Evaluation Report",
        "",
        f"- Dataset: **{meta.get('dataset', 'unknown')}**",
        f"- Eval samples: {meta.get('n_eval', 0)} "
        f"(probe reference: {meta.get('n_train', 0)})",
        f"- Volume depth (Z / learned hierarchy): {meta.get('depth', '?')} · "
        f"SOM grid: {meta.get('som_grid', '?')}",
        "",
        "## 1. Frozen-feature read-out (labels enter only here)",
        "",
    ]

    if probe is None and knn is None:
        lines.append(
            "_Fully-unlabeled mode: no `label_column`, so the linear probe and "
            "k-NN are skipped (the label-free metrics below still run)._"
        )
    else:
        lines += [
            "| Metric | Value | Chance |",
            "|---|---|---|",
        ]
        if probe is not None:
            lines.append(
                f"| Linear probe accuracy | {_fmt(probe.get('accuracy'))} | "
                f"{_fmt(probe.get('chance'))} |"
            )
        if knn is not None:
            lines.append(
                f"| k-NN (k={knn.get('k', '?')}, cosine) | "
                f"{_fmt(knn.get('accuracy'))} | {_fmt(knn.get('chance'))} |"
            )
        if probe is not None and probe.get("per_class_accuracy"):
            per_class = ", ".join(
                f"{c}:{_fmt(a, 3)}" for c, a in sorted(probe["per_class_accuracy"].items())
            )
            lines += ["", f"Per-class probe recall — {per_class}"]

    lines += [
        "",
        "## 2. SOM quality",
        "",
        "| Metric | Value | Direction |",
        "|---|---|---|",
        f"| Quantization error | {_fmt(som.get('quantization_error'))} | lower better |",
        f"| Topographic error | {_fmt(som.get('topographic_error'))} | lower better |",
        f"| Dead-neuron fraction | {_fmt(som.get('dead_neuron_fraction'))} | lower better |",
        "",
        "## 3. Manifold quality (input ↔ latent)",
        "",
        "| Metric | Value | Ideal |",
        "|---|---|---|",
        f"| Trustworthiness (k={manifold.get('k', '?')}) | "
        f"{_fmt(manifold.get('trustworthiness'))} | 1.0 |",
        f"| Continuity (k={manifold.get('k', '?')}) | "
        f"{_fmt(manifold.get('continuity'))} | 1.0 |",
        "",
        "## 4. Z-axis ordering probe (learned hierarchy, not physical depth)",
        "",
    ]

    per_channel = zaxis.get("per_channel_centroids", [])
    channel_mean = zaxis.get("channel_mean_centroids", [])
    if per_channel:
        lines.append(
            "Per-channel spectral centroids by depth (fair measure): "
            + ", ".join(_fmt(c, 3) for c in per_channel)
        )
    if channel_mean:
        lines.append(
            "Channel-mean spectral centroids by depth (legacy): "
            + ", ".join(_fmt(c, 3) for c in channel_mean)
        )
    lines += [
        "",
        f"**Verdict:** {zaxis.get('verdict', 'undetermined')}",
        "",
        "## Caveat",
        "",
        "These are **self-supervised frozen-feature yardsticks** (linear probe / "
        "k-NN on a frozen encoder). They measure how linearly separable the "
        "label-free representation is — they are *not* comparable to supervised "
        "end-to-end accuracies (e.g. DSCATNet's 97.8% on HAM10000). The Z-axis "
        "is a learned representational hierarchy; whether scale ordering emerged "
        "is reported above as measured, never assumed.",
    ]
    return "\n".join(lines)
