"""umtvit — Universal Multi-Scale Topographic Vision Transformer (UMT-ViT).

A self-supervised representation-learning experiment: a dual-scale
cross-attention ViT whose encoder layers are uplifted into a 3-D latent voxel
volume that a differentiable 3-D Self-Organizing Map reorganises into a
topology-preserving, inspectable manifold — trained without labels, on any
image dataset, through configuration alone.

This package is the implementation of ``docs/UMT-VIT-ARCHITECTURE.md``. It is
self-contained (no dependency on ``packages/core``) so the universality
requirement holds. Milestone U0 ships the config schema (single source of
truth), the shapes CI dataset, and the package scaffold; model, loss, engine,
eval, and export logic land in U1-U7 (ARCHITECTURE §9).

Subpackages:

- :mod:`umtvit.data`    loaders (imagefolder, csv, shapes), augmentation
                        registry, two-view wrapper, grouped splits (U0/U1)
- :mod:`umtvit.models`  patch embed, cross-attention, fusion, encoder,
                        uplifting, som3d, heads (U2/U3)
- :mod:`umtvit.losses`  ntxent, som, smoothness, ordering, geodesic (U4)
- :mod:`umtvit.engine`  trainer + ablation runner (U4)
- :mod:`umtvit.eval`    linear probe, k-NN, SOM/manifold/Z-axis metrics (U5)
- :mod:`umtvit.export`  latent cube, SOM maps, curves, run report (U6)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, ConfigError, load_config

__all__ = ["__version__", "Config", "ConfigError", "load_config"]
