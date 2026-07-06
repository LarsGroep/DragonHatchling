"""vitreous — the ViTreous explainable-ViT core.

One Python package containing all instrumentation, XAI, Gaussian, graph, and
pack-writing logic (§2). Kaggle notebooks and the HF Space are thin shells
around it — one code path, two venues.

Submodules (M0 ships interfaces; logic lands per the §16 roadmap):

- :mod:`vitreous.data`        dataset spec + adapter ABC + registry (working)
- :mod:`vitreous.models`      timm ViT-S/16 wrapper (M1)
- :mod:`vitreous.instrument`  observation-only hooks -> Trace (M1)
- :mod:`vitreous.xai`         attribution suite + faithfulness eval (M2)
- :mod:`vitreous.gaussians`   Gaussian Feature Field builder (M3)
- :mod:`vitreous.graph`       GraphProvider + ViTTokenGraphProvider (M3)
- :mod:`vitreous.projections` dataset-level PCA/UMAP/t-SNE projections (M3)
- :mod:`vitreous.concepts`    k-sparse-autoencoder concept tier + k-means fallback (M4)
- :mod:`vitreous.packs`       Explanation Pack manifest models + writer
- :mod:`vitreous.storage`     StorageAdapter (local / Supabase / HF backends) (M4)

Only the M0 runtime deps (pydantic, numpy, jsonschema) are required to import
this package. torch/timm are optional ``[ml]`` extras.
"""

from __future__ import annotations

__version__ = "0.1.0"

# The manifest models are the load-bearing M0 export; import them eagerly so
# `from vitreous import PackManifest` works.
from .packs import PackManifest

__all__ = ["__version__", "PackManifest"]
