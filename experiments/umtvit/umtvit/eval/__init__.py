"""umtvit.eval — evaluation suite (ARCHITECTURE §6, §9 row U5).

Labels enter the experiment only here. The suite (all CPU-runnable):

- :func:`extract_features` — frozen pooled features + labels + flattened pixels
  (the shared input every probe reads).
- :func:`linear_probe` / :func:`knn_accuracy` — the SSL frozen-feature yardsticks
  (pure-torch logistic regression; cosine k-NN). ``None`` in unlabeled mode.
- :func:`som_metrics` — SOM quantization error, topographic error, dead-neuron
  fraction on a seeded held-out voxel subsample.
- :func:`trustworthiness_continuity` — manifold quality between input pixels and
  latent features (both Venna-Kaski measures, ``[0, 1]``).
- :func:`zaxis_probe` — per-slice spectral centroids + monotonicity verdict, the
  measured answer to "did scale ordering emerge?".
- :func:`run_evaluation` / :func:`render_report` — run every §6 metric and
  render the markdown run report.
"""

from __future__ import annotations

from umtvit.eval.features import FrozenFeatures, extract_features, standardize
from umtvit.eval.knn import knn_accuracy
from umtvit.eval.linear_probe import linear_probe
from umtvit.eval.manifold import trustworthiness_continuity
from umtvit.eval.report import render_report, run_evaluation
from umtvit.eval.som_metrics import som_metrics
from umtvit.eval.zaxis_probe import (
    extract_probe_volumes,
    monotonicity_verdict,
    radial_spectrum,
    spectral_centroid,
    zaxis_probe,
)

__all__ = [
    "FrozenFeatures",
    "extract_features",
    "standardize",
    "linear_probe",
    "knn_accuracy",
    "som_metrics",
    "trustworthiness_continuity",
    "radial_spectrum",
    "spectral_centroid",
    "extract_probe_volumes",
    "zaxis_probe",
    "monotonicity_verdict",
    "run_evaluation",
    "render_report",
]
