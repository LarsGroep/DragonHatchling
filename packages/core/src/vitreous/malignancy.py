"""Malignancy lens — honest, label-free clinical readings of a pack (docs/MALIGNANCY-LENS.md).

Three readouts over a HAM10000-style pack, none of which invent a signal that
isn't in the data:

1. **malignant probability** — a deterministic sum of the diagnosis softmax over
   the malignant class group (a :class:`~vitreous.data.Taxonomy`);
2. **category coordinate** — the softmax-weighted position on a coarse ordinal
   ``benign → in-situ → invasive`` axis (a *category* reading, never clinical
   AJCC/TNM staging, which is not present in a dermatoscopic surface image);
3. **manifold position + OOD** — where a lesion's SSL/CLS feature projects on the
   learned benign↔malignant axis, plus an off-axis residual that flags
   out-of-distribution inputs (the honest refusal for e.g. phone uploads).

**Import discipline (M0 rule):** numpy-only. No torch. The per-image feature is
pulled from data the pack already ships (the final-step CLS of ``tokens.bin``);
the only new artifact is the small dataset-level axis asset built here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "malignant_indices",
    "malignant_probability",
    "category_levels",
    "expected_category",
    "hard_category",
    "build_malignancy_axis",
    "project_feature",
]


# --------------------------------------------------------------------------- #
# derived readouts (axes 1 & 2) — pure functions of the diagnosis softmax
# --------------------------------------------------------------------------- #


def malignant_indices(class_names: Sequence[str], taxonomy: Any) -> List[int]:
    """Class indices flagged malignant by ``taxonomy`` (a ``data.Taxonomy``)."""
    mal = taxonomy.malignant if hasattr(taxonomy, "malignant") else dict(taxonomy)
    return [i for i, c in enumerate(class_names) if bool(mal.get(c, False))]


def malignant_probability(
    probabilities: Sequence[float], malignant_idx: Sequence[int]
) -> float:
    """``Σ P(class)`` over the malignant classes → scalar in ``[0, 1]``."""
    p = np.asarray(probabilities, dtype=np.float64)
    if len(malignant_idx) == 0:
        return 0.0
    return float(np.clip(p[list(malignant_idx)].sum(), 0.0, 1.0))


def category_levels(class_names: Sequence[str], taxonomy: Any) -> np.ndarray:
    """Per-class ordinal category level ``[num_classes]`` (0 = least advanced)."""
    lvl = taxonomy.category_level if hasattr(taxonomy, "category_level") else dict(taxonomy)
    return np.asarray([int(lvl.get(c, 0)) for c in class_names], dtype=np.int64)


def expected_category(probabilities: Sequence[float], levels: np.ndarray) -> float:
    """Softmax-weighted category coordinate ``Σ P(c)·level(c)`` → ``[0, K-1]``.

    A smooth "how far along the benign→in-situ→invasive axis" readout that moves
    with the probability mass, rather than a hard class snap.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    lv = np.asarray(levels, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    return float((p * lv).sum() / s)


def hard_category(probabilities: Sequence[float], levels: np.ndarray) -> int:
    """Ordinal level of the argmax class (the discrete category reading)."""
    p = np.asarray(probabilities, dtype=np.float64)
    return int(levels[int(p.argmax())])


# --------------------------------------------------------------------------- #
# unsupervised manifold axis (axis 3)
# --------------------------------------------------------------------------- #


def build_malignancy_axis(
    features: np.ndarray,
    is_malignant: Sequence[bool],
    *,
    space: str = "cls_final",
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    residual_pct: float = 95.0,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the dataset-level ``malignancy_axis.json`` from label-free features.

    Parameters
    ----------
    features:
        ``[N, D]`` per-image features (e.g. final-step CLS embeddings).
    is_malignant:
        ``[N]`` bool — the malignant group flag per image (from the taxonomy +
        each image's diagnosis). This is the *only* place labels touch the axis,
        and only to place the two endpoints; the projection itself is unsupervised.
    lo_pct / hi_pct:
        Percentiles of the benign / malignant projections used as the ``[0, 1]``
        calibration anchors (robust to outliers).
    residual_pct:
        Percentile of the off-axis residual over all inputs → the OOD threshold.

    Returns
    -------
    dict
        The ``malignancy_axis.json`` payload (JSON-serializable).
    """
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"features must be [N, D], got {f.shape}")
    m = np.asarray(is_malignant, dtype=bool)
    if m.shape[0] != f.shape[0]:
        raise ValueError("is_malignant must have one flag per feature row")
    if not m.any() or not (~m).any():
        raise ValueError("need at least one benign and one malignant example")

    c_b = f[~m].mean(axis=0)
    c_m = f[m].mean(axis=0)
    u = c_m - c_b
    nrm = float(np.linalg.norm(u))
    if nrm == 0:
        raise ValueError("benign and malignant centroids coincide; no axis")
    u = u / nrm

    proj = f @ u
    anchor_lo = float(np.percentile(proj[~m], lo_pct))
    anchor_hi = float(np.percentile(proj[m], hi_pct))
    if anchor_hi <= anchor_lo:  # degenerate separation — fall back to full range
        anchor_lo, anchor_hi = float(proj.min()), float(proj.max() + 1e-9)

    # Off-axis residual: distance from the line through c_b along u.
    d = f - c_b
    along = d @ u
    resid = np.linalg.norm(d - along[:, None] * u, axis=1)
    residual_thr = float(np.percentile(resid, residual_pct))

    return {
        "provider": "malignancy_axis",
        "space": space,
        "dim": int(f.shape[1]),
        "u": [float(x) for x in u],
        "centroid_benign": [float(x) for x in c_b],
        "anchor_lo": anchor_lo,
        "anchor_hi": anchor_hi,
        "residual_threshold": residual_thr,
        "provenance": {
            "n_benign": int((~m).sum()),
            "n_malignant": int(m.sum()),
            "lo_pct": lo_pct,
            "hi_pct": hi_pct,
            "residual_pct": residual_pct,
            **(provenance or {}),
        },
    }


def project_feature(feature: Sequence[float], axis: Dict[str, Any]) -> Dict[str, Any]:
    """Project one feature onto a built axis → ``{position, residual, ood}``.

    ``position`` is the clamped ``[0, 1]`` coordinate along the benign→malignant
    axis; ``residual`` is the off-axis distance; ``ood`` flags a feature whose
    residual exceeds the axis's stored threshold (the honest refusal signal).
    """
    f = np.asarray(feature, dtype=np.float64).ravel()
    u = np.asarray(axis["u"], dtype=np.float64)
    c_b = np.asarray(axis["centroid_benign"], dtype=np.float64)
    if f.shape != u.shape:
        raise ValueError(f"feature dim {f.shape} != axis dim {u.shape}")
    lo = float(axis["anchor_lo"])
    hi = float(axis["anchor_hi"])
    d = f - c_b
    along = float(d @ u)
    position = float(np.clip((f @ u - lo) / (hi - lo if hi > lo else 1.0), 0.0, 1.0))
    residual = float(np.linalg.norm(d - along * u))
    thr = float(axis.get("residual_threshold", np.inf))
    return {"position": position, "residual": residual, "ood": bool(residual > thr)}
