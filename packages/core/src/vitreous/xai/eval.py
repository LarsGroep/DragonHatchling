"""Faithfulness evaluation (§6, §10).

Two families:

* **deletion_insertion** — perturbation curves. Patches are masked in ranked
  order (most-important first) and the target-class probability is tracked. In
  the *deletion* direction we start from the full image and progressively mask;
  a faithful ranking makes the probability drop fast → **low** deletion AUC. In
  the *insertion* direction we start from the fully-masked baseline and
  progressively reveal; a faithful ranking makes the probability rise fast →
  **high** insertion AUC. Mask value is the **zero (black) baseline** — chosen
  for determinism and because it matches the IG baseline (documented; a custom
  ``baseline`` scalar is accepted).

* **method_agreement** — pairwise Spearman rank-correlation between the token
  rankings of different methods (the UI reads *disagreement as signal*). Any
  attribution shape is reduced to a per-patch ``[grid²]`` vector first
  (:func:`to_patch_vector`), so heterogeneous methods are comparable.

Spearman is implemented locally (average-rank ties → Pearson of ranks); scipy is
not a dependency. All torch imports are lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DIResult:
    """One ranking's deletion/insertion curves + AUCs."""

    deletion: List[float] = field(default_factory=list)
    insertion: List[float] = field(default_factory=list)
    deletion_auc: float = 0.0
    insertion_auc: float = 0.0
    steps: int = 20
    target: int = 0


@dataclass
class FaithfulnessResult:
    """Aggregate across methods: curves + AUCs + method-agreement matrix."""

    deletion_curves: Dict[str, List[float]] = field(default_factory=dict)
    insertion_curves: Dict[str, List[float]] = field(default_factory=dict)
    deletion_auc: Dict[str, float] = field(default_factory=dict)
    insertion_auc: Dict[str, float] = field(default_factory=dict)
    agreement: Dict[str, Dict[str, float]] = field(default_factory=dict)
    steps: int = 20

    def to_json(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "deletion_curves": self.deletion_curves,
            "insertion_curves": self.insertion_curves,
            "deletion_auc": self.deletion_auc,
            "insertion_auc": self.insertion_auc,
            "agreement": self.agreement,
        }


def to_patch_vector(attr: Any, *, grid: int = 14) -> "Any":
    """Reduce any attribution array to a flat per-patch ``[grid²]`` numpy vector.

    Handles ``[L,T]`` (take last layer, drop CLS), ``[T]`` (drop CLS if
    ``T == grid²+1``), ``[grid,grid]`` / ``[grid²]``, and dense pixel maps
    ``[H,W]`` (average-pool into ``grid×grid``).
    """
    import numpy as np

    a = np.asarray(attr, dtype=np.float64)
    n = grid * grid
    if a.ndim == 2 and a.shape[0] not in (grid,) and a.shape[1] not in (grid,):
        # [L, T] per-layer -> last layer row.
        a = a[-1]
    if a.ndim == 2 and a.shape == (grid, grid):
        return a.reshape(-1).astype(np.float64)
    if a.ndim == 2 and a.shape[0] == a.shape[1] and a.shape[0] > grid:
        # dense pixel map [H,W] -> pool to grid.
        ps = a.shape[0] // grid
        return a[: grid * ps, : grid * ps].reshape(grid, ps, grid, ps).mean(axis=(1, 3)).reshape(-1)
    a = a.reshape(-1)
    if a.shape[0] == n + 1:  # includes CLS at index 0
        a = a[1:]
    if a.shape[0] != n:
        raise ValueError(f"cannot reduce attribution of size {a.shape[0]} to {n} patches")
    return a.astype(np.float64)


def _trapz_unit(curve: List[float]) -> float:
    """Trapezoidal AUC of ``curve`` over an equally-spaced x-grid on ``[0,1]``."""
    n = len(curve)
    if n < 2:
        return float(curve[0]) if curve else 0.0
    total = sum(curve) - 0.5 * (curve[0] + curve[-1])
    return float(total / (n - 1))


def deletion_insertion(
    model: Any,
    image: Any,
    ranking: Any,
    *,
    steps: int = 20,
    baseline: float = 0.0,
    patch_size: int = 16,
    target: Optional[int] = None,
) -> DIResult:
    """Deletion & insertion curves for one patch ``ranking`` (§6).

    ``ranking`` is a per-token/per-patch importance array (``[T]``, ``[196]``,
    or ``[14,14]``); higher = more important. Patches are perturbed most-first
    over ``steps`` chunks. Returns a :class:`DIResult` with curves of length
    ``steps+1`` (endpoints included) and trapezoidal AUCs in ``[0,1]``.
    """
    import numpy as np
    import torch

    from ._common import as_batch, unwrap

    m = unwrap(model)
    x = as_batch(image)
    grid = x.shape[-1] // patch_size
    npatch = grid * grid

    was_training = m.training
    m.eval()
    try:
        with torch.no_grad():
            logits = m(x)
        tgt = int(logits[0].argmax().item()) if target is None else int(target)

        scores = to_patch_vector(ranking, grid=grid)
        order = np.argsort(-scores, kind="stable")  # most-important first

        def _prob(patch_ids, insertion: bool) -> float:
            if insertion:
                xm = torch.full_like(x, float(baseline))
            else:
                xm = x.clone()
            for p in patch_ids:
                row, col = divmod(int(p), grid)
                ys = slice(row * patch_size, (row + 1) * patch_size)
                xs = slice(col * patch_size, (col + 1) * patch_size)
                if insertion:
                    xm[..., ys, xs] = x[..., ys, xs]
                else:
                    xm[..., ys, xs] = float(baseline)
            with torch.no_grad():
                return float(m(xm).softmax(dim=-1)[0, tgt])

        chunks = [c for c in np.array_split(order, steps) if len(c) > 0]
        deletion = [_prob([], insertion=False)]  # full image
        insertion = [_prob([], insertion=True)]  # empty baseline
        removed: List[int] = []
        for ch in chunks:
            removed.extend(int(p) for p in ch)
            deletion.append(_prob(removed, insertion=False))
            insertion.append(_prob(removed, insertion=True))
    finally:
        if was_training:
            m.train()

    del_auc = _trapz_unit(deletion)
    ins_auc = _trapz_unit(insertion)
    return DIResult(
        deletion=[float(v) for v in deletion],
        insertion=[float(v) for v in insertion],
        deletion_auc=del_auc,
        insertion_auc=ins_auc,
        steps=len(chunks),
        target=tgt,
    )


def _rankdata(a: Any) -> Any:
    """Average-rank of a 1-D array (ties share the mean rank)."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # Average tied ranks.
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = (i + j - 1) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(a: Any, b: Any) -> float:
    """Spearman rank correlation = Pearson correlation of average-ranks."""
    import numpy as np

    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0.0:
        return 0.0
    return float((ra * rb).sum() / denom)


def method_agreement(attributions: Dict[str, Any], *, grid: int = 14) -> Dict[str, Dict[str, float]]:
    """Pairwise Spearman rank-correlation matrix over per-patch token rankings.

    ``attributions`` maps method name → attribution array (any shape accepted by
    :func:`to_patch_vector`). Returns a symmetric ``{a: {b: rho}}`` matrix with
    ``1.0`` on the diagonal.
    """
    names = sorted(attributions)
    vecs = {n: to_patch_vector(attributions[n], grid=grid) for n in names}
    out: Dict[str, Dict[str, float]] = {n: {} for n in names}
    for a in names:
        for b in names:
            out[a][b] = 1.0 if a == b else _spearman(vecs[a], vecs[b])
    return out


__all__ = [
    "DIResult",
    "FaithfulnessResult",
    "to_patch_vector",
    "deletion_insertion",
    "method_agreement",
]
