"""Weighted composition of the objective terms (ARCHITECTURE §3.8).

``L = λ₁ L_ntxent + λ₂ L_som + λ₃ L_smooth + λ₄ L_order + λ₅ L_geo`` (plus the
monotone-centroid term), with every λ and each term individually switchable for
ablations. :func:`total_loss` takes the already-computed term tensors and a
weight map and returns the weighted-sum tensor to back-propagate together with a
detached float snapshot of every term (and the total) for history/logging.

The SOM quantization term is *not* recomputed here — it is produced by
:meth:`umtvit.models.som3d.Soft3DSOM.loss` and passed in as one of ``terms``
(reused, never duplicated).
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from torch import Tensor

__all__ = ["total_loss"]


def total_loss(
    terms: Mapping[str, Tensor], weights: Mapping[str, float]
) -> Tuple[Tensor, Dict[str, float]]:
    """Weight and sum the objective terms.

    Args:
        terms: Named loss-term tensors (each a scalar), e.g. ``ntxent``, ``som``,
            ``smooth``, ``order``, ``order_monotone``, ``geodesic``.
        weights: Per-term weights ``λ``. Missing keys default to ``0.0``; a
            weight of ``0`` drops that term from the gradient.

    Returns:
        ``(total, detached)`` where ``total`` is the weighted-sum tensor (carries
        gradient) and ``detached`` maps every term name to its raw (unweighted)
        float value plus ``"total"`` → the weighted-sum float.

    Raises:
        ValueError: if ``terms`` is empty.
    """
    if len(terms) == 0:
        raise ValueError("total_loss requires at least one term")
    total: Tensor | None = None
    detached: Dict[str, float] = {}
    for name, term in terms.items():
        weight = float(weights.get(name, 0.0))
        contribution = weight * term
        total = contribution if total is None else total + contribution
        detached[name] = float(term.detach())
    assert total is not None  # non-empty terms guaranteed above
    detached["total"] = float(total.detach())
    return total, detached
