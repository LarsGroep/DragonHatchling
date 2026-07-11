"""Manifold-quality metrics: trustworthiness & continuity (ARCHITECTURE §6.3).

Two dual measures of how well the pooled feature space preserves the local
neighbour structure of the raw input pixels (Venna & Kaski):

- **Trustworthiness** penalises points that are neighbours in *feature* space
  but were *not* neighbours in *input* space (false neighbours introduced by the
  embedding).
- **Continuity** penalises points that were neighbours in *input* space but are
  *not* neighbours in *feature* space (true neighbours the embedding tore apart).

Both are the same functional with the high/low spaces swapped, so a single
helper computes each. Pure numpy on a **seeded, capped** subsample (default
``n ≤ 400``) — the pairwise-rank computation is O(n²·k). The trustworthiness
branch matches the notebook reference exactly; continuity is its dual. Both lie
in ``[0, 1]`` (1 = perfect local-structure preservation).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["trustworthiness_continuity"]


def _pairwise_dist(a: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix (numerically-floored gram trick)."""
    g = (a * a).sum(1)
    d2 = g[:, None] + g[None, :] - 2.0 * (a @ a.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _quality(rank_dist: np.ndarray, nbr_dist: np.ndarray, k: int) -> float:
    """One side of the trust/continuity pair (Venna & Kaski).

    Points that are among the ``k`` nearest under ``nbr_dist`` but rank beyond
    ``k`` under ``rank_dist`` are penalised by how far past ``k`` they rank.
    Trustworthiness uses ``rank_dist`` = input, ``nbr_dist`` = feature;
    continuity swaps them.
    """
    n = rank_dist.shape[0]
    k = int(min(k, (n - 1) // 2))
    if n < 4 or k < 1:
        return float("nan")
    ind_rank = np.argsort(rank_dist, axis=1)
    rank_of = np.argsort(ind_rank, axis=1)  # rank of j from i (0 = self)
    nn_nbr = np.argsort(nbr_dist, axis=1)[:, 1 : k + 1]  # k nearest in nbr space
    penalty = 0.0
    for i in range(n):
        for j in nn_nbr[i]:
            r = rank_of[i, j]
            if r > k:
                penalty += r - k
    norm = n * k * (2 * n - 3 * k - 1)
    return float(1.0 - (2.0 / norm) * penalty) if norm > 0 else float("nan")


def trustworthiness_continuity(
    pixels,
    features,
    *,
    k: int = 7,
    max_n: int = 400,
    seed: int = 0,
) -> Dict[str, float]:
    """Trustworthiness and continuity between input pixels and pooled features.

    Args:
        pixels: ``[N, P]`` high-dimensional input space (flattened pixels).
        features: ``[N, D]`` low-dimensional feature space (pooled features).
        k: Neighbourhood size (default 7).
        max_n: Cap on samples used (seeded subsample if ``N`` exceeds it).
        seed: Seed for the subsample.

    Returns:
        ``{"trustworthiness", "continuity", "k", "n"}``. The two scores are
        ``nan`` when there are too few samples to define a ``k``-neighbourhood.
    """
    high = np.asarray(pixels, dtype=np.float32)
    low = np.asarray(features, dtype=np.float32)
    n = high.shape[0]
    if n > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.permutation(n)[:max_n])
        high, low = high[idx], low[idx]
        n = high.shape[0]

    if n < 4:
        return {"trustworthiness": float("nan"), "continuity": float("nan"), "k": k, "n": n}

    d_high = _pairwise_dist(high)
    d_low = _pairwise_dist(low)
    trust = _quality(d_high, d_low, k)  # false neighbours in feature space
    cont = _quality(d_low, d_high, k)   # torn-apart neighbours from input space
    return {"trustworthiness": trust, "continuity": cont, "k": k, "n": n}
