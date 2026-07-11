"""Differentiable 3-D Self-Organizing Map over voxel features (ARCHITECTURE §3.5).

A topology-preserving map whose neurons ``w_k`` live on a 3-D grid ``G``
(default ``8×8×8``). For each voxel feature ``v_i`` the map computes a
neighbourhood-weighted soft quantization loss

    L_som = Σ_i  ( Σ_k h(k, BMU(i)) · ‖v_i − w_k‖² ) / Σ_k h(k, BMU(i)),

with the DESOM neighbourhood ``h(k, k*) = exp(−d_G(k,k*)² / 2σ²)`` and σ
annealed exponentially over training. Two update modes (ARCHITECTURE §3.5):

- ``"gradient"`` (DESOM-style, default): ``w`` are learnable parameters, moved
  by backprop through ``L_som`` alongside the encoder.
- ``"kohonen_ema"``: the classical Kohonen/Hebbian EMA update is applied to
  ``w`` **outside** the autograd graph; the returned loss is the plain
  BMU quantization error (a scalar the encoder still receives gradient from),
  kept for the biological-variant ablation.

Ported from the notebook reference's ``Soft3DSOM`` (proven semantics): the same
``grid_d2`` neighbourhood buffer, soft/EMA loss, ``bmu``, ``data_init``,
``revive``, and ``metrics`` (quantization error, topographic error,
dead-neuron fraction). :func:`resolve_sigma` mirrors the notebook's
grid-derived σ defaults for nullable ``loss.sigma_start``/``sigma_end``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from umtvit.config import Config

__all__ = ["Soft3DSOM", "resolve_sigma"]

# Grid-distance² threshold below which two neurons count as topological
# neighbours: a squared Euclidean grid distance of ≤ 3 covers the 26-connected
# 3-D neighbourhood (axis 1, face-diagonal 2, corner-diagonal 3). The second-
# BMU landing beyond this ring is a topographic error.
_ADJACENT_D2 = 3.0


def resolve_sigma(
    sigma_start: Optional[float],
    sigma_end: Optional[float],
    som_grid: Sequence[int],
) -> Tuple[float, float]:
    """Resolve the nullable σ schedule against the SOM grid.

    ``None`` defers to the notebook's grid-derived defaults: a start of
    ``max(som_grid)/2`` (keeps the initial neighbourhood inside the lattice — a
    σ wider than the grid collapses every neuron to the global mean and the map
    never differentiates) and an end of ``0.75``. Explicit values pass through
    unchanged.
    """
    start = max(som_grid) / 2.0 if sigma_start is None else float(sigma_start)
    end = 0.75 if sigma_end is None else float(sigma_end)
    return start, end


class Soft3DSOM(nn.Module):
    """Topology-preserving 3-D SOM over voxel features, trainable end-to-end.

    Args:
        grid: 3-tuple neuron lattice shape ``(Gz, Gy, Gx)``.
        feat_dim: voxel/neuron feature width (``volume_channels``).
        tau: soft-assignment temperature (kept for API parity with the
            notebook; the neighbourhood-weighted loss below is σ-driven).
        update: ``"gradient"`` (learnable ``w``) or ``"kohonen_ema"`` (Hebbian
            EMA update outside autograd).
        ema_lr: EMA step size for the ``"kohonen_ema"`` update.
    """

    def __init__(
        self,
        grid: Sequence[int],
        feat_dim: int,
        tau: float,
        update: str = "gradient",
        ema_lr: float = 0.5,
    ) -> None:
        super().__init__()
        self.grid: Tuple[int, ...] = tuple(int(g) for g in grid)
        self.K = int(np.prod(self.grid))
        self.tau = float(tau)
        self.update = update
        self.ema_lr = float(ema_lr)
        # gradient mode learns w by backprop; kohonen_ema moves w manually.
        w = torch.randn(self.K, feat_dim) * 0.5
        self.weights = nn.Parameter(w, requires_grad=(update == "gradient"))
        zz, yy, xx = torch.meshgrid(
            *(torch.arange(g).float() for g in self.grid), indexing="ij"
        )
        coords = torch.stack([zz, yy, xx], -1).reshape(self.K, 3)
        # Pairwise squared grid distances drive the neighbourhood kernel.
        self.register_buffer("grid_d2", torch.cdist(coords, coords) ** 2)
        self.register_buffer("coords", coords)

    @classmethod
    def from_config(cls, config: Config) -> "Soft3DSOM":
        """Build a SOM from a validated :class:`~umtvit.config.Config`."""
        m, ls = config.model, config.loss
        return cls(
            grid=m.som_grid,
            feat_dim=m.volume_channels,
            tau=ls.som_temperature,
            update=m.som_update,
        )

    def bmu(self, v: Tensor) -> Tensor:
        """Best-matching-unit index per voxel (hard argmin; eval/bookkeeping)."""
        return torch.cdist(v, self.weights.detach()).argmin(1)

    @torch.no_grad()
    def data_init(self, pool: Tensor, noise: float = 0.05) -> None:
        """Seed every neuron from a random voxel of ``pool`` (+ scaled noise).

        Data-driven init (``model.som_init == "data"``) breaks the
        collapse-to-mean failure of random init under a wide neighbourhood σ.
        A no-op on an empty pool.
        """
        if pool.shape[0] == 0:
            return
        idx = torch.randint(0, pool.shape[0], (self.K,), device=pool.device)
        src = pool[idx]
        self.weights.data.copy_(
            src + torch.randn_like(src) * noise * (pool.std() + 1e-8)
        )

    @torch.no_grad()
    def revive(self, hit_counts: Tensor, pool: Tensor, noise: float = 0.1) -> int:
        """Re-seed neurons that won zero BMU assignments from recent voxels.

        Dead neurons (``hit_counts == 0``) are copied from random voxels of
        ``pool`` (+ noise). Returns the number of neurons revived.
        """
        if pool.shape[0] == 0:
            return 0
        dead = torch.nonzero(hit_counts == 0, as_tuple=False).flatten()
        if dead.numel() == 0:
            return 0
        src = pool[torch.randint(0, pool.shape[0], (dead.numel(),), device=pool.device)]
        self.weights.data[dead] = (
            src + torch.randn_like(src) * noise * (pool.std() + 1e-8)
        )
        return int(dead.numel())

    def loss(self, v: Tensor, sigma: float) -> Tensor:
        """Neighbourhood-weighted quantization loss (+ EMA update if Hebbian).

        In ``"kohonen_ema"`` mode the classical Kohonen update is applied to
        ``w`` in ``no_grad`` and the returned scalar is the plain BMU
        quantization error; in ``"gradient"`` mode the soft neighbourhood loss
        drives both ``w`` and the encoder by backprop.
        """
        d2 = torch.cdist(v, self.weights) ** 2                     # [M, K]
        bmu = d2.detach().argmin(1)
        h = torch.exp(-self.grid_d2[bmu] / (2 * sigma**2))         # [M, K]
        if self.update == "kohonen_ema":
            with torch.no_grad():                                  # Hebbian step
                num = h.T @ v.detach()
                den = h.sum(0).unsqueeze(1) + 1e-8
                self.weights.lerp_(num / den, self.ema_lr)
            return (v - self.weights[bmu].detach()).pow(2).sum(1).mean()
        return (h * d2).sum(1).div(h.sum(1) + 1e-8).mean()

    @torch.no_grad()
    def metrics(self, v: Tensor) -> Dict[str, float]:
        """Quantization error · topographic error · dead-neuron fraction.

        - quantization error: mean voxel→nearest-neuron distance.
        - topographic error: fraction of voxels whose first two BMUs are not
          grid neighbours (grid distance² > 3).
        - dead-neuron fraction: share of neurons that are nobody's nearest.
        """
        d = torch.cdist(v, self.weights)
        near = d.topk(2, largest=False).indices
        qe = d.gather(1, near[:, :1]).mean().item()
        te = (self.grid_d2[near[:, 0], near[:, 1]] > _ADJACENT_D2 + 1e-6).float().mean().item()
        dead = 1.0 - torch.unique(near[:, 0]).numel() / self.K
        return {
            "quantization_error": qe,
            "topographic_error": te,
            "dead_neuron_fraction": dead,
        }
