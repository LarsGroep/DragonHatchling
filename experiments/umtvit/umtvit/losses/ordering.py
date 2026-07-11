"""Layer-scale ordering regulariser + monotone-centroid companion (ARCHITECTURE §3.7).

ViT depth does not order itself by spatial scale (Raghu et al.), so UMT-ViT
*imposes* a bias on the Z-axis of the latent volume:

- :func:`ordering_loss` penalises each slice's spatial power above a
  depth-decreasing cutoff ``f(l) = f_max · (1 − l/(L−1))`` — shallow slices may
  carry high-frequency detail, deep slices are pushed toward smooth, coarse
  structure (``L_order = Σ_l ‖HighPass_{f(l)}(V[:,:,l,:])‖²``).
- :func:`monotone_centroid_loss` acts directly on each slice's spectral centroid
  (the power-weighted mean spatial frequency) and penalises any *rise* with
  depth, ``Σ_l relu(c[l+1] − c[l])`` — a strictly monotone-non-increasing
  constraint the ordering cutoff only bounds from above.

Both reuse the shared FFT/radius machinery in :mod:`umtvit.losses._common` so the
two share one notion of spatial frequency. Whether genuine scale ordering
emerges is measured (U5 probes), never assumed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from umtvit.losses._common import radius_grid, slice_power_spectrum

__all__ = ["ordering_loss", "monotone_centroid_loss"]


def ordering_loss(volume: Tensor, fmax: float) -> Tensor:
    """Penalise per-slice spatial power above a depth-decreasing cutoff.

    Args:
        volume: Latent voxel volume ``[B, H', W', L, C]``.
        fmax: Maximum spatial-frequency cutoff (cycles/pixel) at the shallowest
            slice; slice ``l``'s cutoff is ``fmax · (1 − l/(L−1))``.

    Returns:
        Scalar: the depth-averaged high-pass power. Shallow slices are lightly
        penalised (high cutoff), deep slices heavily (low cutoff), pushing coarse
        structure to depth.

    Shape:
        - Input: ``[B, H', W', L, C]``.
        - Output: scalar.
    """
    _, height, width, depth, _ = volume.shape
    spec = slice_power_spectrum(volume)  # [B, L, C, H', Wf]
    r = radius_grid(height, width, volume.device)  # [H', Wf]
    total = volume.new_zeros(())
    for l in range(depth):
        cutoff = fmax * (1 - l / max(depth - 1, 1))
        mask = (r > cutoff).float()
        total = total + (spec[:, l] * mask).mean()
    return total / depth


def monotone_centroid_loss(volume: Tensor) -> Tensor:
    """Penalise any rise, with depth, of the per-slice spectral centroid.

    The spectral centroid of slice ``l`` is the power-weighted mean spatial
    frequency ``c[l] = Σ (power · radius) / Σ power``. The loss is
    ``Σ_l relu(c[l+1] − c[l])`` — zero iff the centroid is monotone
    non-increasing with depth (coarser-with-depth), positive for any inversion.

    Args:
        volume: Latent voxel volume ``[B, H', W', L, C]``.

    Returns:
        Scalar ``≥ 0``; exactly ``0`` when every deeper slice is at least as
        coarse (centroid ≤) as the slice above it.

    Shape:
        - Input: ``[B, H', W', L, C]``.
        - Output: scalar.
    """
    _, height, width, depth, _ = volume.shape
    spec = slice_power_spectrum(volume)  # [B, L, C, H', Wf]
    r = radius_grid(height, width, volume.device)  # [H', Wf]
    centroids = [
        (spec[:, l] * r).sum() / (spec[:, l].sum() + 1e-8) for l in range(depth)
    ]
    total = volume.new_zeros(())
    for l in range(depth - 1):
        total = total + F.relu(centroids[l + 1] - centroids[l])
    return total
