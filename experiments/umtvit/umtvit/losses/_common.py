"""Shared FFT / radius helpers for the spectral losses (ARCHITECTURE §3.7).

Both the layer-scale ordering regulariser (:mod:`umtvit.losses.ordering`) and
its monotone-centroid companion read the per-slice spatial power spectrum of the
latent volume and weight it by an isotropic spatial-frequency radius. Those two
pieces of machinery are factored here so the two losses share one definition of
"spatial frequency" (in cycles/pixel) and one spectrum layout — the volume is
``[B, H', W', L, C]`` and the spectrum is taken over the ``H'×W'`` plane per
``(batch, layer, channel)``.

Torch-only, no learnable state.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["radius_grid", "slice_power_spectrum"]


def radius_grid(height: int, width: int, device: torch.device) -> Tensor:
    """Isotropic spatial-frequency radius of an ``rfft2`` grid, in cycles/pixel.

    Mirrors the notebook reference's ``_radius_grid``: the vertical axis uses the
    full (absolute) ``fftfreq`` and the horizontal axis the half-spectrum
    ``rfftfreq`` (as produced by :func:`torch.fft.rfft2`), combined as
    ``sqrt(fy² + fx²)``.

    Args:
        height: Spatial height ``H'`` of the volume slice.
        width: Spatial width ``W'`` of the volume slice.
        device: Device the returned grid is placed on.

    Shape:
        - Output: ``[H', W'//2 + 1]`` of non-negative frequencies (cycles/px).
    """
    fy = torch.fft.fftfreq(height, device=device).abs()
    fx = torch.fft.rfftfreq(width, device=device)
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def slice_power_spectrum(volume: Tensor) -> Tensor:
    """Per-slice spatial power spectrum of the latent volume.

    Takes the orthonormal 2-D real FFT over the ``H'×W'`` plane of every
    ``(batch, layer, channel)`` and returns squared magnitudes (power). Matches
    the notebook's ``rfft2(V.permute(0, 3, 4, 1, 2), norm="ortho").abs()**2``.

    Args:
        volume: Latent voxel volume ``[B, H', W', L, C]``.

    Shape:
        - Output: ``[B, L, C, H', W'//2 + 1]`` real, non-negative power.
    """
    permuted = volume.permute(0, 3, 4, 1, 2)  # [B, L, C, H', W']
    return torch.fft.rfft2(permuted, norm="ortho").abs() ** 2
