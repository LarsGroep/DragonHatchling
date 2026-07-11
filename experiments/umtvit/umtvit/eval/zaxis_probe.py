"""Z-axis probes: per-slice spatial-frequency analysis (ARCHITECTURE §3.7, §6.4).

The measured answer to the experiment's central question — *did scale ordering
emerge along the learned-hierarchy Z-axis?* For each depth slice ``z`` (encoder
layer) of the latent volume we compute the spectral centroid of its 2-D power
spectrum: a centroid that **falls with depth** means shallow slices carry fine
detail and deep slices coarse structure, i.e. the imposed ordering bias took
effect.

Two centroid variants (post-N2 notebook feedback, issue 2):

- ``per_channel`` — the **fair** measure: take each channel's centroid, then
  average. A channel-mean image can cancel per-channel high-frequency structure
  and understate the centroid, so this drives the verdict.
- ``channel_mean`` — the older channel-mean-then-centroid measure, kept
  alongside once for backward comparability with runs 1–2.

The Z-axis is a *learned representational hierarchy*, not physical depth
(honesty rules §1). Whether ordering emerged is reported, never assumed:
:func:`monotonicity_verdict` states the honest outcome either way.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from umtvit.eval.features import _model_device

__all__ = [
    "radial_spectrum",
    "spectral_centroid",
    "extract_probe_volumes",
    "zaxis_probe",
    "monotonicity_verdict",
]

# Rise tolerance for the monotone-decreasing verdict: a per-step increase below
# this counts as flat, matching the notebook's `np.diff(centroids) <= 1e-4`.
_MONOTONE_TOL = 1e-4


def radial_spectrum(img2d: np.ndarray) -> tuple:
    """Radially-binned power spectrum of a 2-D image (mean-subtracted, ortho FFT).

    Returns ``(bin_centers, power_per_bin)`` — 8 radial frequency bins in
    cycles/pixel. Mirrors the notebook's ``radial_spectrum``.
    """
    img2d = np.asarray(img2d, dtype=np.float32)
    f = np.abs(np.fft.rfft2(img2d - img2d.mean(), norm="ortho")) ** 2
    fy = np.abs(np.fft.fftfreq(img2d.shape[0]))
    fx = np.fft.rfftfreq(img2d.shape[1])
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bins = np.linspace(0, r.max() + 1e-9, 9)
    idx = np.digitize(r.ravel(), bins) - 1
    power = np.bincount(idx, weights=f.ravel(), minlength=len(bins)) / (
        np.bincount(idx, minlength=len(bins)) + 1e-9
    )
    centers = (bins[:-1] + bins[1:]) / 2
    return centers, power[: len(centers)]


def spectral_centroid(img2d: np.ndarray) -> float:
    """Frequency centroid (mean cycles/px weighted by power) of a 2-D image."""
    centers, power = radial_spectrum(img2d)
    return float((centers * power).sum() / (power.sum() + 1e-9))


def _slice_channel_mean(vol: np.ndarray, z: int) -> np.ndarray:
    """Channel-mean of depth slice ``z``, per-slice min-max normalised (notebook)."""
    s = vol[:, :, z, :].astype(np.float32).mean(-1)
    return (s - s.min()) / (np.ptp(s) + 1e-8)


@torch.no_grad()
def extract_probe_volumes(
    model: torch.nn.Module,
    dataset,
    *,
    n_images: int = 8,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run the encoder over a seeded image subset; return ``[N,H',W',L,Cv]`` volumes.

    A seeded random subset (not the first N: a dataset's ordering can make the
    leading rows near single-class, per N2 feedback) of at most ``n_images``.
    """
    device = device if device is not None else _model_device(model)
    was_training = model.training
    model.eval()

    n = min(n_images, len(dataset))
    if n <= 0:
        return np.zeros((0, 0, 0, 0, 0), dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = sorted(rng.permutation(len(dataset))[:n].tolist())
    imgs = torch.stack([dataset[i][0] for i in idx]).to(device)
    volume = model(imgs)["volume"].detach().to("cpu").float().numpy()

    if was_training:
        model.train()
    return volume


def zaxis_probe(volumes: np.ndarray) -> Dict[str, object]:
    """Per-slice spectral centroids (fair + channel-mean) + monotonicity verdict.

    Args:
        volumes: ``[N, H', W', L, Cv]`` latent volumes over probe images.

    Returns:
        A dict with ``"per_channel_centroids"`` and ``"channel_mean_centroids"``
        (both length-``L`` lists, the fair measure first), plus the verdict keys
        from :func:`monotonicity_verdict` computed on the fair measure.
    """
    volumes = np.asarray(volumes, dtype=np.float32)
    if volumes.ndim != 5 or volumes.shape[0] == 0:
        empty: Dict[str, object] = {
            "per_channel_centroids": [],
            "channel_mean_centroids": [],
        }
        empty.update(monotonicity_verdict([]))
        return empty

    n_img, _, _, depth, channels = volumes.shape
    per_channel: List[float] = []
    channel_mean: List[float] = []
    for z in range(depth):
        # Fair: centroid of every channel of every image, then averaged.
        per_ch = [
            spectral_centroid(volumes[i][:, :, z, c])
            for i in range(n_img)
            for c in range(channels)
        ]
        per_channel.append(float(np.mean(per_ch)))
        # Backward-comparable: channel-mean image centroid, averaged over images.
        cm = [spectral_centroid(_slice_channel_mean(volumes[i], z)) for i in range(n_img)]
        channel_mean.append(float(np.mean(cm)))

    result: Dict[str, object] = {
        "per_channel_centroids": per_channel,
        "channel_mean_centroids": channel_mean,
    }
    result.update(monotonicity_verdict(per_channel))
    return result


def monotonicity_verdict(centroids: Sequence[float]) -> Dict[str, object]:
    """Honest monotone-decreasing verdict for a per-depth centroid sequence.

    Returns ``{"monotone_decreasing": bool, "verdict": str}``. A sequence is
    monotone-decreasing if no step rises by more than ``1e-4`` (notebook rule).
    Fewer than two finite centroids is undetermined.
    """
    finite = [float(c) for c in centroids if np.isfinite(c)]
    if len(finite) < 2:
        return {
            "monotone_decreasing": False,
            "verdict": "undetermined (fewer than two depth slices)",
        }
    monotone = bool(np.all(np.diff(finite) <= _MONOTONE_TOL))
    verdict = (
        "monotone decreasing with depth — ordering emerged"
        if monotone
        else "non-monotone — ordering only partially emerged (honest negative result)"
    )
    return {"monotone_decreasing": monotone, "verdict": verdict}
