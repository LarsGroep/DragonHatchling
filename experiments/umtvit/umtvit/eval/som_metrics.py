"""SOM quality metrics on held-out voxels (ARCHITECTURE §6.2).

A thin evaluation wrapper around :meth:`umtvit.models.som3d.Soft3DSOM.metrics`:
run the frozen encoder over an eval split, collect the latent-volume voxels, and
score the trained SOM on a **seeded** subsample of them — quantization error,
topographic error, and dead-neuron fraction. Mirrors the notebook's held-out
``test_voxels`` cell so the package and notebook report the same numbers.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from umtvit.eval.features import _model_device
from umtvit.models.som3d import Soft3DSOM

__all__ = ["som_metrics"]

_NAN_METRICS: Dict[str, float] = {
    "quantization_error": float("nan"),
    "topographic_error": float("nan"),
    "dead_neuron_fraction": float("nan"),
}


@torch.no_grad()
def som_metrics(
    model: torch.nn.Module,
    som: Soft3DSOM,
    dataset,
    *,
    max_imgs: int = 64,
    sample_voxels: int = 2048,
    seed: int = 0,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Score the trained SOM on a seeded voxel subsample of an eval split.

    Args:
        model: Encoder whose ``forward(x)["volume"]`` is ``[B,H',W',L,Cv]``.
        som: The trained 3-D SOM to score.
        dataset: An eval-mode dataset yielding ``(image, label)``.
        max_imgs: Cap on images whose voxels are collected.
        sample_voxels: Cap on voxels scored (seeded random subsample).
        seed: Seed for the voxel subsample (reproducible metrics).
        batch_size: Forward-pass batch size.
        device: Where to run; defaults to the model's device.

    Returns:
        ``{"quantization_error", "topographic_error", "dead_neuron_fraction"}``;
        all-``nan`` when the split yields no voxels.
    """
    device = device if device is not None else _model_device(model)
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    vox, seen = [], 0
    for x, _ in loader:
        volume = model(x.to(device))["volume"]
        vox.append(volume.reshape(-1, volume.shape[-1]).detach().to("cpu").float())
        seen += x.shape[0]
        if seen >= max_imgs:
            break

    if was_training:
        model.train()

    if not vox:
        return dict(_NAN_METRICS)
    voxels = torch.cat(vox)
    if voxels.shape[0] == 0:
        return dict(_NAN_METRICS)

    if voxels.shape[0] > sample_voxels:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(voxels.shape[0], generator=gen)[:sample_voxels]
        voxels = voxels[idx]
    return som.metrics(voxels.to(device))
