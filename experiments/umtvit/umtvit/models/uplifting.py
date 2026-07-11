"""Spatial uplifting: encoder layers → latent voxel volume (ARCHITECTURE §3.4).

This is the architectural contribution of UMT-ViT. The backbone
(:class:`~umtvit.models.backbone.UMTViTBackbone`) keeps **every** encoder
layer's output — a list of ``L`` token sequences, each ``[B, vg², dim]`` on the
common ``vg × vg`` fusion grid. :class:`SpatialUplifting` gives each layer its
own linear projection ``W_l`` down to ``volume_channels`` and stacks the
per-layer slices along a new depth (Z) axis:

    V(x, y, z=l) = W_l · F_l(x, y),   V ∈ R^{H'×W'×L×C}.

The Z-axis is transformer depth — a *learned hierarchy* of representations, not
physical/anatomical depth (ARCHITECTURE §3.4, honesty rules §1).

Semantics are ported verbatim from the notebook reference's monolithic
``UMTViT`` (its ``self.uplift`` module list and the ``volume.mean(dim=(1,2))``
pooled readout). This module isolates that behaviour so the composed
:class:`~umtvit.models.model.UMTViT` and the losses can consume it directly.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor, nn

from umtvit.config import Config

__all__ = ["SpatialUplifting"]


class SpatialUplifting(nn.Module):
    """Per-layer linear uplift of encoder tokens into an ``H'×W'×L×C`` volume.

    Args:
        config: A validated :class:`~umtvit.config.Config`. The volume grid is
            square (``volume_h == volume_w``, asserted by the backbone), so the
            grid side, layer count ``L = depth``, channel width ``Cv =
            volume_channels`` and token dim are read from ``config.model``.

    Shape:
        - Input: ``layers`` — a list of ``depth`` tensors, each ``[B, vg², dim]``
          (the backbone's ``"layers"`` output, shallow → deep).
        - Output: ``volume`` of shape ``[B, vg, vg, depth, volume_channels]``.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m = config.model
        if m.volume_h != m.volume_w:
            raise ValueError(
                "SpatialUplifting requires a square volume grid "
                f"(volume_h == volume_w); got volume_h={m.volume_h}, "
                f"volume_w={m.volume_w}"
            )
        self.volume_grid = m.volume_h
        self.depth = m.depth
        self.volume_channels = m.volume_channels
        self.dim = m.dim
        # One projection W_l per encoder layer (Z-slice), dim -> Cv.
        self.uplift = nn.ModuleList(
            nn.Linear(m.dim, m.volume_channels) for _ in range(m.depth)
        )

    def forward(self, layers: Sequence[Tensor]) -> Tensor:
        """Uplift and stack the per-layer token grids into the latent volume."""
        if len(layers) != self.depth:
            raise ValueError(
                f"expected {self.depth} layer outputs (one per encoder layer), "
                f"got {len(layers)}"
            )
        vg = self.volume_grid
        slices: List[Tensor] = []
        for lift, tokens in zip(self.uplift, layers):
            b = tokens.shape[0]
            if tokens.shape[1] != vg * vg:
                raise ValueError(
                    f"layer token count {tokens.shape[1]} does not match the "
                    f"volume grid vg²={vg * vg}"
                )
            slices.append(lift(tokens).reshape(b, vg, vg, self.volume_channels))
        # Stack along a new Z axis -> [B, H', W', L, Cv].
        return torch.stack(slices, dim=3)

    @staticmethod
    def pooled(volume: Tensor) -> Tensor:
        """Per-slice spatial-mean readout, flattened to ``[B, L*Cv]``.

        Averages each Z-slice over the ``H'×W'`` grid and flattens the depth and
        channel axes together — the label-free pooled feature the projection
        head and downstream probes consume (matches the notebook's
        ``volume.mean(dim=(1, 2)).flatten(1)``).
        """
        return volume.mean(dim=(1, 2)).flatten(1)
