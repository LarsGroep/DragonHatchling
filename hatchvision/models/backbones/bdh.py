"""Experimental Baby Dragon Hatchling (BDH) vision backbone.

BDH ("The Dragon Hatchling", Kosowski et al., arXiv:2509.26507) is a
brain-inspired sequence architecture whose defining traits are:

* a very **high-dimensional neuron space** ``n >> d`` reached through a
  low-rank encoder, with **ReLU-induced positive, sparse activations**
  (interpretable "neurons firing"),
* **linear attention** computed with positive (ReLU) query/key kernels,
* an optionally **weight-shared ("universal") layer** iterated over depth.

This module adapts BDH-GPU to images: the image is patch-embedded into a
token sequence and non-causal BDH blocks are iterated over the patches.
The sparse neuron activations are exposed as a dedicated ``neurons``
submodule per block, which makes this backbone a particularly good citizen
for the Hebbian feature memory — co-activation statistics over genuinely
sparse, positive units are exactly what Hebbian analysis wants.

If the official ``bdh`` package (pathwaycom/bdh) is installed it is detected
and reported via :data:`OFFICIAL_BDH_AVAILABLE`; the official release targets
language modelling, so the vision path always uses this adaptation.

This backbone is **experimental**: it trains, but no claims are made about
matching CNN baselines.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from hatchvision.models.backbones.base import Backbone, register_backbone

try:  # pragma: no cover - depends on environment
    import bdh as _official_bdh  # noqa: F401

    OFFICIAL_BDH_AVAILABLE = True
except ImportError:
    OFFICIAL_BDH_AVAILABLE = False


class _LinearAttention(nn.Module):
    """Non-causal linear attention with positive feature maps (ReLU kernel)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x, v: [B, T, d]
        q = torch.relu(self.q(x))
        k = torch.relu(self.k(x))
        kv = torch.einsum("btd,bte->bde", k, v)          # [B, d, d]
        z = 1.0 / (torch.einsum("btd,bd->bt", q, k.sum(dim=1)) + 1e-6)
        return torch.einsum("btd,bde,bt->bte", q, kv, z)  # [B, T, d]


class BDHBlock(nn.Module):
    """One BDH iteration: sparse neuron lift, attention, synaptic readback."""

    def __init__(self, dim: int, neuron_dim: int) -> None:
        super().__init__()
        # The interpretable unit: positive, sparse activations in R^n.
        self.neurons = nn.Sequential(
            nn.Linear(dim, neuron_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self.readback = nn.Linear(neuron_dim, dim, bias=False)
        self.attn = _LinearAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lift to the sparse neuron space, read back to d, attend over patches.
        y = self.neurons(self.norm1(x))          # [B, T, n]  sparse & positive
        v = self.readback(y)                     # [B, T, d]
        x = x + self.attn(self.norm1(x), v)
        # Second synaptic pass (position-wise integration).
        y2 = self.neurons(self.norm2(x))
        x = x + self.readback(y2)
        return x


@register_backbone("bdh")
class BDHVisionBackbone(Backbone):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 128,
        neuron_dim: int = 512,
        depth: int = 4,
        share_weights: bool = True,
        **_,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        n_tokens = (image_size // patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.depth = depth
        if share_weights:
            # BDH's "universal" layer: one block iterated `depth` times.
            block = BDHBlock(dim, neuron_dim)
            self.blocks = nn.ModuleList([block] * depth)
        else:
            self.blocks = nn.ModuleList(BDHBlock(dim, neuron_dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self._dim = dim
        self._grid = image_size // patch_size

    @property
    def feature_dim(self) -> int:
        return self._dim

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, T, d]
        t = t + self.pos
        for block in self.blocks:
            t = block(t)
        return self.norm(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokens(x).mean(dim=1)

    def cam_layer(self):
        # Patch embedding is the last conv with spatial layout; Grad-CAM on it
        # yields a coarse (grid x grid) saliency map.
        return self.patch_embed

    def hebbian_layers(self) -> Dict[str, nn.Module]:
        # Observe the sparse neuron space of the final block (deduplicated when
        # weights are shared, since it is the same module object).
        seen, out = set(), {}
        for i, block in enumerate(self.blocks):
            if id(block.neurons) not in seen:
                seen.add(id(block.neurons))
                out[f"block{i}.neurons"] = block.neurons
        return out
