"""Dual-scale patch embedding (ARCHITECTURE §3.1).

Tokenises an image at two spatial scales simultaneously. A *fine* stream
(small patches → many tokens → textures/edges) and a *coarse* stream (large
patches → few tokens → geometry/context) are each produced by a strided
convolution ("conv patchify"), prepended with a per-stream learned CLS token,
and offset by a per-stream learned positional embedding.

The two streams are kept separate on purpose: cross-scale attention
(:mod:`umtvit.models.cross_attention`) exchanges information between them
before they are fused (:mod:`umtvit.models.fusion`).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

__all__ = ["DualScalePatchEmbed"]


class DualScalePatchEmbed(nn.Module):
    """Embed one image into fine and coarse token streams.

    Each stream is ``conv patchify → flatten → prepend CLS → add positional
    embedding``. The fine grid is ``image_size / fine_patch`` per side, the
    coarse grid ``image_size / coarse_patch``; both patch sizes must divide
    ``image_size`` exactly (guaranteed upstream by
    :meth:`umtvit.config.ModelConfig.validate`).

    Args:
        image_size: Side length of the (square) input image, in pixels.
        fine_patch: Fine-stream patch side, in pixels. Must divide
            ``image_size``.
        coarse_patch: Coarse-stream patch side, in pixels. Must divide
            ``image_size``.
        dim: Token (embedding) dimension ``D`` shared by both streams.
        channels: Number of input image channels.

    Shape:
        - Input: ``x`` of shape ``[B, channels, image_size, image_size]``.
        - Output: ``(tokens_fine, tokens_coarse)`` where ``tokens_fine`` is
          ``[B, g_fine² + 1, dim]`` and ``tokens_coarse`` is
          ``[B, g_coarse² + 1, dim]`` (index 0 is the CLS token).
    """

    def __init__(
        self,
        image_size: int,
        fine_patch: int,
        coarse_patch: int,
        dim: int,
        channels: int = 3,
    ) -> None:
        super().__init__()
        if image_size % fine_patch != 0:
            raise ValueError(
                f"fine_patch ({fine_patch}) must divide image_size ({image_size})"
            )
        if image_size % coarse_patch != 0:
            raise ValueError(
                f"coarse_patch ({coarse_patch}) must divide image_size ({image_size})"
            )
        self.dim = dim
        self.grid_fine = image_size // fine_patch
        self.grid_coarse = image_size // coarse_patch
        n_fine = self.grid_fine ** 2
        n_coarse = self.grid_coarse ** 2

        # Conv patchify: a stride-p, kernel-p convolution is a linear patch
        # embedding (one output vector per non-overlapping patch).
        self.embed_fine = nn.Conv2d(channels, dim, fine_patch, stride=fine_patch)
        self.embed_coarse = nn.Conv2d(channels, dim, coarse_patch, stride=coarse_patch)

        self.cls_fine = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_coarse = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_fine = nn.Parameter(torch.randn(1, n_fine + 1, dim) * 0.02)
        self.pos_coarse = nn.Parameter(torch.randn(1, n_coarse + 1, dim) * 0.02)

    def _embed(
        self, x: Tensor, conv: nn.Conv2d, cls: Tensor, pos: Tensor
    ) -> Tensor:
        """Patchify ``x`` with ``conv``, prepend ``cls``, add ``pos``."""
        b = x.shape[0]
        # [B, D, g, g] -> [B, g*g, D]
        tokens = conv(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([cls.expand(b, -1, -1), tokens], dim=1)
        return tokens + pos

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(tokens_fine, tokens_coarse)`` for image batch ``x``."""
        tokens_fine = self._embed(x, self.embed_fine, self.cls_fine, self.pos_fine)
        tokens_coarse = self._embed(
            x, self.embed_coarse, self.cls_coarse, self.pos_coarse
        )
        return tokens_fine, tokens_coarse
