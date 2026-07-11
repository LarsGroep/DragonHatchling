"""Contrastive projection head (ARCHITECTURE §3.6).

The pooled latent volume ``[B, L*Cv]`` (per-slice spatial means, see
:meth:`umtvit.models.uplifting.SpatialUplifting.pooled`) is mapped by a 2-layer
MLP to the contrastive embedding ``z ∈ R^{proj_dim}`` on which NT-Xent operates.
Ported from the notebook reference's ``self.head``: ``Linear(L*Cv → dim) →
GELU → Linear(dim → proj_dim)`` (the token dim is the hidden width).
"""

from __future__ import annotations

from torch import Tensor, nn

from umtvit.config import Config

__all__ = ["ProjectionHead"]


class ProjectionHead(nn.Module):
    """2-layer MLP from the pooled volume to the contrastive embedding.

    Args:
        config: A validated :class:`~umtvit.config.Config`. The input width is
            ``depth * volume_channels`` (the pooled volume), the hidden width is
            ``model.dim`` and the output is ``model.proj_dim``.

    Shape:
        - Input: ``pooled`` of shape ``[B, depth * volume_channels]``.
        - Output: ``[B, proj_dim]``.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m = config.model
        in_dim = m.depth * m.volume_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, m.dim),
            nn.GELU(),
            nn.Linear(m.dim, m.proj_dim),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        return self.mlp(pooled)
