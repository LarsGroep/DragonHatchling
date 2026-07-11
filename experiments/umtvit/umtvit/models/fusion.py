"""Fine/coarse feature fusion onto a common volume grid (ARCHITECTURE §3.3).

After cross-scale attention the two streams still live on different spatial
grids (``g_fine²`` fine patch tokens, ``g_coarse²`` coarse patch tokens). This
module drops the per-stream CLS tokens, resamples each stream's patch grid onto
the shared ``volume_grid × volume_grid`` fusion grid (bilinear interpolation
when a stream's grid size differs), sums the two, linear-projects the result,
and adds a learned fused positional embedding.

The output is the token sequence that enters the encoder
(:mod:`umtvit.models.encoder`); its grid side ``volume_grid`` becomes the
``H' = W'`` of the latent volume in U3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["FeatureFusion"]


class FeatureFusion(nn.Module):
    """Fuse the fine and coarse patch grids into one token sequence.

    Args:
        dim: Token dimension shared by both streams and the fused output.
        grid_fine: Fine-stream grid side (``image_size / fine_patch``).
        grid_coarse: Coarse-stream grid side (``image_size / coarse_patch``).
        volume_grid: Side of the shared fusion grid ``H' = W'``.

    Shape:
        - Input: ``(tokens_fine, tokens_coarse)`` of shapes
          ``[B, grid_fine² + 1, dim]`` and ``[B, grid_coarse² + 1, dim]``
          (index 0 is the CLS token, dropped here).
        - Output: fused tokens ``[B, volume_grid², dim]``.
    """

    def __init__(
        self, dim: int, grid_fine: int, grid_coarse: int, volume_grid: int
    ) -> None:
        super().__init__()
        self.grid_fine = grid_fine
        self.grid_coarse = grid_coarse
        self.volume_grid = volume_grid
        self.project = nn.Linear(dim, dim)
        self.pos_fused = nn.Parameter(torch.randn(1, volume_grid ** 2, dim) * 0.02)

    def _to_volume_grid(self, patch_tokens: Tensor, grid: int) -> Tensor:
        """Reshape ``[B, grid², D]`` patch tokens to ``[B, D, vg, vg]``.

        Bilinear interpolation resamples the grid onto the fusion grid when the
        sizes differ; otherwise the tokens pass through unchanged.
        """
        b, _, d = patch_tokens.shape
        x = patch_tokens.transpose(1, 2).reshape(b, d, grid, grid)
        if grid != self.volume_grid:
            x = F.interpolate(
                x,
                size=(self.volume_grid, self.volume_grid),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, tokens_fine: Tensor, tokens_coarse: Tensor) -> Tensor:
        """Return fused tokens ``[B, volume_grid², dim]``."""
        # Drop CLS (index 0); resample both patch grids onto the volume grid.
        grid_f = self._to_volume_grid(tokens_fine[:, 1:], self.grid_fine)
        grid_c = self._to_volume_grid(tokens_coarse[:, 1:], self.grid_coarse)
        fused = grid_f + grid_c  # [B, D, vg, vg]
        tokens = fused.flatten(2).transpose(1, 2)  # [B, vg², D]
        return self.project(tokens) + self.pos_fused
