"""Pre-norm ViT encoder that retains every layer's output (ARCHITECTURE §3.3).

:class:`SelfAttnBlock` is a standard pre-norm transformer block (attention +
MLP with residual connections). :class:`TransformerEncoder` stacks ``depth`` of
them and returns **all** intermediate token sequences, not just the last: the
spatial-uplifting stage (U3) turns each layer output into one Z-slice of the
latent volume, so every layer must be kept.
"""

from __future__ import annotations

from typing import List

from torch import Tensor, nn

__all__ = ["SelfAttnBlock", "TransformerEncoder"]


class SelfAttnBlock(nn.Module):
    """Pre-norm transformer block: ``x + Attn(N(x))`` then ``x + MLP(N(x))``.

    Args:
        dim: Token dimension.
        heads: Number of attention heads (must divide ``dim``).
        mlp_ratio: Hidden width of the MLP as a multiple of ``dim``.

    Shape:
        - Input/Output: ``[B, N, dim]`` (token sequence, unchanged shape).
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply pre-norm self-attention and MLP with residuals."""
        a = self.norm_attn(x)
        x = x + self.attn(a, a, a, need_weights=False)[0]
        return x + self.mlp(self.norm_mlp(x))


class TransformerEncoder(nn.Module):
    """Stack of ``depth`` :class:`SelfAttnBlock`s returning every layer output.

    Args:
        dim: Token dimension.
        depth: Number of encoder layers ``L`` (== latent-volume Z depth).
        heads: Number of attention heads.
        mlp_ratio: MLP hidden-width multiple.

    Shape:
        - Input: ``[B, N, dim]``.
        - Output: ``list`` of exactly ``depth`` tensors, each ``[B, N, dim]``,
          ordered shallow → deep. Element ``l`` is the output *after* the
          ``l``-th block (so the input is not included).
    """

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.depth = depth
        self.layers = nn.ModuleList(
            SelfAttnBlock(dim, heads, mlp_ratio) for _ in range(depth)
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        """Return the output of every layer as a list of ``depth`` tensors."""
        outputs: List[Tensor] = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs
