"""Cross-scale attention between the fine and coarse token streams
(ARCHITECTURE §3.2).

Two modes, selected by config (``model.cross_attention``):

- ``"cls_bridged"`` (CrossViT, the default): each stream's CLS token acts as
  the query over the *other* stream's patch tokens, and the attended result is
  re-injected into that stream's CLS position. Cost is linear in the number of
  tokens — only the two CLS tokens attend across scales.
- ``"full_pair"`` (DSCATNet): *every* token of one stream attends over all
  tokens of the other stream (``Y_f = X_f + softmax(Q_f K_cᵀ/√d) V_c`` and
  symmetrically), so the exchange is dense but quadratic in tokens.

Both modes are pre-norm and use ``batch_first`` :class:`~torch.nn.MultiheadAttention`.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

__all__ = ["CrossScaleBlock", "CROSS_ATTENTION_MODES"]

# Mirror the config enum so a bad mode fails fast at construction rather than
# silently taking the full-pair branch.
CROSS_ATTENTION_MODES = ("cls_bridged", "full_pair")


class CrossScaleBlock(nn.Module):
    """Exchange information between the fine and coarse streams once.

    Args:
        dim: Token dimension shared by both streams.
        heads: Number of attention heads (must divide ``dim``).
        mode: ``"cls_bridged"`` or ``"full_pair"`` (see module docstring).

    Shape:
        - Input: ``(tokens_fine, tokens_coarse)`` of shapes
          ``[B, N_f + 1, dim]`` and ``[B, N_c + 1, dim]`` (index 0 = CLS).
        - Output: same shapes, information mixed across scales.
    """

    def __init__(self, dim: int, heads: int, mode: str) -> None:
        super().__init__()
        if mode not in CROSS_ATTENTION_MODES:
            raise ValueError(
                f"unknown cross-attention mode {mode!r}; "
                f"must be one of {list(CROSS_ATTENTION_MODES)}"
            )
        self.mode = mode
        self.norm_fine = nn.LayerNorm(dim)
        self.norm_coarse = nn.LayerNorm(dim)
        # ``fine_from_coarse``: query = fine, key/value = coarse (and vice
        # versa). Naming reflects where the *information* comes from.
        self.fine_from_coarse = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.coarse_from_fine = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(
        self, tokens_fine: Tensor, tokens_coarse: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Return cross-attended ``(tokens_fine, tokens_coarse)``."""
        norm_f = self.norm_fine(tokens_fine)
        norm_c = self.norm_coarse(tokens_coarse)

        if self.mode == "cls_bridged":
            # Each stream's CLS (query) attends over the other's patch tokens.
            cls_f = tokens_fine[:, :1] + self.fine_from_coarse(
                norm_f[:, :1], norm_c[:, 1:], norm_c[:, 1:], need_weights=False
            )[0]
            cls_c = tokens_coarse[:, :1] + self.coarse_from_fine(
                norm_c[:, :1], norm_f[:, 1:], norm_f[:, 1:], need_weights=False
            )[0]
            out_f = torch.cat([cls_f, tokens_fine[:, 1:]], dim=1)
            out_c = torch.cat([cls_c, tokens_coarse[:, 1:]], dim=1)
            return out_f, out_c

        # full_pair: all tokens of one stream attend over all tokens of the other.
        out_f = tokens_fine + self.fine_from_coarse(
            norm_f, norm_c, norm_c, need_weights=False
        )[0]
        out_c = tokens_coarse + self.coarse_from_fine(
            norm_c, norm_f, norm_f, need_weights=False
        )[0]
        return out_f, out_c
