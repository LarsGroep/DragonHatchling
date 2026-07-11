"""NT-Xent contrastive loss over paired views (ARCHITECTURE §3.6).

SimCLR's normalised-temperature cross-entropy: given the two views' projected
embeddings ``z_a``/``z_b`` (each ``[B, proj_dim]``), the ``2B`` embeddings are
L2-normalised and every embedding is trained to identify its counterpart view of
the same image as the positive among all other ``2B − 1`` embeddings.

Ported verbatim from the notebook reference's ``nt_xent`` (same normalisation,
same in-batch negative construction, same target index scheme).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["nt_xent"]


def nt_xent(za: Tensor, zb: Tensor, tau: float) -> Tensor:
    """NT-Xent loss for a batch of paired views.

    Args:
        za: View-A embeddings ``[B, proj_dim]``.
        zb: View-B embeddings ``[B, proj_dim]`` (``za[i]``/``zb[i]`` are the two
            views of image ``i``).
        tau: Softmax temperature ``τ`` (``> 0``); lower sharpens the contrast.

    Returns:
        Scalar cross-entropy loss where each of the ``2B`` embeddings must pick
        its paired view as the positive.

    Shape:
        - Input: two ``[B, proj_dim]`` tensors.
        - Output: scalar.
    """
    z = F.normalize(torch.cat([za, zb]), dim=1)
    sim = z @ z.T / tau
    n = za.shape[0]
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)
