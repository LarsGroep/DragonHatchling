"""Total-variation smoothness over the volume neighbour graph (ARCHITECTURE §3.6).

Penalises squared differences between neighbouring voxels along the selected
volume axes — the discrete total variation ``L_smooth = Σ_{(i,j)∈E} ‖v_i − v_j‖²``
of §3.6, normalised per edge (mean over each finite-difference tensor, then
averaged over the selected axes).

The axis set is configurable (``loss.smooth_axes``): the spatial axes ``h``/``w``
are the default, while the depth axis ``z`` is **excluded by default** — total
variation along depth is an antagonist to the depth-differentiation the ordering
regulariser (§3.7) is trying to create, so smoothing along Z fights the Z-axis
semantics. Pass ``["h", "w", "z"]`` to restore the fully-3-D behaviour.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor

__all__ = ["smoothness_loss", "SMOOTH_AXES"]

# Volume layout is [B, H', W', L, C]; map axis names to tensor dims.
SMOOTH_AXES = {"h": 1, "w": 2, "z": 3}


def smoothness_loss(volume: Tensor, axes: Sequence[str] = ("h", "w")) -> Tensor:
    """Per-edge total variation over the selected volume axes.

    Args:
        volume: Latent voxel volume ``[B, H', W', L, C]``.
        axes: Non-empty subset of ``{"h", "w", "z"}`` selecting which neighbour
            edges to penalise (default the spatial ``h``/``w`` plane; ``z`` is
            depth).

    Returns:
        Scalar: the mean over ``axes`` of each axis's mean squared first
        difference. A larger axis set can only *add* non-negative terms, so
        toggling ``z`` changes the value.

    Raises:
        ValueError: if ``axes`` is empty or names an unknown axis.

    Shape:
        - Input: ``[B, H', W', L, C]``.
        - Output: scalar.
    """
    if len(axes) == 0:
        raise ValueError("smoothness_loss requires at least one axis")
    unknown = [a for a in axes if a not in SMOOTH_AXES]
    if unknown:
        raise ValueError(
            f"unknown smoothness axis/axes {unknown}; must be in {sorted(SMOOTH_AXES)}"
        )
    terms = [volume.diff(dim=SMOOTH_AXES[a]).pow(2).mean() for a in axes]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total / len(terms)
