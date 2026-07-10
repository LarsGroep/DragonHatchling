"""umtvit.models — backbone, uplifting, and SOM (ARCHITECTURE §3).

Planned modules (land in U2/U3, per ARCHITECTURE §5/§9):

- ``patch_embed``     dual-scale (fine/coarse) tokenisation
- ``cross_attention`` CLS-bridged (default) + full-pair cross-scale attention
- ``fusion``          fine/coarse feature fusion onto a common H'×W' grid
- ``encoder``         pre-norm ViT encoder returning all L layer outputs
- ``uplifting``       per-layer projection into the H'×W'×L×C latent volume
- ``som3d``           differentiable Soft3DSOM (+ kohonen_ema variant)
- ``heads``           projection head for the contrastive objective

Stub package (U0). No model code is defined yet.
"""

from __future__ import annotations

__all__: list[str] = []
