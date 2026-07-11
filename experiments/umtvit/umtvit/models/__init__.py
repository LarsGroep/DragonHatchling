"""umtvit.models — backbone, uplifting, and SOM (ARCHITECTURE §3).

Modules:

- ``patch_embed``     dual-scale (fine/coarse) tokenisation (U2)
- ``cross_attention`` CLS-bridged (default) + full-pair cross-scale attention (U2)
- ``fusion``          fine/coarse feature fusion onto a common H'×W' grid (U2)
- ``encoder``         pre-norm ViT encoder returning all L layer outputs (U2)
- ``backbone``        UMTViTBackbone wiring embed→cross→fusion→encoder (U2)
- ``uplifting``       per-layer projection into the H'×W'×L×C latent volume (U3)
- ``som3d``           differentiable Soft3DSOM (+ kohonen_ema variant) (U3)
- ``heads``           projection head for the contrastive objective (U3)
"""

from __future__ import annotations

from umtvit.models.backbone import UMTViTBackbone
from umtvit.models.cross_attention import CrossScaleBlock
from umtvit.models.encoder import SelfAttnBlock, TransformerEncoder
from umtvit.models.fusion import FeatureFusion
from umtvit.models.patch_embed import DualScalePatchEmbed

__all__ = [
    "DualScalePatchEmbed",
    "CrossScaleBlock",
    "FeatureFusion",
    "SelfAttnBlock",
    "TransformerEncoder",
    "UMTViTBackbone",
]
