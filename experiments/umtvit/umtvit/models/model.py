"""UMT-ViT model: backbone + spatial uplifting + projection head (ARCHITECTURE §3).

Composes the U2 backbone (dual-scale embed → cross-attention → fusion → encoder,
all ``L`` layer outputs kept), the U3 spatial uplifting into the latent voxel
volume, and the contrastive projection head into a single ``forward``:

    x → backbone → layers[L] → uplifting → volume [B,H',W',L,Cv]
                                      ↓ pooled [B,L*Cv] → head → proj [B,proj_dim]

The 3-D SOM (:class:`~umtvit.models.som3d.Soft3DSOM`) is deliberately kept as a
separate module — it operates on the volume's voxel features and, in
``kohonen_ema`` mode, updates its own weights outside autograd, so it does not
belong inside the encoder's parameter graph. The trainer (U4) holds both and
feeds ``forward(x)["volume"]`` to the SOM.

This mirrors the notebook reference's monolithic ``UMTViT`` in *semantics and
shapes* (same uplift, same pooled readout, same head), but is built by
composing the already-committed backbone rather than re-implementing embed /
cross / fusion / encoder inline — so numerical parity with the notebook class is
not expected (different module nesting → different init RNG draws), only shape
and behavioural parity.
"""

from __future__ import annotations

from typing import Dict, List

from torch import Tensor, nn

from umtvit.config import Config
from umtvit.models.backbone import UMTViTBackbone
from umtvit.models.heads import ProjectionHead
from umtvit.models.uplifting import SpatialUplifting

__all__ = ["UMTViT"]


class UMTViT(nn.Module):
    """Full UMT-ViT encoder: backbone → uplifting → projection head.

    Args:
        config: A validated :class:`~umtvit.config.Config`. All geometry is read
            from it; nothing is hardcoded.

    Shape:
        - Input: ``x`` of shape ``[B, channels, image_size, image_size]``.
        - Output: ``dict`` with
          ``"volume"`` — ``[B, vg, vg, depth, volume_channels]`` latent voxel
          volume; ``"pooled"`` — ``[B, depth*volume_channels]`` per-slice
          spatial-mean readout; ``"proj"`` — ``[B, proj_dim]`` contrastive
          embedding; ``"layers"`` — the ``depth`` raw encoder layer outputs
          (each ``[B, vg², dim]``), passed through for downstream losses/probes.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.backbone = UMTViTBackbone(config)
        self.uplifting = SpatialUplifting(config)
        self.head = ProjectionHead(config)

    def forward(self, x: Tensor) -> Dict[str, object]:
        out = self.backbone(x)
        layers: List[Tensor] = out["layers"]  # type: ignore[assignment]
        volume = self.uplifting(layers)
        pooled = self.uplifting.pooled(volume)
        proj = self.head(pooled)
        return {
            "volume": volume,
            "pooled": pooled,
            "proj": proj,
            "layers": layers,
        }
