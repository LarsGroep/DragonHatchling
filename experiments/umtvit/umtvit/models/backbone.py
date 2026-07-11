"""UMT-ViT dual-scale backbone (ARCHITECTURE §3.1-§3.3, §9 row U2).

Wires the U2 components into the encoder trunk of UMT-ViT:

    embed → cross_rounds × (per-stream self-attn + cross-scale attn)
          → fusion → encoder (all L layer outputs kept)

This module is the front end only: spatial uplifting into the latent voxel
volume, the 3-D SOM, and the projection/heads are U3+ and live elsewhere. The
backbone therefore exposes exactly what U3 consumes — every encoder layer's
token sequence and the fused tokens that entered the encoder.

Schema note (``volume_h``/``volume_w`` vs ``volume_grid``): the notebook
reference uses a single ``volume_grid`` int, while ``config.py`` carries
separate ``volume_h``/``volume_w``. We keep the schema unchanged (smallest
diff) and **assert the volume grid is square** here, using ``volume_h`` as the
grid side. A non-square volume would need a rectangular fusion grid, which the
architecture does not (yet) call for.
"""

from __future__ import annotations

from typing import Dict, List

from torch import Tensor, nn

from umtvit.config import Config
from umtvit.models.cross_attention import CrossScaleBlock
from umtvit.models.encoder import SelfAttnBlock, TransformerEncoder
from umtvit.models.fusion import FeatureFusion
from umtvit.models.patch_embed import DualScalePatchEmbed

__all__ = ["UMTViTBackbone"]


class UMTViTBackbone(nn.Module):
    """Dual-scale cross-attention ViT backbone, config-driven.

    Args:
        config: A validated :class:`~umtvit.config.Config`. All geometry
            (image size, patch sizes, dim, depth, heads, mlp_ratio,
            cross-attention mode, cross_rounds, volume grid) is read from
            ``config.model``/``config.dataset``; nothing is hardcoded.

    Shape:
        - Input: ``x`` of shape ``[B, channels, image_size, image_size]``.
        - Output: ``dict`` with
          ``"layers"`` — a list of ``depth`` tensors, each
          ``[B, volume_grid², dim]`` (per-layer encoder outputs, shallow →
          deep), and ``"fused"`` — the ``[B, volume_grid², dim]`` fused tokens
          that entered the encoder.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m = config.model
        d = config.dataset

        # image_size is unified onto dataset.image_size by Config.validate();
        # fall back to the dataset value if a bare (unvalidated) config is used.
        image_size = m.image_size if m.image_size is not None else d.image_size

        if m.volume_h != m.volume_w:
            raise ValueError(
                "UMTViTBackbone requires a square volume grid "
                f"(volume_h == volume_w); got volume_h={m.volume_h}, "
                f"volume_w={m.volume_w}"
            )
        self.volume_grid = m.volume_h
        self.dim = m.dim
        self.depth = m.depth

        self.embed = DualScalePatchEmbed(
            image_size=image_size,
            fine_patch=m.fine_patch,
            coarse_patch=m.coarse_patch,
            dim=m.dim,
            channels=d.channels,
        )

        # One self-attn block per stream per cross round, then a cross-scale
        # exchange: cross_rounds × (per-stream self-attn + cross).
        self.stream_fine = nn.ModuleList(
            SelfAttnBlock(m.dim, m.heads, m.mlp_ratio) for _ in range(m.cross_rounds)
        )
        self.stream_coarse = nn.ModuleList(
            SelfAttnBlock(m.dim, m.heads, m.mlp_ratio) for _ in range(m.cross_rounds)
        )
        self.cross = nn.ModuleList(
            CrossScaleBlock(m.dim, m.heads, m.cross_attention)
            for _ in range(m.cross_rounds)
        )

        self.fusion = FeatureFusion(
            dim=m.dim,
            grid_fine=self.embed.grid_fine,
            grid_coarse=self.embed.grid_coarse,
            volume_grid=self.volume_grid,
        )

        self.encoder = TransformerEncoder(
            dim=m.dim, depth=m.depth, heads=m.heads, mlp_ratio=m.mlp_ratio
        )

    def forward(self, x: Tensor) -> Dict[str, object]:
        """Run the backbone; return ``{"layers": [...], "fused": ...}``."""
        tokens_fine, tokens_coarse = self.embed(x)
        for self_f, self_c, cross in zip(
            self.stream_fine, self.stream_coarse, self.cross
        ):
            tokens_fine, tokens_coarse = cross(
                self_f(tokens_fine), self_c(tokens_coarse)
            )
        fused = self.fusion(tokens_fine, tokens_coarse)
        layers: List[Tensor] = self.encoder(fused)
        return {"layers": layers, "fused": fused}
