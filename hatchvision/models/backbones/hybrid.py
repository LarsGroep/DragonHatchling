"""Hybrid backbone: frozen pretrained encoder → BDH sparse neuron layer.

The practical default for fine-grained datasets like CUB-200: a pretrained
torchvision encoder (frozen by default) supplies strong visual features,
and a BDH-style lift maps them into a high-dimensional **positive, sparse
neuron space** (Linear → ReLU, ``neuron_dim >> feature_dim``) — the same
interpretable unit the pure BDH backbone exposes.  The Hebbian feature
memory observes those neurons, so concept clustering and attribute
grounding work identically for pure-BDH and hybrid runs; only accuracy and
compute differ.

With the encoder frozen only the BDH lift + classifier head train, which is
fast even on CPU and keeps the pretrained features stationary — a stable
substrate for Hebbian statistics.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torchvision import models as tvm

from hatchvision.models.backbones.base import Backbone, register_backbone

_ENCODERS = {
    "resnet18": (tvm.resnet18, 512),
    "resnet34": (tvm.resnet34, 512),
    "resnet50": (tvm.resnet50, 2048),
}


@register_backbone("hybrid")
class HybridBDHBackbone(Backbone):
    def __init__(
        self,
        encoder: str = "resnet50",
        pretrained: bool = True,
        freeze_encoder: bool = True,
        neuron_dim: int = 4096,
        in_channels: int = 3,
        **_,
    ) -> None:
        super().__init__()
        if encoder not in _ENCODERS:
            raise KeyError(f"unknown encoder {encoder!r}; options: {sorted(_ENCODERS)}")
        factory, enc_dim = _ENCODERS[encoder]
        net = factory(weights="DEFAULT" if pretrained else None)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        net.fc = nn.Identity()
        self.encoder = net
        self.frozen = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # The interpretable unit: sparse, positive BDH neurons.
        self.neurons = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, neuron_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self.readback = nn.Linear(neuron_dim, enc_dim, bias=False)
        self._dim = enc_dim

    @property
    def feature_dim(self) -> int:
        return self._dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            self.encoder.eval()  # keep BatchNorm statistics fixed
            with torch.no_grad():
                f = self.encoder(x)
        else:
            f = self.encoder(x)
        y = self.neurons(f)                 # [B, neuron_dim] sparse & positive
        return f + self.readback(y)         # residual synaptic integration

    def cam_layer(self):
        return self.encoder.layer4

    def hebbian_layers(self) -> Dict[str, nn.Module]:
        return {"neurons": self.neurons}
