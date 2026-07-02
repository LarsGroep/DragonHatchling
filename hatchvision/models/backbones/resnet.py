"""torchvision ResNet backbones behind the common Backbone interface."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

from hatchvision.models.backbones.base import Backbone, register_backbone

_FACTORIES = {
    "resnet18": (tvm.resnet18, 512),
    "resnet34": (tvm.resnet34, 512),
    "resnet50": (tvm.resnet50, 2048),
}


class ResNetBackbone(Backbone):
    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = False,
        in_channels: int = 3,
        small_input: bool = False,
        **_,
    ) -> None:
        super().__init__()
        factory, dim = _FACTORIES[arch]
        weights = "DEFAULT" if pretrained else None
        net = factory(weights=weights)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        if small_input:
            # CIFAR-style stem: keep resolution for 32x32 inputs.
            net.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        self.net = net
        self._feature_dim = dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def cam_layer(self):
        return self.net.layer4

    def hebbian_layers(self):
        return {"layer3": self.net.layer3, "layer4": self.net.layer4}


for _arch in _FACTORIES:
    register_backbone(_arch)(
        lambda _a=_arch, **kw: ResNetBackbone(arch=_a, **kw)
    )
