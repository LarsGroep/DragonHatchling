"""A small, fast CNN backbone — the default for notebook demos on CPU."""

from __future__ import annotations

import torch
from torch import nn

from hatchvision.models.backbones.base import Backbone, register_backbone


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


@register_backbone("simple_cnn")
class SimpleCNN(Backbone):
    def __init__(self, in_channels: int = 3, width: int = 32, **_) -> None:
        super().__init__()
        w = width
        self.stage1 = _block(in_channels, w)
        self.stage2 = _block(w, w * 2)
        self.stage3 = _block(w * 2, w * 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._feature_dim = w * 4

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage3(self.stage2(self.stage1(x)))
        return self.pool(x).flatten(1)

    def cam_layer(self):
        return self.stage3

    def hebbian_layers(self):
        return {"stage2": self.stage2, "stage3": self.stage3}
