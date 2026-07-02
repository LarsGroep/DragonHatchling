"""Classifier = any Backbone + a linear head."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from hatchvision.data.base import DatasetSpec
from hatchvision.models.backbones import Backbone, build_backbone


class ImageClassifier(nn.Module):
    def __init__(self, backbone: Backbone, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.feature_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def cam_layer(self) -> Optional[nn.Module]:
        return self.backbone.cam_layer()

    def hebbian_layers(self) -> Dict[str, nn.Module]:
        return self.backbone.hebbian_layers()


def create_model(
    backbone_name: str,
    spec: DatasetSpec,
    dropout: float = 0.1,
    **backbone_kwargs,
) -> ImageClassifier:
    """Build a classifier for a dataset; backbone kwargs are passed through.

    Dataset-dependent arguments (channels, image size, small-input stem) are
    derived from the spec so callers never hard-code them.
    """
    backbone_kwargs.setdefault("in_channels", spec.in_channels)
    backbone_kwargs.setdefault("image_size", spec.image_size)
    if backbone_name.startswith("resnet"):
        backbone_kwargs.setdefault("small_input", spec.image_size <= 64)
        backbone_kwargs.pop("image_size", None)  # resnets are size-agnostic
    backbone = build_backbone(backbone_name, **backbone_kwargs)
    return ImageClassifier(backbone, num_classes=spec.num_classes, dropout=dropout)
