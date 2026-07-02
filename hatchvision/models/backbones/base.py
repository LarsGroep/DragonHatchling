"""Backbone interface + registry.

A backbone is any encoder that maps an image batch to a feature vector.
The contract is intentionally small so that very different architectures
(CNNs, ViTs, the experimental BDH encoder) are interchangeable:

* ``forward(x) -> [B, feature_dim]`` — pooled feature vector.
* ``feature_dim`` — width of that vector.
* ``cam_layer()`` — the module whose output is the last spatially-resolved
  feature map, used by Grad-CAM.  Return ``None`` if the architecture has no
  meaningful spatial map.
* ``hebbian_layers()`` — named modules whose activations the Hebbian feature
  memory should observe (usually late, semantically rich layers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type

import torch
from torch import nn


class Backbone(nn.Module, ABC):
    """Common interface for all pluggable encoders."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the pooled feature vector."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images ``[B, C, H, W]`` to features ``[B, feature_dim]``."""

    def cam_layer(self) -> Optional[nn.Module]:
        """Module producing the last spatial feature map (for Grad-CAM)."""
        return None

    def hebbian_layers(self) -> Dict[str, nn.Module]:
        """Named modules the Hebbian memory should record from."""
        return {}


_BACKBONES: Dict[str, Callable[..., Backbone]] = {}


def register_backbone(name: str) -> Callable:
    """Decorator registering a Backbone class or factory function."""

    def wrap(fn):
        key = name.lower()
        if key in _BACKBONES:
            raise KeyError(f"backbone {key!r} already registered")
        _BACKBONES[key] = fn
        return fn

    return wrap


def build_backbone(name: str, **kwargs) -> Backbone:
    key = name.lower()
    if key not in _BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; available: {sorted(_BACKBONES)}")
    return _BACKBONES[key](**kwargs)


def available_backbones() -> List[str]:
    return sorted(_BACKBONES)
