"""Dataset interface: spec, loader base class, and registry.

The framework is dataset-agnostic by construction: everything downstream
(model input channels, transforms, class names, normalization constants,
attribute grounding, ONNX manifest) is derived from a :class:`DatasetSpec`.
Swapping datasets therefore means swapping the loader — nothing else.

A loader implements two methods (``train_dataset`` / ``val_dataset``) and
provides a spec.  Optionally it can expose *attributes* — human-readable
binary features per validation image (e.g. CUB-200's "has_wing_color::yellow")
— which the explainability layer uses to ground Hebbian concepts in
nameable visual features.  Datasets without attributes simply return ``None``
and concept labeling falls back to class affinity + exemplars.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class DatasetSpec:
    """Everything downstream components need to know about a dataset."""

    name: str
    num_classes: int
    class_names: Sequence[str]
    image_size: int
    in_channels: int = 3
    mean: Tuple[float, ...] = ()   # per-channel; empty = 0.5 per channel
    std: Tuple[float, ...] = ()    # per-channel; empty = 0.5 per channel

    def normalization(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        mean = self.mean or tuple([0.5] * self.in_channels)
        std = self.std or tuple([0.5] * self.in_channels)
        return mean, std


class DatasetLoader(ABC):
    """Base class for dataset plugins.

    Subclasses implement :meth:`train_dataset` and :meth:`val_dataset`
    (returning tensor-yielding ``torch.utils.data.Dataset`` objects) and set
    ``self.spec`` in ``__init__``.  ``limit_train`` / ``limit_val`` subset the
    datasets deterministically — useful for smoke runs on any dataset.
    """

    spec: DatasetSpec

    def __init__(
        self,
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
    ) -> None:
        self.limit_train = limit_train
        self.limit_val = limit_val

    # ------------------------------------------------------------- interface

    @abstractmethod
    def train_dataset(self) -> Dataset:
        """Training split with augmentation transforms applied."""

    @abstractmethod
    def val_dataset(self) -> Dataset:
        """Validation split with deterministic transforms applied."""

    # Attribute annotations (optional; enables concept grounding) -----------

    def attribute_names(self) -> Optional[List[str]]:
        """Human-readable names of binary attributes, or None."""
        return None

    def val_attribute_matrix(self) -> Optional[torch.Tensor]:
        """``[len(val), num_attributes]`` binary/float matrix aligned with
        the *unlimited* validation dataset order, or None."""
        return None

    # ------------------------------------------------------------ conveniences

    def _maybe_limit(self, ds: Dataset, limit: Optional[int]) -> Dataset:
        if limit is not None and limit < len(ds):
            return Subset(ds, range(limit))
        return ds

    def dataloaders(
        self,
        batch_size: int = 64,
        num_workers: int = 2,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        train = self._maybe_limit(self.train_dataset(), self.limit_train)
        val = self._maybe_limit(self.val_dataset(), self.limit_val)
        return (
            DataLoader(
                train,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                drop_last=False,
            ),
            DataLoader(val, batch_size=batch_size, num_workers=num_workers),
        )

    def probe_batch(self, n: int = 64) -> torch.Tensor:
        """First ``n`` validation images as one tensor (concept probing)."""
        val = self.val_dataset()
        n = min(n, len(val))
        return torch.stack([val[i][0] for i in range(n)])

    def probe_attributes(self, n: int = 64) -> Optional[torch.Tensor]:
        """Attribute rows aligned with :meth:`probe_batch`, or None."""
        mat = self.val_attribute_matrix()
        if mat is None:
            return None
        return mat[: min(n, mat.shape[0])]


# ------------------------------------------------------------------ registry

_LOADERS: Dict[str, Callable[..., DatasetLoader]] = {}


def register_loader(name: str) -> Callable:
    """Decorator registering a DatasetLoader class or factory."""

    def wrap(fn):
        key = name.lower()
        if key in _LOADERS:
            raise KeyError(f"dataset loader {key!r} already registered")
        _LOADERS[key] = fn
        return fn

    return wrap


def build_loader(name: str, **kwargs) -> DatasetLoader:
    key = name.lower()
    if key not in _LOADERS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(_LOADERS)}")
    return _LOADERS[key](**kwargs)


def available_loaders() -> List[str]:
    return sorted(_LOADERS)
