"""Dataset abstraction layer (§4).

A dataset is a declarative :class:`DatasetSpec` plus a :class:`DatasetAdapter`
subclass. Everything downstream (transforms, training, packs, UI labels,
colors) derives from the spec.

At M0 the **registry is fully working** (register / get / list). Adapter
methods that require torch/PIL/data on disk (``load``, ``preprocess``,
``augment``, ``splits``, ``gallery``, ``viz_hooks``) are abstract and land at
M1. Import of this module must not require torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
)

# A preprocessing/augmentation transform is any callable image -> image/tensor.
# Typed loosely at M0 to avoid a hard torch/torchvision dependency.
Transform = Callable[[Any], Any]


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of a dataset (§4)."""

    name: str
    display_name: str
    num_classes: int
    image_size: int = 224
    channels: int = 3
    class_names: List[str] = field(default_factory=list)
    class_colors: List[str] = field(default_factory=list)
    license: str = ""
    citation: str = ""
    kaggle_sources: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.class_names and len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but "
                f"num_classes is {self.num_classes}"
            )
        if self.class_colors and len(self.class_colors) != self.num_classes:
            raise ValueError(
                f"class_colors has {len(self.class_colors)} entries but "
                f"num_classes is {self.num_classes}"
            )


@dataclass
class Sample:
    """One dataset item: an image reference plus its label."""

    image: Any
    label: int
    image_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitPolicy:
    """How the dataset is partitioned, including grouped (leak-free) splits."""

    train: List[int] = field(default_factory=list)
    val: List[int] = field(default_factory=list)
    test: List[int] = field(default_factory=list)
    group_key: Optional[str] = None


@dataclass
class VizHooks:
    """Per-dataset UI extras (color overrides, legends, etc.)."""

    extras: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Adapter turning a raw on-disk dataset into the ViTreous pipeline (§4).

    Concrete adapters set a class-level :attr:`spec` and implement the data
    methods. Implementations arrive at M1 (EuroSAT, Oxford-IIIT Pet,
    imagefolder).
    """

    spec: DatasetSpec

    @abstractmethod
    def load(self, root: str, split: str) -> Iterable[Sample]:
        """Yield samples for a split from a raw dataset directory."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self) -> Transform:
        """Return the deterministic eval transform."""
        raise NotImplementedError

    @abstractmethod
    def augment(self) -> Transform:
        """Return the train-time augmentation transform."""
        raise NotImplementedError

    @abstractmethod
    def splits(self) -> SplitPolicy:
        """Return the split policy (including grouped splits)."""
        raise NotImplementedError

    @abstractmethod
    def gallery(self, n: int = 75) -> List[Sample]:
        """Return a curated set of demo images."""
        raise NotImplementedError

    @abstractmethod
    def viz_hooks(self) -> VizHooks:
        """Return per-dataset UI extras."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Registry — fully functional at M0.
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}

A = TypeVar("A", bound=DatasetAdapter)


def register_dataset(name: str) -> Callable[[Type[A]], Type[A]]:
    """Class decorator registering a :class:`DatasetAdapter` under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is empty or already registered.
    """

    if not name:
        raise ValueError("dataset name must be a non-empty string")

    def _decorator(cls: Type[A]) -> Type[A]:
        if not isinstance(cls, type) or not issubclass(cls, DatasetAdapter):
            raise TypeError(
                f"{cls!r} must be a subclass of DatasetAdapter to register"
            )
        if name in _REGISTRY:
            raise ValueError(f"dataset {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_dataset(name: str) -> Type[DatasetAdapter]:
    """Return the registered adapter class for ``name``.

    Raises
    ------
    KeyError
        If no adapter is registered under ``name``.
    """

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"no dataset registered as {name!r}; available: {available}"
        ) from exc


def list_datasets() -> List[str]:
    """Return the sorted names of all registered datasets."""

    return sorted(_REGISTRY)


def _clear_registry() -> None:
    """Test helper: empty the registry. Not part of the public API."""

    _REGISTRY.clear()


__all__ = [
    "Transform",
    "DatasetSpec",
    "Sample",
    "SplitPolicy",
    "VizHooks",
    "DatasetAdapter",
    "register_dataset",
    "get_dataset",
    "list_datasets",
]
