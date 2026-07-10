"""umtvit.data — universal data pipeline (ARCHITECTURE §4).

Declarative, config-driven loading: ``imagefolder``, ``csv``, and the
zero-download ``shapes`` CI dataset, plus (U1) the augmentation registry,
two-view contrastive wrapper, and leakage-free grouped splits. Model and
training code never read dataset specifics from anywhere but the config.

U0 ships the ``shapes`` generator/dataset (:mod:`umtvit.data.shapes`). U1 adds
the imagefolder/csv item loaders (:mod:`umtvit.data.loaders`), the hash-based
grouped splits (:mod:`umtvit.data.splits`), the augmentation-policy registry
(:mod:`umtvit.data.augment`), and the config-driven, loader-agnostic
:class:`~umtvit.data.dataset.UniversalDataset` two-view / eval wrapper.

Only torch-free names are imported eagerly here (item enumeration, splits, the
policy registry); :class:`UniversalDataset` and :func:`augment` import torch
lazily, so ``import umtvit.data`` stays torch-free until a tensor is requested.
"""

from __future__ import annotations

from .augment import AUGMENTATION_POLICIES, augment, get_policy
from .dataset import UniversalDataset
from .loaders import IMAGE_EXTENSIONS, build_items
from .shapes import SHAPE_CLASSES, ShapesDataset, generate_shapes_dataset
from .splits import SPLIT_NAMES, split_of

__all__ = [
    "SHAPE_CLASSES",
    "ShapesDataset",
    "generate_shapes_dataset",
    "build_items",
    "IMAGE_EXTENSIONS",
    "split_of",
    "SPLIT_NAMES",
    "AUGMENTATION_POLICIES",
    "get_policy",
    "augment",
    "UniversalDataset",
]
