"""umtvit.data — universal data pipeline (ARCHITECTURE §4).

Declarative, config-driven loading: ``imagefolder``, ``csv``, and the
zero-download ``shapes`` CI dataset, plus (U1) the augmentation registry,
two-view contrastive wrapper, and leakage-free grouped splits. Model and
training code never read dataset specifics from anywhere but the config.

U0 ships the ``shapes`` generator/dataset (:mod:`umtvit.data.shapes`); the
imagefolder/csv loaders, augmentation registry, and split logic land in U1.
"""

from __future__ import annotations

from .shapes import SHAPE_CLASSES, ShapesDataset, generate_shapes_dataset

__all__ = ["SHAPE_CLASSES", "ShapesDataset", "generate_shapes_dataset"]
