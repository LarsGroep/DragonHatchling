"""Universal item enumeration for the three loaders (ARCHITECTURE §4).

:func:`build_items` turns a :class:`~umtvit.config.Config` into a flat list of
``(source, label, group)`` records plus the ordered class names — the single
shape every downstream stage consumes, regardless of where the data came
from. This is the "dataset swap = config only" contract: model and training
code never learn which loader produced the items.

Record fields:

- ``source`` — how :class:`~umtvit.data.dataset.UniversalDataset` will load the
  image. For ``imagefolder``/``csv`` it is a filesystem path (``str``); for
  ``shapes`` it is a ``(class_name, seed)`` tuple rendered on the fly by
  :mod:`umtvit.data.shapes` (no files touched).
- ``label`` — integer class index, or ``-1`` for unlabeled items (when no
  ``label_column`` is configured). Labels are used only by evaluation.
- ``group`` — the grouping key for leakage-free splits (e.g. a lesion id), or
  ``None`` when no ``group_column`` is configured.

Torch-free: this module builds Python lists from ``csv``/``os``/``pathlib``
and numpy only. Tensor materialisation happens later, in the dataset.
"""

from __future__ import annotations

import csv as csvmod
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from ..config import Config, ConfigError
from .shapes import SHAPE_CLASSES

__all__ = ["IMAGE_EXTENSIONS", "build_items", "Item"]

# Accepted image extensions for the file-backed loaders (lower-cased compare).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# An item is (source, label, group). ``source`` is a path or a (class, seed).
Source = Union[str, Tuple[str, int]]
Item = Tuple[Source, int, Optional[str]]


def build_items(cfg: Config) -> Tuple[List[Item], List[str]]:
    """Enumerate ``(items, class_names)`` for the configured dataset.

    Dispatches on ``cfg.dataset.loader``. Never opens image files (the csv
    loader stats candidate paths only to resolve which directory an image
    lives in); image bytes are read lazily by the dataset.
    """
    dataset = cfg.dataset
    if dataset.loader == "shapes":
        return _build_shapes(dataset)
    if dataset.loader == "imagefolder":
        return _build_imagefolder(dataset)
    if dataset.loader == "csv":
        return _build_csv(dataset)
    # Unreachable: config validation restricts loader to the known set.
    raise ConfigError(f"dataset.loader: unknown loader {dataset.loader!r}")


def _build_shapes(dataset) -> Tuple[List[Item], List[str]]:
    """Generate ``(class_name, seed)`` items for the synthetic shapes dataset.

    ``n_per_class`` images per class, each given a distinct per-item seed drawn
    from a generator seeded by ``splits.seed`` so the enumeration is stable.
    Rendering itself is delegated to :mod:`umtvit.data.shapes` at load time.
    """
    if dataset.n_per_class is None:
        raise ConfigError(
            "dataset.n_per_class: is required when loader == 'shapes'"
        )
    rng = np.random.default_rng(dataset.splits.seed)
    items: List[Item] = []
    for class_index, class_name in enumerate(SHAPE_CLASSES):
        for _ in range(dataset.n_per_class):
            seed = int(rng.integers(0, 2 ** 31))
            items.append(((class_name, seed), class_index, None))
    return items, list(SHAPE_CLASSES)


def _subtree_roots(root: Path) -> List[Path]:
    """Return the ``train/val/test`` subtrees under ``root`` if any exist.

    An imagefolder tree may either place class directories directly under the
    root, or split them under ``train/``, ``val/``, ``test/``. When any of the
    latter exist we enumerate across them (splits are still re-derived by the
    hash assignment; the subtrees are just where the images live).
    """
    subtrees = [root / name for name in ("train", "val", "test") if (root / name).is_dir()]
    return subtrees or [root]


def _build_imagefolder(dataset) -> Tuple[List[Item], List[str]]:
    """Enumerate a ``root/<class>/<image>`` tree (optionally split subtrees)."""
    root = Path(_single_dir(dataset.image_dir))
    roots = _subtree_roots(root)
    class_names = sorted(
        {child.name for r in roots for child in r.iterdir() if child.is_dir()}
    )
    class_to_index = {name: i for i, name in enumerate(class_names)}
    labeled = bool(dataset.label_column)

    items: List[Item] = []
    for r in roots:
        for class_dir in sorted(p for p in r.iterdir() if p.is_dir()):
            label = class_to_index[class_dir.name] if labeled else -1
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((str(image_path), label, None))
    return items, class_names


def _build_csv(dataset) -> Tuple[List[Item], List[str]]:
    """Enumerate items from a metadata CSV via the stdlib csv module.

    Each row's image name is ``row[path_column] + path_suffix``, resolved
    against ``image_dir`` (a directory or a list of directories, tried in
    order). Rows whose image is not found in any directory are skipped.
    Labels come from ``label_column`` when configured (else ``-1``); groups
    from ``group_column`` when configured (else ``None``).
    """
    dirs = _dir_list(dataset.image_dir)
    with open(dataset.metadata_csv, newline="", encoding="utf-8") as handle:
        rows = list(csvmod.DictReader(handle))

    label_column = dataset.label_column
    group_column = dataset.group_column
    _check_columns(rows, dataset)

    if label_column:
        class_names = sorted({row[label_column] for row in rows})
        class_to_index = {name: i for i, name in enumerate(class_names)}
    else:
        class_names = []
        class_to_index = {}

    items: List[Item] = []
    for row in rows:
        filename = row[dataset.path_column] + dataset.path_suffix
        path = _resolve(filename, dirs)
        if path is None:
            continue
        label = class_to_index[row[label_column]] if label_column else -1
        group = row[group_column] if group_column else None
        items.append((path, label, group))
    return items, class_names


def _check_columns(rows: List[dict], dataset) -> None:
    """Raise a field-named :class:`ConfigError` if a referenced column is absent."""
    if not rows:
        return
    header = rows[0].keys()
    for field_name, column in (
        ("path_column", dataset.path_column),
        ("label_column", dataset.label_column),
        ("group_column", dataset.group_column),
    ):
        if column and column not in header:
            raise ConfigError(
                f"dataset.{field_name}: column {column!r} not found in "
                f"{dataset.metadata_csv} (columns: {sorted(header)})"
            )


def _single_dir(image_dir: Union[str, List[str]]) -> str:
    """Return a single directory for the imagefolder loader (first if a list)."""
    if isinstance(image_dir, (list, tuple)):
        return image_dir[0]
    return image_dir


def _dir_list(image_dir: Union[str, List[str]]) -> List[str]:
    """Normalise ``image_dir`` to a list of directories."""
    if isinstance(image_dir, (list, tuple)):
        return list(image_dir)
    return [image_dir]


def _resolve(filename: str, dirs: List[str]) -> Optional[str]:
    """Return the first ``dir/filename`` that exists, or ``None``."""
    for directory in dirs:
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return candidate
    return None
