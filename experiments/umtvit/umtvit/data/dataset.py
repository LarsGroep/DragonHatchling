"""The one dataset class for every loader (ARCHITECTURE §4, §3.6).

:class:`UniversalDataset` is the single map-style dataset the whole experiment
uses. It reads its entire behaviour from a :class:`~umtvit.config.Config`:
:func:`~umtvit.data.loaders.build_items` enumerates records, the hash-based
:func:`~umtvit.data.splits.split_of` keeps the requested ``split``, and the
named policy from the augmentation registry shapes each view — so swapping
datasets is a config-only operation, exactly as required.

Two modes:

- ``"two_view"`` — self-supervised sampling. Returns ``(view_a, view_b,
  label)``: two independently augmented views of the same image (fresh
  randomness each), the pair NT-Xent contrasts.
- ``"eval"`` — deterministic resize-only view. Returns ``(image, label)`` with
  no augmentation, for probing / k-NN / metrics.

Labels are integer class indices, or ``-1`` in unlabeled mode (no
``label_column``); either way iteration works. Views are ``[C, H, W]`` float
tensors in ``[0, 1]``.

This is a map-style dataset (``__len__`` + ``__getitem__``); a
``torch.utils.data.DataLoader`` consumes it directly. torch is imported lazily
so enumerating items / building splits stays torch-free.
"""

from __future__ import annotations

from typing import Tuple

from ..config import Config
from .augment import augment, get_policy
from .loaders import build_items
from .shapes import render_shape_image
from .splits import split_of

__all__ = ["UniversalDataset"]


class UniversalDataset:
    """Config-driven, loader-agnostic dataset (map-style)."""

    def __init__(self, cfg: Config, split: str, mode: str) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train'/'val'/'test', got {split!r}"
            )
        if mode not in ("two_view", "eval"):
            raise ValueError(
                f"mode must be 'two_view'/'eval', got {mode!r}"
            )
        dataset = cfg.dataset
        all_items, self.classes = build_items(cfg)
        # Keep only items landing in this split. Grouped items hash by their
        # group key (whole group ⇒ one split); ungrouped items hash by index.
        self.items = [
            item
            for index, item in enumerate(all_items)
            if split_of(item[2] if item[2] is not None else index, dataset.splits) == split
        ]
        self.size = dataset.image_size
        self.mode = mode
        self.split = split
        self.policy = get_policy(dataset.augmentation)

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, source):
        """Materialise a source into an RGB PIL image."""
        from PIL import Image

        if isinstance(source, tuple):  # shapes: (class_name, seed)
            class_name, seed = source
            # Render at 2× so the crop op has room; deterministic in the seed.
            array = render_shape_image(class_name, self.size * 2, index=0, seed=seed)
            return Image.fromarray(array, mode="RGB")
        return Image.open(source).convert("RGB")

    def __getitem__(self, index: int) -> Tuple:
        import numpy as np
        import torch

        source, label, _group = self.items[index]
        img = self._load(source)

        if self.mode == "two_view":
            # Fresh entropy per view so the two augmentations differ.
            view_a = augment(img, self.size, self.policy, np.random.default_rng())
            view_b = augment(img, self.size, self.policy, np.random.default_rng())
            return view_a, view_b, label

        # eval: deterministic resize-only tensor.
        from PIL import Image

        resized = img.resize((self.size, self.size), Image.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized, np.float32) / 255.0).permute(2, 0, 1)
        return tensor.contiguous(), label
