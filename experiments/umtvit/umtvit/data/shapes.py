"""Deterministic synthetic shapes dataset (ARCHITECTURE §4, §7).

A zero-download, seeded, CPU-only dataset of RGB images — each a single simple
shape (circle / square / triangle) drawn on a noisy background — with the
shape class as the label. It is the CI workhorse for every later milestone:
fast, reproducible, and free of any network or licensing dependency.

Determinism contract: pixels are a pure function of
``(shape_class, sample_index, seed, image_size)``. The imagefolder writer
:func:`generate_shapes_dataset` and the in-memory :class:`ShapesDataset` both
render through :func:`render_shape_image`, so the file on disk and the tensor
in memory are bit-identical for the same inputs. Re-running with the same seed
reproduces every pixel (see ``tests/test_shapes.py``).

Generation uses numpy + PIL only. :class:`ShapesDataset` additionally imports
torch (lazily, at construction) to yield ``[C, H, W]`` float tensors in
``[0, 1]``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

__all__ = ["SHAPE_CLASSES", "render_shape_image", "generate_shapes_dataset", "ShapesDataset"]

# The three shape classes. Index in this tuple is the integer label.
SHAPE_CLASSES: Tuple[str, ...] = ("circle", "square", "triangle")


def _sample_rng(shape_class: str, index: int, seed: int) -> np.random.Generator:
    """Return a numpy Generator seeded purely by ``(class, index, seed)``.

    Using a :class:`numpy.random.SeedSequence` over the class *index* (not its
    name) and the sample index gives every (class, index) pair an independent,
    reproducible stream that is stable across processes and platforms.
    """
    class_index = SHAPE_CLASSES.index(shape_class)
    seed_sequence = np.random.SeedSequence([int(seed), class_index, int(index)])
    return np.random.default_rng(seed_sequence)


def render_shape_image(shape_class: str, image_size: int, index: int, seed: int) -> np.ndarray:
    """Render one shape image deterministically.

    Args:
        shape_class: one of :data:`SHAPE_CLASSES`.
        image_size: side length of the square RGB image (pixels).
        index: sample index within the class (varies the instance).
        seed: dataset seed.

    Returns:
        ``uint8`` array of shape ``(image_size, image_size, 3)`` in ``[0, 255]``.
    """
    if shape_class not in SHAPE_CLASSES:
        raise ValueError(
            f"unknown shape_class {shape_class!r}; must be one of {list(SHAPE_CLASSES)}"
        )
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    rng = _sample_rng(shape_class, index, seed)

    # Light, mildly-tinted background with per-pixel Gaussian speckle so the
    # model cannot trivially threshold foreground from background.
    base = rng.integers(170, 230, size=3).astype(np.float32)
    noise = rng.normal(0.0, 12.0, size=(image_size, image_size, 3)).astype(np.float32)
    background = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)

    image = Image.fromarray(background, mode="RGB")
    draw = ImageDraw.Draw(image)

    # A dark, saturated foreground colour, well separated from the background.
    color = tuple(int(v) for v in rng.integers(0, 110, size=3))
    radius = int(rng.integers(image_size // 6, max(image_size // 6 + 1, image_size // 3)))
    cx = int(rng.integers(radius + 1, image_size - radius))
    cy = int(rng.integers(radius + 1, image_size - radius))

    if shape_class == "circle":
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    elif shape_class == "square":
        draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    elif shape_class == "triangle":
        points = [
            (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            for angle in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)
        ]
        draw.polygon(points, fill=color)

    return np.asarray(image, dtype=np.uint8)


def generate_shapes_dataset(
    root: Union[str, Path],
    n_per_class: int,
    image_size: int = 64,
    seed: int = 0,
) -> Path:
    """Write a deterministic imagefolder-style shapes tree to ``root``.

    Layout: ``root/<class>/<class>_<index>.png``, one subdirectory per entry
    of :data:`SHAPE_CLASSES`, ``n_per_class`` images each. Re-running with the
    same arguments overwrites with byte-identical pixels.

    Args:
        root: destination directory (created if absent).
        n_per_class: number of images to write per shape class (> 0).
        image_size: side length of each square image.
        seed: dataset seed controlling all randomness.

    Returns:
        The ``root`` path as a :class:`~pathlib.Path`.
    """
    if n_per_class <= 0:
        raise ValueError(f"n_per_class must be positive, got {n_per_class}")

    root_path = Path(root)
    for shape_class in SHAPE_CLASSES:
        class_dir = root_path / shape_class
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(n_per_class):
            array = render_shape_image(shape_class, image_size, index, seed)
            Image.fromarray(array, mode="RGB").save(class_dir / f"{shape_class}_{index:05d}.png")
    return root_path


class ShapesDataset:
    """In-memory, deterministic torch ``Dataset`` of shapes (ARCHITECTURE §4).

    Yields ``(image, label)`` where ``image`` is a float32 tensor of shape
    ``[3, H, W]`` in ``[0, 1]`` (channels-first, PyTorch convention) and
    ``label`` is the ``int`` class index into :data:`SHAPE_CLASSES`. No files
    are written; images are rendered on access through
    :func:`render_shape_image`, so a given ``(seed, image_size)`` always
    produces the same pixels.

    Samples are ordered class-major: all of class 0, then class 1, ....
    """

    classes: Tuple[str, ...] = SHAPE_CLASSES

    def __init__(self, n_per_class: int, image_size: int = 64, seed: int = 0) -> None:
        if n_per_class <= 0:
            raise ValueError(f"n_per_class must be positive, got {n_per_class}")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.n_per_class = int(n_per_class)
        self.image_size = int(image_size)
        self.seed = int(seed)
        # Flat (class_index, sample_index) addressing over the class-major order.
        self._samples: List[Tuple[int, int]] = [
            (class_index, sample_index)
            for class_index in range(len(SHAPE_CLASSES))
            for sample_index in range(self.n_per_class)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple["torch.Tensor", int]:  # noqa: F821
        # torch is imported lazily so `import umtvit.data` stays torch-free
        # until a tensor is actually requested.
        import torch

        if index < 0:
            index += len(self._samples)
        if not 0 <= index < len(self._samples):
            raise IndexError(f"index {index} out of range for {len(self._samples)} samples")

        class_index, sample_index = self._samples[index]
        array = render_shape_image(
            SHAPE_CLASSES[class_index], self.image_size, sample_index, self.seed
        )
        # HWC uint8 [0,255] -> CHW float32 [0,1]. Copy first: the array shares
        # PIL's read-only buffer, which torch.from_numpy would warn about.
        tensor = (
            torch.from_numpy(np.array(array, copy=True))
            .float()
            .div_(255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        return tensor, class_index
