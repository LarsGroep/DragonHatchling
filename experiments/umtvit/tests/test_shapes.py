"""Shapes dataset tests (ARCHITECTURE §4, §7).

Covers the determinism contract (same seed => identical pixels), the in-memory
torch tensor shape/dtype/range, and the imagefolder-tree structure written by
:func:`generate_shapes_dataset`. CPU-only; file output goes to ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from umtvit.data.shapes import (
    SHAPE_CLASSES,
    ShapesDataset,
    generate_shapes_dataset,
    render_shape_image,
)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_identical_pixels():
    a = render_shape_image("circle", 32, index=0, seed=7)
    b = render_shape_image("circle", 32, index=0, seed=7)
    assert np.array_equal(a, b)


def test_different_seed_differs():
    a = render_shape_image("triangle", 32, index=0, seed=7)
    b = render_shape_image("triangle", 32, index=0, seed=8)
    assert not np.array_equal(a, b)


def test_different_index_differs():
    a = render_shape_image("square", 32, index=0, seed=7)
    b = render_shape_image("square", 32, index=1, seed=7)
    assert not np.array_equal(a, b)


def test_dataset_reinstantiation_is_deterministic():
    ds1 = ShapesDataset(n_per_class=3, image_size=32, seed=7)
    ds2 = ShapesDataset(n_per_class=3, image_size=32, seed=7)
    for i in range(len(ds1)):
        t1, l1 = ds1[i]
        t2, l2 = ds2[i]
        assert l1 == l2
        assert np.array_equal(t1.numpy(), t2.numpy())


def test_render_rejects_unknown_class():
    with pytest.raises(ValueError):
        render_shape_image("hexagon", 32, index=0, seed=0)


# --------------------------------------------------------------------------- #
# Tensor contract
# --------------------------------------------------------------------------- #
def test_tensor_shape_dtype_range():
    import torch

    ds = ShapesDataset(n_per_class=2, image_size=32, seed=7)
    tensor, label = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert isinstance(label, int)


def test_dataset_length_and_labels_class_major():
    ds = ShapesDataset(n_per_class=4, image_size=32, seed=7)
    assert len(ds) == 4 * len(SHAPE_CLASSES)
    labels = [ds[i][1] for i in range(len(ds))]
    # Class-major ordering: [0,0,0,0, 1,1,1,1, 2,2,2,2].
    assert labels == [c for c in range(len(SHAPE_CLASSES)) for _ in range(4)]
    assert set(labels) == set(range(len(SHAPE_CLASSES)))


def test_negative_index_supported():
    ds = ShapesDataset(n_per_class=2, image_size=32, seed=7)
    assert ds[-1][1] == ds[len(ds) - 1][1]


def test_out_of_range_index_raises():
    ds = ShapesDataset(n_per_class=1, image_size=32, seed=7)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_dataset_rejects_bad_args():
    with pytest.raises(ValueError):
        ShapesDataset(n_per_class=0, image_size=32, seed=0)
    with pytest.raises(ValueError):
        ShapesDataset(n_per_class=1, image_size=0, seed=0)


# --------------------------------------------------------------------------- #
# Imagefolder tree
# --------------------------------------------------------------------------- #
def test_generate_imagefolder_tree_structure(tmp_path: Path):
    root = generate_shapes_dataset(tmp_path / "shapes", n_per_class=3, image_size=32, seed=7)
    assert root.is_dir()
    for shape_class in SHAPE_CLASSES:
        class_dir = root / shape_class
        assert class_dir.is_dir()
        pngs = sorted(class_dir.glob("*.png"))
        assert len(pngs) == 3
        assert all(p.name.startswith(shape_class) for p in pngs)


def test_disk_pixels_match_memory(tmp_path: Path):
    from PIL import Image

    root = generate_shapes_dataset(tmp_path / "shapes", n_per_class=1, image_size=32, seed=7)
    on_disk = np.asarray(Image.open(root / "circle" / "circle_00000.png"))
    in_memory = render_shape_image("circle", 32, index=0, seed=7)
    assert np.array_equal(on_disk, in_memory)


def test_generate_rejects_non_positive_count(tmp_path: Path):
    with pytest.raises(ValueError):
        generate_shapes_dataset(tmp_path / "shapes", n_per_class=0, image_size=32, seed=7)
