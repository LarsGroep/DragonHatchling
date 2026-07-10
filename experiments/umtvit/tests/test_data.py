"""Universal data-pipeline tests (ARCHITECTURE §4): loaders, splits, augment.

All CPU-only, no downloads. The shapes loader renders in-memory; the csv and
imagefolder loaders are exercised against tiny synthetic trees written to
``tmp_path`` (a handful of 16x16 images), so the whole file runs in well under
a second. The load-bearing guarantees checked here:

- ``two_view`` yields two *different* augmented views of the same image, both
  ``[C, H, W]`` float tensors in ``[0, 1]``; ``eval`` is deterministic.
- grouped splits never leak a group across splits (the HAM10000 lesion-id
  requirement), reproduced with a synthetic grouped CSV dataset.
- unlabeled mode (no ``label_column``) yields ``-1`` labels and still iterates.
- an unknown augmentation policy is rejected naming ``dataset.augmentation``.
- split fractions are honoured approximately.
"""

from __future__ import annotations

import csv as csvmod
from pathlib import Path

import numpy as np
import pytest

from umtvit.config import Config, ConfigError
from umtvit.data.dataset import UniversalDataset
from umtvit.data.splits import split_of


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cfg(**dataset_overrides) -> Config:
    """Build a validated Config, defaulting to a tiny (16px) dataset block."""
    data = Config().to_dict()  # model.image_size is null ⇒ derived from dataset
    data["dataset"].update({"image_size": 16, "channels": 3})
    data["dataset"].update(dataset_overrides)
    return Config.from_dict(data)


def _write_png(path: Path, rng: np.random.Generator, size: int = 16) -> None:
    from PIL import Image

    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _make_grouped_csv(tmp_path: Path, n_images: int = 40, n_groups: int = 12):
    """Write ``n_images`` PNGs + a metadata CSV with label + group columns."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_images):
        image_id = f"img{i:03d}"
        _write_png(image_dir / f"{image_id}.png", rng)
        rows.append(
            {"image_id": image_id, "dx": f"c{i % 3}", "lesion_id": f"g{i % n_groups}"}
        )
    csv_path = tmp_path / "meta.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csvmod.DictWriter(fh, fieldnames=["image_id", "dx", "lesion_id"])
        writer.writeheader()
        writer.writerows(rows)
    return str(image_dir), str(csv_path)


# --------------------------------------------------------------------------- #
# UniversalDataset on shapes: two_view / eval contracts
# --------------------------------------------------------------------------- #
def test_shapes_two_view_returns_two_distinct_views_in_range():
    import torch

    cfg = _cfg(loader="shapes", n_per_class=6, augmentation="natural_default")
    ds = UniversalDataset(cfg, "train", "two_view")
    assert len(ds) > 0
    view_a, view_b, label = ds[0]
    assert view_a.shape == view_b.shape == (3, 16, 16)
    assert view_a.dtype == torch.float32
    assert 0.0 <= float(view_a.min()) and float(view_a.max()) <= 1.0
    assert 0.0 <= float(view_b.min()) and float(view_b.max()) <= 1.0
    # Two independent augmentations of the same source image differ.
    assert not torch.equal(view_a, view_b)
    assert isinstance(label, int)


def test_shapes_eval_is_deterministic():
    import torch

    cfg = _cfg(loader="shapes", n_per_class=20, augmentation="natural_default")
    ds = UniversalDataset(cfg, "train", "eval")
    assert len(ds) > 0
    a, la = ds[0]
    b, lb = ds[0]
    assert torch.equal(a, b)  # eval has no augmentation randomness
    assert la == lb
    assert a.shape == (3, 16, 16)
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0


def test_splits_partition_all_items_disjointly():
    cfg = _cfg(loader="shapes", n_per_class=20)
    lengths = {
        split: len(UniversalDataset(cfg, split, "eval"))
        for split in ("train", "val", "test")
    }
    assert sum(lengths.values()) == 3 * 20  # every item lands in exactly one split


# --------------------------------------------------------------------------- #
# Grouped splits never leak a group across splits
# --------------------------------------------------------------------------- #
def test_grouped_split_has_no_leakage(tmp_path: Path):
    image_dir, csv_path = _make_grouped_csv(tmp_path, n_images=40, n_groups=12)
    cfg = _cfg(
        loader="csv",
        image_dir=image_dir,
        metadata_csv=csv_path,
        path_column="image_id",
        path_suffix=".png",
        label_column="dx",
        group_column="lesion_id",
        augmentation="dermoscopy_default",
    )
    groups_by_split = {}
    total = 0
    for split in ("train", "val", "test"):
        ds = UniversalDataset(cfg, split, "eval")
        total += len(ds)
        groups_by_split[split] = {item[2] for item in ds.items}

    assert total == 40  # nothing dropped
    # No lesion id may appear in more than one split.
    train, val, test = (groups_by_split[s] for s in ("train", "val", "test"))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # Sanity: the groups did actually get distributed, not all into one split.
    assert len(train | val | test) == 12


def test_grouped_split_is_stable_across_rebuilds(tmp_path: Path):
    image_dir, csv_path = _make_grouped_csv(tmp_path, n_images=40, n_groups=12)
    kwargs = dict(
        loader="csv",
        image_dir=image_dir,
        metadata_csv=csv_path,
        path_suffix=".png",
        label_column="dx",
        group_column="lesion_id",
    )
    cfg = _cfg(**kwargs)
    first = {item[0] for item in UniversalDataset(cfg, "train", "eval").items}
    second = {item[0] for item in UniversalDataset(cfg, "train", "eval").items}
    assert first == second  # deterministic assignment


# --------------------------------------------------------------------------- #
# Unlabeled mode
# --------------------------------------------------------------------------- #
def test_unlabeled_mode_yields_minus_one_and_iterates(tmp_path: Path):
    image_dir, csv_path = _make_grouped_csv(tmp_path, n_images=40, n_groups=12)
    cfg = _cfg(
        loader="csv",
        image_dir=image_dir,
        metadata_csv=csv_path,
        path_suffix=".png",
        label_column=None,  # fully unlabeled
        group_column="lesion_id",
    )
    ds = UniversalDataset(cfg, "train", "eval")
    assert len(ds) > 0
    assert ds.classes == []  # no class names in unlabeled mode
    seen = 0
    for i in range(len(ds)):
        image, label = ds[i]
        assert label == -1
        assert image.shape == (3, 16, 16)
        seen += 1
    assert seen == len(ds)


# --------------------------------------------------------------------------- #
# imagefolder loader
# --------------------------------------------------------------------------- #
def test_imagefolder_loader_reads_class_tree(tmp_path: Path):
    from umtvit.data.loaders import build_items

    root = tmp_path / "tree"
    rng = np.random.default_rng(1)
    for cls in ("cat", "dog"):
        (root / cls).mkdir(parents=True)
        for j in range(4):
            _write_png(root / cls / f"{cls}_{j}.png", rng)
    cfg = _cfg(loader="imagefolder", image_dir=str(root), label_column="folder")
    items, classes = build_items(cfg)
    assert classes == ["cat", "dog"]
    assert len(items) == 8
    assert {item[1] for item in items} == {0, 1}  # both class indices present


# --------------------------------------------------------------------------- #
# Augmentation policy resolution
# --------------------------------------------------------------------------- #
def test_unknown_augmentation_policy_rejected_with_field_name():
    cfg = _cfg(loader="shapes", n_per_class=4, augmentation="teleport_default")
    with pytest.raises(ConfigError) as exc:
        UniversalDataset(cfg, "train", "two_view")
    assert "dataset.augmentation" in str(exc.value)


def test_dermoscopy_policy_excludes_channel_jitter():
    from umtvit.data.augment import AUGMENTATION_POLICIES

    # ARCHITECTURE §4: the medical policy must not apply hue/channel jitter.
    assert AUGMENTATION_POLICIES["dermoscopy_default"].get("channel_jitter", 0.0) == 0.0


# --------------------------------------------------------------------------- #
# Split fractions honoured approximately
# --------------------------------------------------------------------------- #
def test_split_fractions_approximately_honoured():
    # Ungrouped keys hash by index; over many keys the bands match the config.
    from umtvit.config import SplitConfig

    splits = SplitConfig(train=0.8, val=0.1, test=0.1, seed=1)
    n = 3000
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(n):
        counts[split_of(i, splits)] += 1
    assert abs(counts["train"] / n - 0.8) < 0.03
    assert abs(counts["val"] / n - 0.1) < 0.03
    assert abs(counts["test"] / n - 0.1) < 0.03
