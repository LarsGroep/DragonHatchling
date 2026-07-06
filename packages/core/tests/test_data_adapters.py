"""Dataset adapter behavior — layout parsing, deterministic splits, registry.

All fixtures are synthesized on disk by ``make_synthetic_dataset`` (tiny
stdlib-written PNGs); nothing is downloaded and no image/ML library is needed.
"""

from __future__ import annotations

from collections import Counter

import pytest

from vitreous.data import (
    DatasetAdapter,
    EuroSATAdapter,
    ImageFolderAdapter,
    OxfordPetAdapter,
    Sample,
    SplitPolicy,
    deterministic_splits,
    get_dataset,
    list_datasets,
    make_synthetic_dataset,
    _register_builtins,
)

SPLITS = ("train", "val", "test")


@pytest.fixture(autouse=True)
def _ensure_builtins():
    # Other test modules clear the global registry; restore the built-ins so
    # registry lookups here are order-independent.
    _register_builtins()
    yield


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #


def test_builtin_datasets_registered():
    for name in ("eurosat", "oxford_pet", "imagefolder"):
        assert name in list_datasets()


@pytest.mark.parametrize(
    "name,cls",
    [
        ("eurosat", EuroSATAdapter),
        ("oxford_pet", OxfordPetAdapter),
        ("imagefolder", ImageFolderAdapter),
    ],
)
def test_registry_lookup_returns_class(name, cls):
    assert get_dataset(name) is cls
    assert issubclass(get_dataset(name), DatasetAdapter)


# --------------------------------------------------------------------------- #
# Shared assertions.
# --------------------------------------------------------------------------- #


def _assert_valid_samples(samples, expected_split=None):
    assert samples, "expected non-empty sample list"
    for s in samples:
        assert isinstance(s, Sample)
        assert isinstance(s.image, str) and s.image
        assert isinstance(s.label, int) and s.label >= 0
        assert s.split in SPLITS
        assert s.image_id
        assert "class_name" in s.meta
        if expected_split is not None:
            assert s.split == expected_split


def _labels_are_dense(samples, num_classes):
    labels = {s.label for s in samples}
    assert labels == set(range(num_classes))


# --------------------------------------------------------------------------- #
# EuroSAT / folder-per-class.
# --------------------------------------------------------------------------- #


def test_eurosat_num_classes_and_split_proportions(tmp_path):
    root = make_synthetic_dataset(
        str(tmp_path / "eurosat"), "eurosat", num_classes=3, per_class=10
    )
    adapter = EuroSATAdapter()
    all_samples = adapter.load(root, "all")

    assert len(all_samples) == 30
    _assert_valid_samples(all_samples)
    _labels_are_dense(all_samples, 3)

    counts = Counter(s.split for s in all_samples)
    # 80/10/10 of 10 per class → 8/1/1, stratified across 3 classes → 24/3/3.
    assert counts["train"] == 24
    assert counts["val"] == 3
    assert counts["test"] == 3

    # Per-split loads match the "all" filtering.
    for sp in SPLITS:
        sub = adapter.load(root, sp)
        _assert_valid_samples(sub, expected_split=sp)
        assert len(sub) == counts[sp]


def test_eurosat_split_is_deterministic_across_runs(tmp_path):
    root = make_synthetic_dataset(
        str(tmp_path / "eurosat"), "eurosat", num_classes=3, per_class=10
    )
    adapter = EuroSATAdapter()
    a = {s.image_id: s.split for s in adapter.load(root, "all")}
    b = {s.image_id: s.split for s in adapter.load(root, "all")}
    assert a == b
    # A fresh adapter instance yields the identical assignment.
    c = {s.image_id: s.split for s in EuroSATAdapter().load(root, "all")}
    assert a == c


def test_eurosat_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EuroSATAdapter().load(str(tmp_path / "does-not-exist"), "all")


# --------------------------------------------------------------------------- #
# Oxford-IIIT Pet.
# --------------------------------------------------------------------------- #


def test_oxford_pet_with_annotation_lists(tmp_path):
    root = make_synthetic_dataset(
        str(tmp_path / "pet"),
        "oxford_pet",
        num_classes=3,
        per_class=10,
        with_annotations=True,
    )
    adapter = OxfordPetAdapter()
    all_samples = adapter.load(root, "all")

    assert len(all_samples) == 30
    _assert_valid_samples(all_samples)
    _labels_are_dense(all_samples, 3)

    # Class name derived from filename stem, trailing _<n> stripped, underscores
    # in the class name preserved (class_a, class_b, ...).
    names = {s.meta["class_name"] for s in all_samples}
    assert names == {"class_a", "class_b", "class_c"}

    counts = Counter(s.split for s in all_samples)
    # One image/class placed in test by the fixture → 3 test; rest split
    # train/val by the seeded sub-policy over 9/class (~8:1 → 8 train, 1 val).
    assert counts["test"] == 3
    assert counts["train"] + counts["val"] == 27
    assert counts["train"] > 0 and counts["val"] > 0


def test_oxford_pet_fallback_filename_parsing(tmp_path):
    root = make_synthetic_dataset(
        str(tmp_path / "pet"),
        "oxford_pet",
        num_classes=3,
        per_class=10,
        with_annotations=False,
    )
    adapter = OxfordPetAdapter()
    all_samples = adapter.load(root, "all")
    assert len(all_samples) == 30
    _assert_valid_samples(all_samples)
    _labels_are_dense(all_samples, 3)
    counts = Counter(s.split for s in all_samples)
    assert counts["train"] == 24 and counts["val"] == 3 and counts["test"] == 3
    for s in all_samples:
        assert s.meta["list"] == "filename"


def test_oxford_pet_deterministic(tmp_path):
    root = make_synthetic_dataset(
        str(tmp_path / "pet"), "oxford_pet", num_classes=3, per_class=10
    )
    a = {s.image_id: s.split for s in OxfordPetAdapter().load(root, "all")}
    b = {s.image_id: s.split for s in OxfordPetAdapter().load(root, "all")}
    assert a == b


# --------------------------------------------------------------------------- #
# ImageFolder — flat and pre-split.
# --------------------------------------------------------------------------- #


def test_imagefolder_flat_seeded_split(tmp_path):
    root = make_synthetic_dataset(
        str(tmp_path / "flat"), "imagefolder_flat", num_classes=3, per_class=10
    )
    adapter = ImageFolderAdapter()
    all_samples = adapter.load(root, "all")
    assert len(all_samples) == 30
    _assert_valid_samples(all_samples)
    _labels_are_dense(all_samples, 3)
    counts = Counter(s.split for s in all_samples)
    assert counts["train"] == 24 and counts["val"] == 3 and counts["test"] == 3


def test_imagefolder_presplit_tree(tmp_path):
    root = make_synthetic_dataset(
        str(tmp_path / "split"), "imagefolder_split", num_classes=3, per_class=10
    )
    adapter = ImageFolderAdapter()
    all_samples = adapter.load(root, "all")
    _assert_valid_samples(all_samples)
    _labels_are_dense(all_samples, 3)

    counts = Counter(s.split for s in all_samples)
    # per_class 10 → 8/1/1 per split-dir per class × 3 classes.
    assert counts["train"] == 24
    assert counts["val"] == 3
    assert counts["test"] == 3

    train_only = adapter.load(root, "train")
    _assert_valid_samples(train_only, expected_split="train")
    assert len(train_only) == 24


def test_imagefolder_presplit_detected_over_flat(tmp_path):
    # With train/ and val/ present the pre-split branch is used (splits honored
    # exactly, not re-seeded).
    root = make_synthetic_dataset(
        str(tmp_path / "split"), "imagefolder_split", num_classes=2, per_class=10
    )
    val = ImageFolderAdapter().load(root, "val")
    assert all(s.split == "val" for s in val)


# --------------------------------------------------------------------------- #
# Split helper properties.
# --------------------------------------------------------------------------- #


def test_deterministic_splits_order_independent():
    policy = SplitPolicy(seed=7)
    keys = [f"img_{i}" for i in range(20)]
    a = deterministic_splits(keys, policy, salt="c")
    b = deterministic_splits(list(reversed(keys)), policy, salt="c")
    assert a == b


def test_deterministic_splits_seed_changes_assignment():
    keys = [f"img_{i}" for i in range(20)]
    a = deterministic_splits(keys, SplitPolicy(seed=1), salt="c")
    b = deterministic_splits(keys, SplitPolicy(seed=2), salt="c")
    assert a != b


def test_split_policy_validates_fractions():
    with pytest.raises(ValueError):
        SplitPolicy(fractions=(0.5, 0.4, 0.4))
    with pytest.raises(ValueError):
        SplitPolicy(fractions=(0.8, 0.2))  # wrong length
