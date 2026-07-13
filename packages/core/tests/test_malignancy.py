"""Malignancy lens core (vitreous.malignancy) + HAM10000 taxonomy — numpy-only.

Covers the derived softmax readouts, the ordinal category coordinate, the
label-free axis build + projection + OOD gate, and the HAM10000 grouping wired
onto the DatasetSpec. No torch; mirrors the M0 discipline of test_som.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from vitreous.data import Taxonomy, get_dataset
from vitreous.malignancy import (
    build_malignancy_axis,
    category_levels,
    expected_category,
    hard_category,
    malignant_indices,
    malignant_probability,
    project_feature,
)

# A tiny 4-class stand-in: two benign, one in-situ, one invasive.
CLASSES = ["nevus", "keratosis", "insitu", "melanoma"]
TAX = Taxonomy(
    malignant={"nevus": False, "keratosis": False, "insitu": True, "melanoma": True},
    category_level={"nevus": 0, "keratosis": 0, "insitu": 1, "melanoma": 2},
    category_labels=["benign", "in-situ", "invasive"],
)


# --------------------------------------------------------------------------- #
# derived readouts
# --------------------------------------------------------------------------- #


def test_malignant_indices_and_probability():
    idx = malignant_indices(CLASSES, TAX)
    assert idx == [2, 3]  # insitu, melanoma
    p = [0.6, 0.1, 0.1, 0.2]
    assert malignant_probability(p, idx) == pytest.approx(0.3)
    # empty malignant set → 0
    assert malignant_probability(p, []) == 0.0


def test_expected_and_hard_category():
    levels = category_levels(CLASSES, TAX)
    assert list(levels) == [0, 0, 1, 2]
    # all mass on melanoma → coordinate 2, hard level 2.
    assert expected_category([0, 0, 0, 1], levels) == pytest.approx(2.0)
    assert hard_category([0, 0, 0, 1], levels) == 2
    # split benign/melanoma → between 0 and 2, hard snaps to the argmax.
    assert expected_category([0.5, 0, 0, 0.5], levels) == pytest.approx(1.0)
    assert hard_category([0.6, 0, 0, 0.4], levels) == 0
    # a mostly in-situ lesion → coordinate near 1.
    assert expected_category([0.1, 0.1, 0.8, 0.0], levels) == pytest.approx(0.8)


def test_expected_category_zero_mass():
    assert expected_category([0, 0, 0, 0], category_levels(CLASSES, TAX)) == 0.0


# --------------------------------------------------------------------------- #
# manifold axis
# --------------------------------------------------------------------------- #


def _two_blobs(seed: int = 0, d: int = 16, n: int = 80):
    """Benign blob near -e0, malignant blob near +e0, separated along axis 0."""
    rng = np.random.default_rng(seed)
    benign = rng.standard_normal((n, d)) * 0.3
    benign[:, 0] -= 3.0
    malignant = rng.standard_normal((n, d)) * 0.3
    malignant[:, 0] += 3.0
    feats = np.vstack([benign, malignant])
    is_mal = np.array([False] * n + [True] * n)
    return feats, is_mal


def test_build_axis_recovers_separation_direction():
    feats, is_mal = _two_blobs()
    axis = build_malignancy_axis(feats, is_mal, provenance={"dataset": "toy"})
    assert axis["provider"] == "malignancy_axis"
    assert axis["dim"] == 16
    # the axis should point essentially along dim 0.
    u = np.asarray(axis["u"])
    assert abs(u[0]) > 0.95
    assert axis["provenance"]["n_benign"] == 80
    assert axis["provenance"]["n_malignant"] == 80


def test_projection_orders_benign_below_malignant():
    feats, is_mal = _two_blobs()
    axis = build_malignancy_axis(feats, is_mal)
    benign_pos = project_feature(feats[0], axis)["position"]
    malignant_pos = project_feature(feats[-1], axis)["position"]
    assert benign_pos < 0.2
    assert malignant_pos > 0.8
    # a mid lesion lands in the middle.
    mid = feats[:1].copy()[0]
    mid[0] = 0.0
    assert 0.2 < project_feature(mid, axis)["position"] < 0.8


def test_ood_flag_fires_off_axis():
    feats, is_mal = _two_blobs()
    axis = build_malignancy_axis(feats, is_mal)
    # An in-distribution point is not OOD.
    assert project_feature(feats[0], axis)["ood"] is False
    # A point flung far off the axis (orthogonal dims) is OOD.
    outlier = feats[0].copy()
    outlier[1:] += 25.0
    r = project_feature(outlier, axis)
    assert r["ood"] is True
    assert r["residual"] > axis["residual_threshold"]


def test_build_axis_needs_both_groups():
    feats = np.random.default_rng(0).standard_normal((10, 8))
    with pytest.raises(ValueError):
        build_malignancy_axis(feats, [False] * 10)  # no malignant


def test_project_dim_mismatch_raises():
    feats, is_mal = _two_blobs(d=8)
    axis = build_malignancy_axis(feats, is_mal)
    with pytest.raises(ValueError):
        project_feature(np.zeros(4), axis)


# --------------------------------------------------------------------------- #
# HAM10000 taxonomy wired onto the spec
# --------------------------------------------------------------------------- #


def test_ham10000_taxonomy_is_medically_correct():
    spec = get_dataset("ham10000").spec
    tax = spec.taxonomy
    assert tax is not None
    # malignant = melanoma + BCC + actinic keratosis (the correct grouping).
    mal = {name for name, is_mal in tax.malignant.items() if is_mal}
    assert mal == {"Melanoma", "Basal cell carcinoma", "Actinic keratoses"}
    # every class is covered.
    assert set(tax.malignant) == set(spec.class_names)
    # ordinal: nevi benign(0), actinic keratoses in-situ(1), melanoma invasive(2).
    assert tax.category_level["Melanocytic nevi"] == 0
    assert tax.category_level["Actinic keratoses"] == 1
    assert tax.category_level["Melanoma"] == 2
    assert tax.category_level["Basal cell carcinoma"] == 2
    assert tax.category_labels == ["benign", "in-situ", "invasive"]


def test_taxonomy_json_roundtrip():
    spec = get_dataset("ham10000").spec
    payload = spec.taxonomy.to_json()
    import json

    reparsed = json.loads(json.dumps(payload))
    assert reparsed["category_labels"] == ["benign", "in-situ", "invasive"]
    assert reparsed["malignant"]["Melanoma"] is True


def test_taxonomy_missing_class_rejected():
    from vitreous.data import DatasetSpec

    with pytest.raises(ValueError):
        DatasetSpec(
            name="x",
            display_name="x",
            num_classes=2,
            class_names=["a", "b"],
            taxonomy=Taxonomy(malignant={"a": True}),  # missing "b"
        )
