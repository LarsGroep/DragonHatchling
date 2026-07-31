"""Localization metrics (vitreous.localization) — numpy-only, no torch/sklearn/PIL.

The module measures *where* a model looked (CADe — detection with localization)
rather than *what* it concluded (CADx — diagnosis), so every expected value here
is hand-computed from a tiny fixture that can be checked on paper: two
overlapping squares, a central lesion, a one-pixel border ring.

Two tests carry most of the weight:

* ``test_planted_shortcut_is_caught`` — attribution mass parked on the image
  border while the lesion sits in the middle. This is the artifact-driven
  failure mode (rulers, ink, vignetting) that ``docs/UX-VISION.md`` lay
  question #2 exists to catch, and the shortcut detector must call it.
* ``test_best_over_threshold_flags_its_own_optimism`` — a best-over-threshold
  Dice is model selection on the test set, and the returned dict has to say so
  in a machine-readable field, not only in a docstring.

Mask PNG fixtures are written with stdlib ``zlib``/``struct``, mirroring
test_dermoscopy.py, so the suite never needs Pillow.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from vitreous.localization import (
    FROC_OPERATING_POINTS,
    LOCALIZATION_SCHEMA_VERSION,
    OPTIMISTIC_THRESHOLD_NOTE,
    Z_95,
    as_binary_mask,
    as_saliency_map,
    assign_marks_to_lesions,
    attribution_mass,
    binarize,
    border_mask,
    dice,
    distance_to_mask,
    format_localization_summary,
    froc_curve,
    iou,
    load_binary_mask,
    localization_report,
    mass_within_mask,
    mass_within_mask_dataset,
    normalize_map,
    overlap_at_threshold,
    overlap_dataset,
    overlap_threshold_sweep,
    peak_location,
    pointing_game,
    pointing_game_hit,
    resize_map,
    shortcut_report,
)

# --------------------------------------------------------------------------- #
# hand-worked fixtures
# --------------------------------------------------------------------------- #

H = W = 10


def _square(r0, r1, c0, c1, *, shape=(H, W)):
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


# Two overlapping 4x4 squares in a 10x10 frame:
#   A = rows/cols 2..5, B = rows/cols 4..7, overlap = rows/cols 4..5 (2x2 = 4 px).
#   |A| = |B| = 16, |A n B| = 4, |A u B| = 28  ->  IoU = 4/28 = 1/7, Dice = 8/32 = 1/4.
SQUARE_A = _square(2, 6, 2, 6)
SQUARE_B = _square(4, 8, 4, 8)

#: A central lesion (2x2 = 4 px of 100) — small enough that "mass inside" is a
#: real claim rather than an artifact of the lesion filling the frame.
LESION = _square(4, 6, 4, 6)


def _border_ring(shape=(H, W)):
    """The one-pixel frame border_mask() produces at border_fraction=0.1."""
    return border_mask(shape, border_fraction=0.1)


# --------------------------------------------------------------------------- #
# (a) overlap — hand-computed on two overlapping squares
# --------------------------------------------------------------------------- #


def test_iou_and_dice_hand_computed():
    assert iou(SQUARE_A, SQUARE_B) == pytest.approx(4 / 28)
    assert dice(SQUARE_A, SQUARE_B) == pytest.approx(8 / 32)
    # The algebraic identity between them, checked rather than assumed.
    j = iou(SQUARE_A, SQUARE_B)
    assert dice(SQUARE_A, SQUARE_B) == pytest.approx(2 * j / (1 + j))
    # Symmetric.
    assert iou(SQUARE_B, SQUARE_A) == pytest.approx(4 / 28)


def test_overlap_at_threshold_hand_computed():
    # saliency is 1.0 on square A, 0 elsewhere; ground truth is square B.
    s = SQUARE_A.astype(np.float64)
    r = overlap_at_threshold(s, SQUARE_B, 0.5)
    assert r["counts"] == {"tp": 4, "fp": 12, "fn": 12, "tn": 72}
    assert r["n_pixels"] == 100
    assert r["predicted_pixels"] == 16 and r["true_pixels"] == 16
    assert r["iou"] == pytest.approx(4 / 28)
    assert r["dice"] == pytest.approx(8 / 32)
    assert r["precision"]["value"] == pytest.approx(4 / 16)
    assert r["recall"]["value"] == pytest.approx(4 / 16)
    assert r["precision"]["n"] == 16 and r["recall"]["n"] == 16
    assert r["decision_rule"] == "pixel is predicted iff saliency >= threshold"
    assert r["empty_ground_truth"] is False


def test_pixel_rates_carry_the_correlation_caveat():
    """A pixel-level Wilson interval is optimistic and must say so."""
    r = overlap_at_threshold(SQUARE_A.astype(float), SQUARE_B, 0.5)
    for key in ("precision", "recall"):
        assert "spatially correlated" in r[key]["interval_caveat"]
    assert "spatially correlated" in r["interval_caveat"]


def test_perfect_prediction_scores_one_everywhere():
    s = LESION.astype(np.float64)
    r = overlap_at_threshold(s, LESION, 0.5)
    assert r["iou"] == 1.0 and r["dice"] == 1.0
    assert r["precision"]["value"] == 1.0 and r["recall"]["value"] == 1.0
    assert r["counts"]["fp"] == 0 and r["counts"]["fn"] == 0

    pg = pointing_game([s], [LESION])
    assert pg["hit_rate"]["value"] == 1.0
    assert pg["hit_rate"]["n"] == 1

    m = mass_within_mask(s, LESION)
    assert m["fraction_inside"] == 1.0
    # 4 px of 100 carrying all the mass = 25x what its area would get by chance.
    assert m["concentration"] == pytest.approx(25.0)


def test_disjoint_prediction_scores_zero_and_misses():
    left = _square(0, 3, 0, 3)
    right = _square(7, 10, 7, 10)
    assert iou(left, right) == 0.0
    assert dice(left, right) == 0.0

    s = left.astype(np.float64)
    r = overlap_at_threshold(s, right, 0.5)
    assert r["iou"] == 0.0 and r["dice"] == 0.0
    assert r["counts"]["tp"] == 0

    hit = pointing_game_hit(s, right)
    assert hit["hit"] is False
    assert hit["distance"] > 0
    assert pointing_game([s], [right])["hit_rate"]["value"] == 0.0

    m = mass_within_mask(s, right)
    assert m["fraction_inside"] == 0.0
    assert m["concentration"] == 0.0


def test_shape_mismatch_raises_and_points_at_resize_map():
    grid = np.ones((14, 14))
    mask = np.zeros((224, 224), dtype=bool)
    with pytest.raises(ValueError) as exc:
        overlap_at_threshold(grid, mask, 0.5)
    msg = str(exc.value)
    assert "resize_map" in msg
    assert "(14, 14)" in msg and "(224, 224)" in msg
    for fn in (pointing_game_hit, mass_within_mask):
        with pytest.raises(ValueError):
            fn(grid, mask)


# --------------------------------------------------------------------------- #
# threshold sweep — and its own optimism
# --------------------------------------------------------------------------- #


def _ramp_saliency():
    """1.0 on the lesion, 0.4 on a ring around it, 0 elsewhere.

    Thresholding anywhere in (0.4, 1.0] recovers the lesion exactly, so the best
    achievable Dice is 1.0 and the lowest threshold attaining it is 0.5 on a
    linspace(0, 1, 11) grid.
    """
    s = np.zeros((H, W), dtype=np.float64)
    s[3:7, 3:7] = 0.4
    s[4:6, 4:6] = 1.0
    return s


def test_threshold_sweep_finds_the_argmax_threshold():
    sw = overlap_threshold_sweep(_ramp_saliency(), LESION, num_points=11)
    assert sw["n_thresholds"] == 11
    assert len(sw["points"]) == 11
    assert sw["best_dice"]["value"] == pytest.approx(1.0)
    assert sw["best_dice"]["threshold"] == pytest.approx(0.5)
    assert sw["best_iou"]["value"] == pytest.approx(1.0)
    assert sw["best_iou"]["threshold"] == pytest.approx(0.5)
    # the curve really does dip either side of the argmax
    at_zero = [p for p in sw["points"] if p["threshold"] == pytest.approx(0.0)][0]
    assert at_zero["dice"] < 1.0
    assert at_zero["predicted_fraction"] == 1.0  # >= 0 selects everything


def test_best_over_threshold_flags_its_own_optimism():
    """A best-over-threshold number is tuned on the data it is reported on."""
    sw = overlap_threshold_sweep(_ramp_saliency(), LESION, num_points=11)
    for block in (sw["best_dice"], sw["best_iou"]):
        assert block["threshold_selection"] == OPTIMISTIC_THRESHOLD_NOTE
        assert "optimistic" in block["threshold_selection"]
        assert "held-out" in block["warning"]
        assert "model selection on the test set" in block["warning"]
    assert sw["threshold_selection"] == OPTIMISTIC_THRESHOLD_NOTE

    ds = overlap_dataset([_ramp_saliency()] * 3, [LESION] * 3, num_points=11)
    assert ds["best_over_threshold"]["threshold_selection"] == OPTIMISTIC_THRESHOLD_NOTE
    assert ds["best_over_threshold"]["dice"]["value"] == pytest.approx(1.0)
    assert ds["best_over_threshold"]["dice"]["threshold"] == pytest.approx(0.5)
    # the fixed-threshold readout is explicitly NOT tuned, and says so
    assert "not tuned" in ds["at_threshold"]["threshold_selection"]


def test_overlap_dataset_means_and_spread():
    good = LESION.astype(np.float64)
    bad = _square(0, 2, 0, 2).astype(np.float64)
    ds = overlap_dataset([good, bad], [LESION, LESION], threshold=0.5, num_points=5)
    assert ds["n_images"] == 2
    assert ds["at_threshold"]["dice"]["n"] == 2
    assert ds["at_threshold"]["dice"]["mean"] == pytest.approx((1.0 + 0.0) / 2)
    assert ds["at_threshold"]["dice"]["min"] == 0.0
    assert ds["at_threshold"]["dice"]["max"] == 1.0
    # a mean of ratios gets spread, not a fake Wilson interval
    assert "no Wilson interval applies" in ds["at_threshold"]["dice"]["statistic"]
    assert "lo" not in ds["at_threshold"]["dice"]


# --------------------------------------------------------------------------- #
# (b) pointing game
# --------------------------------------------------------------------------- #


def test_pointing_game_peak_inside_and_outside():
    s = np.zeros((H, W))
    s[4, 4] = 1.0  # inside LESION (rows/cols 4..5)
    assert pointing_game_hit(s, LESION)["hit"] is True
    assert pointing_game_hit(s, LESION)["distance"] == 0.0
    assert pointing_game_hit(s, LESION)["peak_row"] == 4

    s2 = np.zeros((H, W))
    s2[0, 0] = 1.0
    miss = pointing_game_hit(s2, LESION)
    assert miss["hit"] is False
    # nearest annotated pixel is (4, 4): distance = sqrt(32).
    assert miss["distance"] == pytest.approx(np.sqrt(32))


def test_pointing_game_tolerance_variant():
    s = np.zeros((H, W))
    s[4, 7] = 1.0  # two columns right of the lesion's right edge (col 5)
    assert pointing_game_hit(s, LESION, tolerance=0.0)["hit"] is False
    assert pointing_game_hit(s, LESION, tolerance=1.0)["hit"] is False
    assert pointing_game_hit(s, LESION, tolerance=2.0)["hit"] is True
    assert pointing_game_hit(s, LESION, tolerance=2.0)["distance"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        pointing_game_hit(s, LESION, tolerance=-1.0)


def test_pointing_game_dataset_rate_has_wilson_and_n():
    inside = np.zeros((H, W))
    inside[4, 4] = 1.0
    outside = np.zeros((H, W))
    outside[0, 0] = 1.0
    maps = [inside] * 3 + [outside]
    pg = pointing_game(maps, [LESION] * 4)
    rate = pg["hit_rate"]
    assert rate["value"] == pytest.approx(0.75)
    assert rate["n"] == 4 and rate["successes"] == 3
    assert rate["lo"] <= rate["value"] <= rate["hi"]
    assert rate["ci_level"] == pytest.approx(0.95, abs=1e-4)
    # 4 images is a tiny sample and the interval must show it
    assert rate["hi"] - rate["lo"] > 0.5
    assert pg["n_images"] == 4 and pg["n_scored"] == 4
    assert pg["distance"]["n"] == 4
    assert "counts images, not pixels" in pg["note"]


def test_peak_location_is_deterministic_on_ties():
    flat = np.ones((4, 4))
    assert peak_location(flat) == (0, 0)  # row-major first maximum, never random
    two = np.zeros((4, 4))
    two[1, 2] = two[3, 0] = 5.0
    assert peak_location(two) == (1, 2)


# --------------------------------------------------------------------------- #
# (c) attribution mass + THE SHORTCUT TEST
# --------------------------------------------------------------------------- #


def test_planted_shortcut_is_caught():
    """A model reading the image border, not the lesion — the classic artifact.

    The lesion is a 2x2 block in the middle of the frame. The attribution puts
    1.0 on every pixel of the one-pixel border ring (rulers, ink, vignetting,
    hair sweeping in from the edge) and only 0.1 on each lesion pixel. Mass
    inside the lesion must come out near zero, mass in the distractor near one,
    and the detector must *say* it is distracted — this is lay question #2 of
    docs/UX-VISION.md, and a tool that reassured here would be worse than none.
    """
    ring = _border_ring()
    saliency = np.zeros((H, W), dtype=np.float64)
    saliency[ring] = 1.0
    saliency[LESION] = 0.1

    n_ring = int(ring.sum())
    assert n_ring == 36  # 10x10 minus the 8x8 interior
    total = 36 * 1.0 + 4 * 0.1

    inside = mass_within_mask(saliency, LESION)
    assert inside["fraction_inside"] == pytest.approx(0.4 / total)
    assert inside["fraction_inside"] < 0.02
    # and it is *below* chance for the lesion's area share, not merely small
    assert inside["concentration"] < 1.0

    report = shortcut_report(saliency, LESION)
    assert report["fraction_in_distractor"] == pytest.approx(36.0 / total)
    assert report["fraction_in_distractor"] > 0.98
    assert report["shortcut_score"] == pytest.approx(36.0 / total)
    assert report["flagged"] is True
    assert report["focused"] is False
    assert report["verdict"] == "distracted"
    assert "image border" in report["message"]
    assert "caution" in report["message"]

    # The same machinery must NOT cry wolf on a model that reads the lesion.
    focused_map = np.zeros((H, W), dtype=np.float64)
    focused_map[LESION] = 1.0
    focused_map[ring] = 0.01
    ok = shortcut_report(focused_map, LESION)
    assert ok["flagged"] is False
    assert ok["focused"] is True
    assert ok["verdict"] == "focused"
    assert "focused on the lesion" in ok["message"]
    assert ok["shortcut_score"] < 0.1


def test_shortcut_distractor_never_double_counts_the_lesion():
    """A lesion touching the frame must not be scored as its own distractor."""
    edge_lesion = _square(0, 2, 0, 2)  # in the corner, overlapping the border ring
    saliency = edge_lesion.astype(np.float64)
    r = shortcut_report(saliency, edge_lesion)
    assert r["distractor_overlap_pixels"] > 0
    assert r["distractor_excludes_lesion"] is True
    assert r["fraction_in_distractor"] == 0.0
    assert r["flagged"] is False


def test_shortcut_accepts_a_caller_supplied_distractor():
    ruler = _square(0, 10, 0, 2)  # a "ruler" down the left edge
    saliency = np.zeros((H, W))
    saliency[ruler] = 1.0
    r = shortcut_report(saliency, LESION, ruler)
    assert r["distractor_source"] == "caller-supplied"
    assert r["shortcut_score"] == pytest.approx(1.0)
    assert r["flagged"] is True
    assert "distractor region" in r["message"]


def test_mass_is_area_normalized_by_concentration():
    """A uniform map gets exactly its area share — concentration 1.0, not skill."""
    uniform = np.ones((H, W), dtype=np.float64)
    m = mass_within_mask(uniform, LESION)
    assert m["fraction_inside"] == pytest.approx(0.04)
    assert m["area_fraction"] == pytest.approx(0.04)
    assert m["concentration"] == pytest.approx(1.0)
    assert "chance" in m["concentration_meaning"]

    # A lesion filling most of the frame collects most of the mass for free.
    big = _square(0, 9, 0, 9)  # 81 of 100 px
    m2 = mass_within_mask(uniform, big)
    assert m2["fraction_inside"] == pytest.approx(0.81)
    assert m2["concentration"] == pytest.approx(1.0)


def test_negative_attribution_is_excluded_and_reported():
    s = np.full((H, W), -5.0)
    s[LESION] = 1.0
    m = attribution_mass(s, LESION)
    assert m["mass_total"] == pytest.approx(4.0)  # only the positive part
    assert m["fraction_inside"] == pytest.approx(1.0)
    assert m["negative_mass"] == pytest.approx(5.0 * 96)
    assert m["positive_only"] is True


def test_mass_with_no_positive_attribution_is_undefined():
    s = np.full((H, W), -1.0)
    m = mass_within_mask(s, LESION)
    assert m["fraction_inside"] is None
    assert m["concentration"] is None
    assert "no positive mass" in m["undefined_reason"]


def test_mass_dataset_pairs_the_mean_with_a_real_proportion():
    good = np.zeros((H, W))
    good[LESION] = 1.0
    bad = np.zeros((H, W))
    bad[_border_ring()] = 1.0
    ds = mass_within_mask_dataset([good, good, good, bad], [LESION] * 4)
    assert ds["fraction"]["n"] == 4
    assert ds["fraction"]["mean"] == pytest.approx(0.75)
    assert "no Wilson interval applies" in ds["fraction"]["statistic"]
    rate = ds["majority_inside_rate"]
    assert rate["value"] == pytest.approx(0.75)
    assert rate["n"] == 4 and rate["successes"] == 3
    assert rate["lo"] <= rate["value"] <= rate["hi"]
    assert ds["majority_bar"] == 0.5


# --------------------------------------------------------------------------- #
# empty ground truth — an explicit, documented case, never a crash or a zero
# --------------------------------------------------------------------------- #


EMPTY = np.zeros((H, W), dtype=bool)


def test_empty_ground_truth_overlap_is_explicit():
    s = LESION.astype(np.float64)
    r = overlap_at_threshold(s, EMPTY, 0.5)
    assert r["empty_ground_truth"] is True
    assert r["true_pixels"] == 0
    assert r["recall"]["value"] is None
    assert "lesion-free" in r["recall"]["undefined_reason"]
    # something was predicted, so IoU/Dice are a well-defined 0.0
    assert r["iou"] == 0.0 and r["dice"] == 0.0

    # nothing predicted AND nothing annotated -> 0/0, undefined (never 1.0)
    r2 = overlap_at_threshold(s, EMPTY, 5.0)
    assert r2["iou"] is None and r2["dice"] is None
    assert "0/0" in r2["undefined_reason"]
    assert "predicting nothing" in r2["undefined_reason"]
    assert iou(EMPTY, EMPTY) is None
    assert dice(EMPTY, EMPTY) is None


def test_empty_ground_truth_pointing_game_is_undefined_not_a_miss():
    s = LESION.astype(np.float64)
    hit = pointing_game_hit(s, EMPTY)
    assert hit["hit"] is None  # not False
    assert hit["empty_ground_truth"] is True
    assert "neither hit nor miss" in hit["undefined_reason"]
    assert hit["distance"] is None
    assert distance_to_mask((0, 0), EMPTY) is None

    # a dataset drops it from the denominator rather than scoring it 0
    pg = pointing_game([s, s], [LESION, EMPTY])
    assert pg["n_images"] == 2
    assert pg["n_scored"] == 1
    assert pg["n_empty_ground_truth"] == 1
    assert pg["hit_rate"]["value"] == 1.0
    assert pg["hit_rate"]["n"] == 1

    # every mask empty -> the rate itself is undefined, with a reason
    pg2 = pointing_game([s], [EMPTY])
    assert pg2["hit_rate"]["value"] is None
    assert "nothing annotated" in pg2["hit_rate"]["undefined_reason"]


def test_empty_ground_truth_mass_is_undefined_not_zero():
    s = LESION.astype(np.float64)
    m = mass_within_mask(s, EMPTY)
    # None, not 0.0: "no mass landed on the lesion" and "there is no lesion" are
    # different statements and only one of them is true here.
    assert m["fraction_inside"] is None
    assert m["concentration"] is None
    assert m["empty_region"] is True
    assert "lesion-free" in m["undefined_reason"]
    assert m["mass_total"] > 0  # the map itself is intact and still reported

    r = shortcut_report(s, EMPTY)
    assert r["verdict"] == "not applicable"
    assert r["flagged"] is False
    assert "nothing to be focused on" in r["message"]


def test_report_survives_an_all_empty_dataset():
    s = LESION.astype(np.float64)
    rep = localization_report([s, s], [EMPTY, EMPTY])
    assert rep["n_empty_ground_truth"] == 2
    assert rep["pointing_game"]["hit_rate"]["value"] is None
    assert rep["shortcut"]["flag_rate"]["value"] is None
    assert rep["summary"]  # a sentence, not an exception
    json.dumps(rep)


# --------------------------------------------------------------------------- #
# resize_map + friends
# --------------------------------------------------------------------------- #


def test_resize_map_nearest_2x2_to_4x4_element_wise():
    src = np.array([[0.0, 1.0], [2.0, 3.0]])
    out = resize_map(src, (4, 4))
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [2.0, 2.0, 3.0, 3.0],
        ]
    )
    assert out.shape == (4, 4)
    assert np.array_equal(out, expected)
    # nearest never invents a value that was not in the source
    assert set(np.unique(out).tolist()) <= set(np.unique(src).tolist())


def test_resize_map_patch_grid_to_pixels():
    """The case this helper exists for: a [14,14] ViT patch grid -> [224,224]."""
    grid = np.arange(196, dtype=np.float64).reshape(14, 14)
    up = resize_map(grid, (224, 224))
    assert up.shape == (224, 224)
    # each patch becomes an exact 16x16 block
    assert np.all(up[:16, :16] == 0.0)
    assert np.all(up[16:32, :16] == 14.0)
    assert np.all(up[-16:, -16:] == 195.0)


def test_resize_map_bilinear_known_values_and_downsample():
    src = np.array([[0.0, 1.0], [2.0, 3.0]])
    out = resize_map(src, (4, 4), mode="bilinear")
    # align_corners=False: sample centres at x = (i+0.5)/2 - 0.5 -> [-0.25, .25, .75, 1.25]
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(0.25)
    assert out[3, 3] == pytest.approx(3.0)
    assert out[1, 1] == pytest.approx(0.75)
    # downsampling works too: each output centre lands on source index
    # floor((i + 0.5) * 4 / 2) = 1, 3 — i.e. rows/cols 1 and 3 are sampled.
    down = resize_map(np.arange(16, dtype=float).reshape(4, 4), (2, 2))
    assert down.tolist() == [[5.0, 7.0], [13.0, 15.0]]


def test_resize_map_rejects_bad_arguments():
    with pytest.raises(ValueError):
        resize_map(np.ones((2, 2)), (0, 4))
    with pytest.raises(ValueError):
        resize_map(np.ones((2, 2)), (4,))
    with pytest.raises(ValueError):
        resize_map(np.ones((2, 2)), (4, 4), mode="bicubic")
    with pytest.raises(ValueError):
        resize_map(np.ones((2, 2, 2)), (4, 4))


def test_report_can_resize_and_records_that_it_did():
    grid = np.zeros((5, 5))
    grid[2, 2] = 1.0
    mask = _square(4, 6, 4, 6)
    with pytest.raises(ValueError):
        localization_report([grid], [mask])
    rep = localization_report([grid], [mask], resize="nearest")
    assert rep["provenance"]["resize"] == "nearest"
    assert rep["pointing_game"]["hit_rate"]["value"] == 1.0


def test_normalize_map_and_constant_map():
    m = normalize_map(np.array([[2.0, 4.0], [6.0, 10.0]]))
    assert m.min() == 0.0 and m.max() == 1.0
    assert m[0, 1] == pytest.approx(0.25)
    flat = normalize_map(np.full((3, 3), 7.0))
    assert np.all(flat == 0.0)  # a constant map carries no ranking


def test_border_mask_geometry():
    b = border_mask((H, W), border_fraction=0.1)
    assert b.dtype == np.bool_
    assert int(b.sum()) == 100 - 64
    assert b[0, 0] and b[0, 5] and b[9, 9]
    assert not b[5, 5]
    with pytest.raises(ValueError):
        border_mask((H, W), border_fraction=0.9)


# --------------------------------------------------------------------------- #
# (d) FROC
# --------------------------------------------------------------------------- #


def test_froc_hand_computed_counting_rules():
    """3 images, 3 lesions, 5 marks — including a duplicate on lesion 1."""
    scores = [0.9, 0.8, 0.7, 0.6, 0.1]
    ids = [1, -1, 1, 2, -1]  # the 0.7 mark is a second hit on lesion 1
    f = froc_curve(scores, ids, n_images=3, n_lesions=3)
    assert f["n_marks"] == 5 and f["n_false_marks_total"] == 2
    by_thr = {round(p["threshold"], 3): p for p in f["points"]}

    # strictest first: one true mark, no false marks
    assert by_thr[0.9]["n_lesions_detected"] == 1
    assert by_thr[0.9]["false_marks_per_image"] == 0.0
    assert by_thr[0.9]["sensitivity"] == pytest.approx(1 / 3)
    # the duplicate hit on lesion 1 adds neither a detection nor a false mark
    assert by_thr[0.7]["n_marks_kept"] == 3
    assert by_thr[0.7]["n_lesions_detected"] == 1
    assert by_thr[0.7]["n_false_marks"] == 1
    # lesion 3 is never marked and stays in the denominator
    assert by_thr[0.1]["n_lesions_detected"] == 2
    assert by_thr[0.1]["sensitivity"] == pytest.approx(2 / 3)
    assert by_thr[0.1]["false_marks_per_image"] == pytest.approx(2 / 3)

    at = {p["fp_per_image"]: p for p in f["sensitivity_at_fp_per_image"]}
    assert at[0.125]["sensitivity"] == pytest.approx(1 / 3)  # only the fp-free point
    assert at[0.5]["sensitivity"] == pytest.approx(2 / 3)
    assert at[8.0]["sensitivity"] == pytest.approx(2 / 3)
    assert f["froc_score"] == pytest.approx((1 / 3 + 1 / 3 + 5 * 2 / 3) / 7)
    assert list(FROC_OPERATING_POINTS) == f["operating_points"]

    # sensitivity carries its Wilson interval over lesions
    assert by_thr[0.1]["sensitivity_lo"] <= by_thr[0.1]["sensitivity"]
    assert by_thr[0.1]["sensitivity_hi"] >= by_thr[0.1]["sensitivity"]
    assert by_thr[0.1]["sensitivity_n"] == 3
    assert "not independent" in f["interval_caveat"]
    assert "mark rather than" in f["why_froc"]


def test_froc_sensitivity_rises_as_false_marks_are_allowed():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0.5, 1.0, 20), rng.uniform(0.0, 0.7, 40)])
    ids = np.concatenate([np.arange(1, 21), np.full(40, -1)])
    f = froc_curve(scores, ids, n_images=10, n_lesions=25)
    fp = [p["false_marks_per_image"] for p in f["points"]]
    sens = [p["sensitivity"] for p in f["points"]]
    assert fp == sorted(fp), "points must run from strict to permissive"
    assert sens == sorted(sens), "sensitivity is non-decreasing in false marks"
    assert max(sens) == pytest.approx(20 / 25)  # 5 lesions were never marked


def test_froc_no_lesions_is_undefined_not_zero():
    f = froc_curve([0.5], [-1], n_images=1, n_lesions=0)
    assert f["points"][0]["sensitivity"] is None
    assert f["froc_score"] is None
    assert all(p["achieved"] is False for p in f["sensitivity_at_fp_per_image"])
    assert "no threshold" in f["sensitivity_at_fp_per_image"][0]["undefined_reason"]


def test_froc_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        froc_curve([0.5, 0.4], [-1], n_images=1, n_lesions=1)
    with pytest.raises(ValueError):
        froc_curve([0.5], [-1], n_images=0, n_lesions=1)


def test_assign_marks_to_lesions():
    labels = np.zeros((H, W), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    ids = assign_marks_to_lesions([(1, 1), (7, 7), (4, 0)], labels)
    assert ids == [1, 2, -1]
    # tolerance pulls a near-miss onto the nearest lesion
    assert assign_marks_to_lesions([(3, 1)], labels, tolerance=1.0) == [1]
    assert assign_marks_to_lesions([(3, 1)], labels, tolerance=0.0) == [-1]
    # a bool mask is a single-lesion label map
    assert assign_marks_to_lesions([(4, 4)], LESION) == [1]
    with pytest.raises(ValueError):
        assign_marks_to_lesions([(0, 0)], np.zeros((2, 2, 2)))


# --------------------------------------------------------------------------- #
# (e) the bundle report + the one-line summary
# --------------------------------------------------------------------------- #


def _dataset(n_focused=8, n_shortcut=2):
    """A dataset where most images are read correctly and a few are shortcuts."""
    ring = _border_ring()
    focused = np.zeros((H, W))
    focused[LESION] = 1.0
    focused[3:7, 3:7] = np.maximum(focused[3:7, 3:7], 0.2)
    shortcut = np.zeros((H, W))
    shortcut[ring] = 1.0
    shortcut[LESION] = 0.05
    maps = [focused] * n_focused + [shortcut] * n_shortcut
    return maps, [LESION] * len(maps)


def test_localization_report_shape_and_provenance():
    maps, masks = _dataset()
    rep = localization_report(
        maps,
        masks,
        threshold=0.5,
        tolerance=2.0,
        num_points=11,
        provenance={"split": "test", "method": "gradcam", "model": "toy"},
    )
    assert rep["localization_schema_version"] == LOCALIZATION_SCHEMA_VERSION
    assert rep["n_images"] == 10
    assert rep["n_empty_ground_truth"] == 0
    assert "CADe" in rep["task"] and "prevalence" in rep["task"]
    assert rep["mass_within_mask"]["fraction"]["n"] == 10
    assert rep["pointing_game"]["hit_rate"]["value"] == pytest.approx(0.8)
    assert rep["pointing_game"]["tolerance"] == 0.0
    assert rep["pointing_game_tolerant"]["tolerance"] == 2.0
    assert rep["shortcut"]["n_flagged"] == 2
    assert rep["shortcut"]["flag_rate"]["value"] == pytest.approx(0.2)
    assert rep["shortcut"]["flag_rate"]["n"] == 10
    assert rep["overlap"]["at_threshold"]["threshold"] == 0.5
    assert rep["froc"] is None
    assert rep["provenance"]["split"] == "test"
    assert rep["provenance"]["binarization_threshold"] == 0.5
    assert "ISIC 2018 Task 1" in rep["provenance"]["ground_truth_expected"]
    assert any("deletion_insertion" in c for c in rep["caveats"])
    assert any(OPTIMISTIC_THRESHOLD_NOTE in c for c in rep["caveats"])


def test_report_always_carries_the_strict_pointing_game_beside_the_tolerant_one():
    maps, masks = _dataset()
    rep = localization_report(maps, masks, tolerance=4.0, num_points=5)
    assert rep["pointing_game"]["tolerance"] == 0.0
    assert rep["pointing_game_tolerant"]["tolerance"] == 4.0
    strict = rep["pointing_game"]["hit_rate"]["value"]
    tolerant = rep["pointing_game_tolerant"]["hit_rate"]["value"]
    assert tolerant >= strict, "a tolerance can only ever add hits"


def test_report_includes_froc_when_marks_are_given():
    maps, masks = _dataset(2, 0)
    rep = localization_report(
        maps,
        masks,
        num_points=5,
        marks={
            "mark_scores": [0.9, 0.4],
            "mark_lesion_ids": [1, -1],
            "n_images": 2,
            "n_lesions": 2,
        },
    )
    assert rep["froc"]["n_lesions"] == 2
    assert rep["froc"]["points"][0]["sensitivity"] == pytest.approx(0.5)
    assert "FROC score" in rep["summary"]
    with pytest.raises(ValueError):
        localization_report(maps, masks, marks={"mark_scores": [0.9]})


def test_summary_pairs_mass_with_area_and_flags_tuned_thresholds():
    maps, masks = _dataset()
    rep = localization_report(maps, masks, threshold=0.5, num_points=11)
    s = rep["summary"]
    assert "attribution mass inside lesion" in s
    # a mass fraction is never quoted without its area share and chance multiple
    assert "lesion area" in s and "chance" in s
    assert "majority-inside on" in s and "95% CI" in s and "n=10" in s
    assert "pointing-game hit rate" in s
    assert "border-distracted on" in s
    # the tuned Dice is present but explicitly marked optimistic
    assert "at fixed threshold" in s
    assert "best achievable" in s
    assert OPTIMISTIC_THRESHOLD_NOTE in s
    assert s.index("at fixed threshold") < s.index("best achievable")


def test_summary_tolerates_missing_sections():
    assert format_localization_summary({}) == "no localization metrics available"
    assert format_localization_summary({"overlap": {}}) == "no localization metrics available"


def test_report_rejects_misaligned_inputs():
    maps, masks = _dataset(2, 0)
    with pytest.raises(ValueError):
        localization_report(maps, masks[:1])
    with pytest.raises(ValueError):
        localization_report([], [])
    with pytest.raises(ValueError):
        localization_report(maps, masks, distractor_masks=[LESION])


# --------------------------------------------------------------------------- #
# coercion, NaN discipline, mask loading
# --------------------------------------------------------------------------- #


def test_saliency_rejects_nan_and_bad_shapes():
    bad = np.zeros((H, W))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        as_saliency_map(bad)
    with pytest.raises(ValueError):
        as_saliency_map(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        as_saliency_map(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        mass_within_mask(bad, LESION)


def test_as_binary_mask_presence_rule_matches_dermoscopy():
    """0/255 PNG masks: any non-zero pixel is annotated (threshold=0, strict >)."""
    raw = np.zeros((4, 4), dtype=np.uint8)
    raw[1, 1] = 255
    raw[2, 2] = 1
    m = as_binary_mask(raw)
    assert m.dtype == np.bool_
    assert int(m.sum()) == 2
    assert as_binary_mask(np.ones((2, 2), dtype=bool)).all()
    assert int(as_binary_mask(raw, threshold=100).sum()) == 1


def test_binarize_is_inclusive():
    s = np.array([[0.2, 0.5], [0.8, 0.0]])
    assert binarize(s, 0.5).tolist() == [[False, True], [True, False]]
    assert binarize(s, 0.0).all()  # threshold at the floor selects everything


# --- stdlib PNG fixture writer (mirrors test_dermoscopy.py; no Pillow) ------- #


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, rows) -> Path:
    height, width = len(rows), len(rows[0])
    raw = bytearray()
    for r in rows:
        raw.append(0)  # filter type 0
        raw.extend(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    return path


def test_load_binary_mask_reads_an_isic_style_png(tmp_path):
    """ISIC Task 1/2 masks are 8-bit 0/255 PNGs; the decoder is shared with dermoscopy."""
    rows = [[0] * 6 for _ in range(6)]
    for y in (2, 3):
        for x in (2, 3):
            rows[y][x] = 255
    p = _write_png(tmp_path / "ISIC_0000000_segmentation.png", rows)
    m = load_binary_mask(p)
    assert m.shape == (6, 6)
    assert m.dtype == np.bool_
    assert int(m.sum()) == 4
    assert m[2, 2] and not m[0, 0]
    # and it drops straight into the metrics
    assert iou(m, m) == 1.0


# --------------------------------------------------------------------------- #
# serialization, determinism, import discipline
# --------------------------------------------------------------------------- #


def _assert_json_native(obj, path="root"):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        assert not isinstance(obj, np.generic), f"numpy scalar leaked at {path}"
        if isinstance(obj, float):
            assert not np.isnan(obj), f"NaN leaked at {path}"
            assert np.isfinite(obj), f"inf leaked at {path}"
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), f"non-str key at {path}"
            _assert_json_native(v, f"{path}.{k}")
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_json_native(v, f"{path}[{i}]")
        return
    raise AssertionError(f"non-JSON type {type(obj)!r} at {path}")


def test_everything_is_json_serializable():
    maps, masks = _dataset()
    s, g = maps[0], masks[0]
    blobs = [
        localization_report(
            maps,
            masks,
            threshold=0.5,
            tolerance=2.0,
            num_points=11,
            marks={
                "mark_scores": [0.9, 0.3],
                "mark_lesion_ids": [1, -1],
                "n_images": 10,
                "n_lesions": 10,
            },
        ),
        overlap_at_threshold(s, g, 0.5),
        overlap_at_threshold(s, EMPTY, 0.5),
        overlap_threshold_sweep(s, g, num_points=11),
        overlap_dataset(maps, masks, num_points=5),
        pointing_game_hit(s, g),
        pointing_game_hit(s, EMPTY),
        pointing_game(maps, masks, tolerance=1.0),
        attribution_mass(s, g),
        mass_within_mask(s, EMPTY),
        mass_within_mask_dataset(maps, masks),
        shortcut_report(s, g),
        froc_curve([0.9, 0.2], [1, -1], n_images=2, n_lesions=3),
    ]
    for blob in blobs:
        _assert_json_native(blob)
        assert json.loads(json.dumps(blob)) == json.loads(json.dumps(blob))


def test_numpy_inputs_accepted():
    maps, masks = _dataset(2, 1)
    stacked = np.stack(maps)
    stacked_masks = np.stack(masks)
    rep = localization_report(stacked, stacked_masks, num_points=5)
    assert rep["n_images"] == 3
    # integer masks (0/255) work the same as bool ones
    as_uint = (np.stack(masks) * 255).astype(np.uint8)
    rep2 = localization_report(stacked, as_uint, num_points=5)
    assert rep2["summary"] == rep["summary"]


def test_determinism():
    maps, masks = _dataset()
    a = localization_report(maps, masks, threshold=0.5, tolerance=2.0, num_points=11)
    b = localization_report(maps, masks, threshold=0.5, tolerance=2.0, num_points=11)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    # and the pieces, called repeatedly, do not drift
    for _ in range(3):
        assert pointing_game(maps, masks)["hit_rate"]["value"] == a["pointing_game"][
            "hit_rate"
        ]["value"]


def test_module_does_not_import_torch_sklearn_or_pillow():
    code = (
        "import sys; import vitreous.localization; "
        "print('torch' in sys.modules, 'sklearn' in sys.modules, 'PIL' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False False False", out.stdout


def test_z_constant_is_shared_with_clinical():
    from vitreous.clinical import Z_95 as CLINICAL_Z

    assert Z_95 == CLINICAL_Z
    maps, masks = _dataset(2, 0)
    pg = pointing_game(maps, masks, z=Z_95)
    assert pg["hit_rate"]["ci_level"] == pytest.approx(0.95, abs=1e-5)
