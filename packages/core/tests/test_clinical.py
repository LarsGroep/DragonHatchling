"""Clinical metrics (vitreous.clinical) — numpy-only, no torch, no sklearn.

Every expected value here is either hand-computed from a tiny confusion matrix
or a published Wilson-interval figure, so the tests fail if the arithmetic
drifts. Mirrors the M0 discipline of test_malignancy.py.

The centrepiece is ``test_ham10000_imbalance_trap``: it encodes the exact
mistake this module exists to fix — a constant "always predict nevi" model
scoring 0.669 accuracy while its balanced accuracy is 0.143, and uniform chance
(1/7) being reported as the baseline instead of the majority-class rate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vitreous.clinical import (
    Z_95,
    argmax_predictions,
    baselines,
    binary_metrics_at_threshold,
    class_recall,
    clinical_report,
    confusion_matrix,
    format_honest_summary,
    group_probability,
    melanoma_index,
    melanoma_sensitivity,
    multiclass_metrics,
    resolve_class_index,
    rate_with_ci,
    roc_auc,
    threshold_for_target_sensitivity,
    threshold_sweep,
    wilson_interval,
)

# --------------------------------------------------------------------------- #
# fixtures — hand-worked, so the asserted values are checkable on paper
# --------------------------------------------------------------------------- #

# A 3-class fixture. Confusion matrix (rows = true, cols = predicted):
#     [[2, 1, 1],
#      [1, 2, 0],
#      [0, 0, 2]]
# support = [4, 3, 2], predicted = [3, 3, 3], accuracy = 6/9.
HAND_TRUE = [0, 0, 0, 0, 1, 1, 1, 2, 2]
HAND_PRED = [0, 0, 1, 2, 1, 1, 0, 2, 2]
HAND_NAMES = ["nevus", "keratosis", "melanoma"]

# A binary fixture with a known operating point at threshold 0.5:
# tp=4, fn=1, fp=2, tn=8  →  sens 0.80, spec 0.80, ppv 4/6, npv 8/9.
BIN_POS = [True] * 5 + [False] * 10
BIN_SCORE = [0.9, 0.8, 0.7, 0.6, 0.1] + [0.6, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]

# Published HAM10000 class counts (10015 images, 67% melanocytic nevi).
HAM_COUNTS = {"nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514, "akiec": 327, "vasc": 142, "df": 115}


def _ham_labels():
    """y_true for the whole of HAM10000, class 0 = nv (the majority class)."""
    names = list(HAM_COUNTS)  # nv first
    y = np.concatenate([np.full(HAM_COUNTS[c], i, dtype=np.int64) for i, c in enumerate(names)])
    return y, names


# --------------------------------------------------------------------------- #
# (e) Wilson score intervals
# --------------------------------------------------------------------------- #


def test_wilson_known_values():
    # Published 95% Wilson intervals.
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))
    lo, hi = wilson_interval(8, 10)
    assert (lo, hi) == (pytest.approx(0.4902, abs=1e-4), pytest.approx(0.9434, abs=1e-4))


def test_wilson_contains_point_estimate():
    for n in (5, 12, 111, 1000):
        for s in (0, 1, n // 3, n - 1, n):
            lo, hi = wilson_interval(s, n)
            assert lo <= s / n <= hi


def test_wilson_narrows_as_n_grows():
    widths = []
    for n in (10, 100, 1000, 10000):
        lo, hi = wilson_interval(n // 2, n)
        widths.append(hi - lo)
    assert widths == sorted(widths, reverse=True)
    assert widths[-1] < 0.05


def test_wilson_stays_in_unit_interval_at_extremes():
    for n in (1, 3, 12, 500):
        lo0, hi0 = wilson_interval(0, n)
        lo1, hi1 = wilson_interval(n, n)
        assert lo0 == 0.0 and 0.0 < hi0 <= 1.0
        assert hi1 == 1.0 and 0.0 <= lo1 < 1.0


def test_wilson_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)
    with pytest.raises(ValueError):
        wilson_interval(-1, 3)
    with pytest.raises(ValueError):
        wilson_interval(1, -3)
    with pytest.raises(ValueError):
        wilson_interval(1, 3, z=0.0)


def test_rate_with_ci_zero_support_is_none_not_nan():
    r = rate_with_ci(0, 0)
    assert r["value"] is None and r["lo"] is None and r["hi"] is None
    assert r["n"] == 0
    assert "no support" in r["undefined_reason"]
    # json-serializable as null, not NaN.
    assert json.loads(json.dumps(r))["value"] is None


def test_rate_with_ci_reports_level_and_support():
    r = rate_with_ci(10, 12)
    assert r["value"] == pytest.approx(10 / 12)
    assert r["n"] == 12 and r["successes"] == 10
    assert r["ci_level"] == pytest.approx(0.95, abs=1e-4)
    # A 12-example rate must be visibly uncertain — this is the dermatofibroma
    # case (115 images dataset-wide → ~12 in a 10% test split).
    assert r["hi"] - r["lo"] > 0.3


# --------------------------------------------------------------------------- #
# (a) baselines
# --------------------------------------------------------------------------- #


def test_baselines_hand_fixture():
    b = baselines(HAND_TRUE, class_names=HAND_NAMES)
    assert b["n"] == 9 and b["num_classes"] == 3
    assert b["uniform_chance"] == pytest.approx(1 / 3)
    assert b["majority_class_rate"] == pytest.approx(4 / 9)
    assert b["majority_class_index"] == 0
    assert b["majority_class_name"] == "nevus"
    assert b["support"] == [4, 3, 2]
    assert b["prevalence"] == pytest.approx([4 / 9, 3 / 9, 2 / 9])


def test_baselines_reports_every_reference_point_together():
    b = baselines(HAND_TRUE)
    for key in ("uniform_chance", "majority_class_rate", "prevalence", "support"):
        assert key in b, f"a caller must not be able to quote only one baseline: {key} missing"


def test_baselines_empty_input_raises():
    with pytest.raises(ValueError):
        baselines([])


def test_baselines_rejects_out_of_range_label():
    with pytest.raises(ValueError):
        baselines([0, 1, 5], num_classes=3)


# --------------------------------------------------------------------------- #
# (b) multi-class metrics — hand-computed
# --------------------------------------------------------------------------- #


def test_confusion_matrix_hand_fixture():
    cm = confusion_matrix(HAND_TRUE, HAND_PRED)
    assert cm.tolist() == [[2, 1, 1], [1, 2, 0], [0, 0, 2]]


def test_multiclass_metrics_hand_fixture():
    m = multiclass_metrics(HAND_TRUE, HAND_PRED, class_names=HAND_NAMES)
    assert m["confusion_matrix"] == [[2, 1, 1], [1, 2, 0], [0, 0, 2]]
    assert m["accuracy"]["value"] == pytest.approx(6 / 9)
    assert m["accuracy"]["n"] == 9

    # recall = 2/4, 2/3, 2/2 ; balanced accuracy = mean of those.
    rec = [c["recall"]["value"] for c in m["per_class"]]
    assert rec == pytest.approx([0.5, 2 / 3, 1.0])
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert m["balanced_accuracy_classes_counted"] == 3

    # precision: every column sums to 3, diagonal 2 → 2/3 each.
    prec = [c["precision"]["value"] for c in m["per_class"]]
    assert prec == pytest.approx([2 / 3, 2 / 3, 2 / 3])

    # specificity (one-vs-rest): 4/5, 5/6, 6/7.
    spec = [c["specificity"]["value"] for c in m["per_class"]]
    assert spec == pytest.approx([4 / 5, 5 / 6, 6 / 7])

    # f1 = 4/7, 2/3, 0.8 ; macro f1 = their mean.
    f1 = [c["f1"] for c in m["per_class"]]
    assert f1 == pytest.approx([4 / 7, 2 / 3, 0.8])
    assert m["macro_f1"] == pytest.approx(np.mean([4 / 7, 2 / 3, 0.8]))

    # per-class counts for class 0: tp=2, fn=2, fp=1, tn=4.
    c0 = m["per_class"][0]["counts"]
    assert c0 == {"tp": 2, "fp": 1, "tn": 4, "fn": 2}
    assert m["per_class"][0]["support"] == 4
    assert m["per_class"][0]["predicted"] == 3


def test_multiclass_accuracy_always_ships_with_baselines():
    m = multiclass_metrics(HAND_TRUE, HAND_PRED, class_names=HAND_NAMES)
    assert m["baselines"]["majority_class_rate"] == pytest.approx(4 / 9)
    assert m["baselines"]["uniform_chance"] == pytest.approx(1 / 3)


def test_multiclass_zero_support_class_is_none_not_nan():
    # class 2 never appears as a true label and is never predicted.
    m = multiclass_metrics([0, 0, 1], [0, 1, 1], num_classes=3, class_names=HAND_NAMES)
    c2 = m["per_class"][2]
    assert c2["recall"]["value"] is None
    assert "no true examples" in c2["recall"]["undefined_reason"]
    assert c2["precision"]["value"] is None
    assert "never predicted" in c2["precision"]["undefined_reason"]
    assert c2["f1"] is None
    # balanced accuracy skips it and says so.
    assert m["balanced_accuracy_classes_counted"] == 2
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert not any(isinstance(v, float) and np.isnan(v) for v in [m["balanced_accuracy"]])


def test_multiclass_single_class_present():
    m = multiclass_metrics([0, 0, 0], [0, 0, 0], num_classes=2)
    assert m["accuracy"]["value"] == 1.0
    assert m["per_class"][0]["specificity"]["value"] is None  # no negatives
    assert "every example is" in m["per_class"][0]["specificity"]["undefined_reason"]
    assert m["balanced_accuracy"] == pytest.approx(1.0)


def test_multiclass_length_mismatch_raises():
    with pytest.raises(ValueError):
        multiclass_metrics([0, 1, 2], [0, 1])


def test_multiclass_empty_raises():
    with pytest.raises(ValueError):
        multiclass_metrics([], [])


# --------------------------------------------------------------------------- #
# THE TRAP — the exact mistake this module exists to fix
# --------------------------------------------------------------------------- #


def test_ham10000_imbalance_trap():
    """A do-nothing model scores 0.669 accuracy and 0.143 balanced accuracy.

    HAM10000 is ~67% melanocytic nevi. `webapp/graph.json` and the SGP bundle
    reported accuracy against ``"chance": 0.14285`` (uniform 1/7); this test
    pins down why that framing is misleading.
    """
    y, names = _ham_labels()
    assert y.size == 10015
    always_nv = np.zeros_like(y)  # the constant majority-class predictor

    b = baselines(y, class_names=names)
    assert b["num_classes"] == 7
    assert b["uniform_chance"] == pytest.approx(1 / 7, abs=1e-9)
    assert b["majority_class_rate"] == pytest.approx(0.6695, abs=1e-3)
    assert b["majority_class_name"] == "nv"
    assert b["imbalanced"] is True
    assert b["meaningful_floor"] == "majority_class_rate"

    m = multiclass_metrics(y, always_nv, class_names=names)
    # Knows nothing, yet "beats" uniform chance by 4.7x.
    assert m["accuracy"]["value"] == pytest.approx(0.6695, abs=1e-3)
    assert m["balanced_accuracy"] == pytest.approx(1 / 7, abs=1e-9)
    assert m["per_class"][names.index("mel")]["recall"]["value"] == 0.0

    # The published number: 0.7922 "vs chance 0.14285". Against the honest floor
    # the real margin is ~0.12, not ~0.65 — a >4x inflation.
    published = 0.7922
    honest_margin = published - b["majority_class_rate"]
    inflated_margin = published - b["uniform_chance"]
    assert honest_margin == pytest.approx(0.1227, abs=1e-3)
    assert inflated_margin / honest_margin > 4.0


# --------------------------------------------------------------------------- #
# (c) binary readout, sweep, AUC
# --------------------------------------------------------------------------- #


def test_binary_metrics_hand_fixture():
    r = binary_metrics_at_threshold(BIN_POS, BIN_SCORE, 0.5, label="melanoma")
    assert r["counts"] == {"tp": 4, "fp": 2, "tn": 8, "fn": 1}
    assert r["n"] == 15 and r["n_positive"] == 5 and r["n_negative"] == 10
    assert r["sensitivity"]["value"] == pytest.approx(0.8)
    assert r["sensitivity"]["n"] == 5
    assert r["specificity"]["value"] == pytest.approx(0.8)
    assert r["specificity"]["n"] == 10
    assert r["ppv"]["value"] == pytest.approx(4 / 6)
    assert r["npv"]["value"] == pytest.approx(8 / 9)
    assert r["accuracy"]["value"] == pytest.approx(12 / 15)
    assert r["youden_j"] == pytest.approx(0.6)
    assert r["prevalence"] == pytest.approx(5 / 15)
    assert r["flagged_fraction"] == pytest.approx(6 / 15)
    assert r["decision_rule"] == "score >= threshold"
    assert r["label"] == "melanoma"


def test_binary_threshold_is_inclusive():
    # A score exactly at the threshold is flagged (>=, not >).
    r = binary_metrics_at_threshold([True], [0.2], 0.2)
    assert r["counts"]["tp"] == 1


def test_binary_perfect_classifier_all_rates_one():
    y = [True] * 5 + [False] * 5
    s = [0.9] * 5 + [0.1] * 5
    r = binary_metrics_at_threshold(y, s, 0.5)
    for key in ("sensitivity", "specificity", "ppv", "npv", "accuracy"):
        assert r[key]["value"] == 1.0
        assert r[key]["hi"] == 1.0
    assert roc_auc(y, s) == 1.0


def test_binary_undefined_rates_are_none():
    # no positives → sensitivity undefined; nothing flagged → ppv undefined.
    r = binary_metrics_at_threshold([False, False], [0.1, 0.2], 0.9)
    assert r["sensitivity"]["value"] is None
    assert "no true" in r["sensitivity"]["undefined_reason"]
    assert r["ppv"]["value"] is None
    assert "nothing flagged" in r["ppv"]["undefined_reason"]
    assert r["youden_j"] is None
    # all positives → specificity undefined.
    r2 = binary_metrics_at_threshold([True, True], [0.1, 0.2], 0.05)
    assert r2["specificity"]["value"] is None
    assert r2["npv"]["value"] is None


def test_binary_length_mismatch_and_empty_raise():
    with pytest.raises(ValueError):
        binary_metrics_at_threshold([True, False], [0.1], 0.5)
    with pytest.raises(ValueError):
        binary_metrics_at_threshold([], [], 0.5)


def test_roc_auc_perfect_inverted_and_random():
    y = [True] * 5 + [False] * 5
    s = [0.9, 0.8, 0.7, 0.6, 0.55] + [0.5, 0.4, 0.3, 0.2, 0.1]
    assert roc_auc(y, s) == pytest.approx(1.0)
    assert roc_auc(y, [-v for v in s]) == pytest.approx(0.0)
    # a constant scorer is exactly 0.5 (all pairs tied) — not 0 or 1.
    assert roc_auc(y, [0.42] * 10) == pytest.approx(0.5)


def test_roc_auc_random_scores_near_half():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.3
    s = rng.random(4000)
    assert roc_auc(y, s) == pytest.approx(0.5, abs=0.03)


def test_roc_auc_tie_handling_matches_pairwise_definition():
    cases = [
        ([0, 1, 1, 0], [0.1, 0.1, 0.9, 0.9], 0.5),
        ([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.9], 0.875),
        ([0, 1], [0.5, 0.5], 0.5),
        # pos {0.7, 0.3} vs neg {0.7, 0.3, 0.3}: 2 wins, 3 ties, 1 loss → 3.5/6.
        ([1, 1, 0, 0, 0], [0.7, 0.3, 0.7, 0.3, 0.3], 3.5 / 6),
    ]
    for y, s, expected in cases:
        yb = np.asarray(y, dtype=bool)
        sa = np.asarray(s, dtype=float)
        # brute-force pairwise definition: win 1, tie 0.5, loss 0.
        pos, neg = sa[yb], sa[~yb]
        pairwise = float(
            sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
        ) / (pos.size * neg.size)
        assert roc_auc(y, s) == pytest.approx(expected)
        assert roc_auc(y, s) == pytest.approx(pairwise)


def test_roc_auc_single_class_returns_none():
    assert roc_auc([True, True, True], [0.1, 0.5, 0.9]) is None
    assert roc_auc([False, False], [0.1, 0.9]) is None


def test_threshold_sweep_is_monotone():
    rng = np.random.default_rng(1)
    y = rng.random(500) < 0.2
    s = np.clip(rng.random(500) * 0.6 + y * 0.35, 0, 1)
    sw = threshold_sweep(y, s, num_points=51)
    assert len(sw["points"]) == 51
    sens = [p["sensitivity"] for p in sw["points"]]
    spec = [p["specificity"] for p in sw["points"]]
    assert all(a >= b - 1e-12 for a, b in zip(sens, sens[1:])), "sensitivity must fall"
    assert all(a <= b + 1e-12 for a, b in zip(spec, spec[1:])), "specificity must rise"
    # endpoints: flag everything → sens 1, spec 0.
    assert sens[0] == pytest.approx(1.0)
    assert spec[0] == pytest.approx(0.0)
    assert sw["roc_auc"] > 0.5
    assert sw["n_positive"] + sw["n_negative"] == 500
    # every point carries its Wilson band for the UI.
    assert sw["points"][10]["sensitivity_lo"] <= sw["points"][10]["sensitivity"]
    assert sw["points"][10]["sensitivity_hi"] >= sw["points"][10]["sensitivity"]


def test_threshold_sweep_matches_pointwise_readout():
    sw = threshold_sweep(BIN_POS, BIN_SCORE, thresholds=[0.5])
    pt = sw["points"][0]
    r = binary_metrics_at_threshold(BIN_POS, BIN_SCORE, 0.5)
    assert (pt["tp"], pt["fp"], pt["tn"], pt["fn"]) == (4, 2, 8, 1)
    assert pt["sensitivity"] == pytest.approx(r["sensitivity"]["value"])
    assert pt["specificity"] == pytest.approx(r["specificity"]["value"])


def test_threshold_sweep_answers_what_02_buys():
    """The workbench's hardcoded 0.2 'high-sensitivity default', measured."""
    r = binary_metrics_at_threshold(BIN_POS, BIN_SCORE, 0.2, label="malignancy")
    # negatives at or above 0.2: 0.6, 0.55, 0.4, 0.3, 0.2 → fp = 5, tn = 5.
    assert r["counts"] == {"tp": 4, "fp": 5, "tn": 5, "fn": 1}
    assert r["sensitivity"]["value"] == pytest.approx(0.8)
    assert r["specificity"]["value"] == pytest.approx(0.5)
    # lowering 0.5 → 0.2 bought no sensitivity here and cost specificity.
    hi = binary_metrics_at_threshold(BIN_POS, BIN_SCORE, 0.5)
    assert r["sensitivity"]["value"] == hi["sensitivity"]["value"]
    assert r["specificity"]["value"] < hi["specificity"]["value"]


def test_target_sensitivity_lookup():
    pos = [0.5, 0.6, 0.7, 0.8, 0.9]
    neg = [0.65, 0.45, 0.3, 0.1, 0.05]
    y = [True] * 5 + [False] * 5
    s = pos + neg

    # 4/5 positives are >= 0.6, and 0.65 (a negative's score) only gets 3/5,
    # so the highest threshold reaching 0.8 sensitivity is exactly 0.6.
    r = threshold_for_target_sensitivity(y, s, 0.8)
    assert r["achieved"] is True
    assert r["threshold"] == pytest.approx(0.6)
    assert r["sensitivity"]["value"] == pytest.approx(0.8)
    assert r["target_sensitivity"] == 0.8

    # 100% sensitivity → the lowest positive score.
    r1 = threshold_for_target_sensitivity(y, s, 1.0)
    assert r1["threshold"] == pytest.approx(0.5)
    assert r1["sensitivity"]["value"] == pytest.approx(1.0)

    # It picks the *highest* qualifying threshold, i.e. the best specificity.
    assert r["specificity"]["value"] >= r1["specificity"]["value"]


def test_target_sensitivity_no_positives_is_explicit():
    r = threshold_for_target_sensitivity([False, False], [0.1, 0.9], 0.95)
    assert r["achieved"] is False
    assert r["threshold"] is None
    assert "no true" in r["reason"]


def test_target_sensitivity_rejects_bad_target():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            threshold_for_target_sensitivity(BIN_POS, BIN_SCORE, bad)


# --------------------------------------------------------------------------- #
# (d) named / melanoma-specific recall
# --------------------------------------------------------------------------- #


def test_class_recall_by_name_and_index():
    by_name = class_recall(HAND_TRUE, HAND_PRED, "melanoma", class_names=HAND_NAMES)
    by_index = class_recall(HAND_TRUE, HAND_PRED, 2, class_names=HAND_NAMES)
    assert by_name["value"] == pytest.approx(1.0)
    assert by_name["n"] == 2
    assert by_name["class"] == "melanoma" and by_name["index"] == 2
    assert by_index["value"] == by_name["value"]
    # matches the per-class array — the convenience path is not a second opinion.
    m = multiclass_metrics(HAND_TRUE, HAND_PRED, class_names=HAND_NAMES)
    assert m["per_class"][2]["recall"]["value"] == pytest.approx(by_name["value"])


def test_class_recall_carries_interval_on_tiny_support():
    # dermatofibroma-sized support: 10 of 12 caught.
    y = [0] * 12 + [1] * 100
    p = [0] * 10 + [1, 1] + [1] * 100
    r = class_recall(y, p, 0, class_names=["df", "nv"])
    assert r["value"] == pytest.approx(10 / 12)
    assert r["n"] == 12
    assert r["lo"] < 0.6 and r["hi"] > 0.94, "a 12-example rate must be reported as uncertain"


def test_melanoma_helpers():
    assert melanoma_index(["nv", "mel", "bkl"]) == 1
    assert melanoma_index(["Melanocytic nevi", "Melanoma"]) == 1
    assert melanoma_index(["cat", "dog"]) is None
    r = melanoma_sensitivity(HAND_TRUE, HAND_PRED, HAND_NAMES)
    assert r["class"] == "melanoma" and r["value"] == pytest.approx(1.0)
    assert melanoma_sensitivity([0, 1], [0, 1], ["cat", "dog"]) is None


def test_resolve_class_index_errors():
    with pytest.raises(ValueError):
        resolve_class_index(HAND_NAMES, "sarcoma")
    with pytest.raises(ValueError):
        resolve_class_index(HAND_NAMES, 7)


def test_class_recall_missing_class_is_none():
    r = class_recall([0, 0, 1], [0, 1, 1], 2, num_classes=3, class_names=HAND_NAMES)
    assert r["value"] is None
    assert "no true examples" in r["undefined_reason"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def test_argmax_predictions_and_group_probability():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    assert argmax_predictions(p).tolist() == [0, 2]
    assert group_probability(p, [1, 2]).tolist() == pytest.approx([0.3, 0.9])
    assert group_probability(p, []).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        group_probability(p, [5])
    with pytest.raises(ValueError):
        argmax_predictions([0.1, 0.2])  # not [N, K]


# --------------------------------------------------------------------------- #
# (f) the bundle report + the one-line honest summary
# --------------------------------------------------------------------------- #


def _ham_like_report(seed: int = 0):
    """A 7-class HAM10000-shaped fixture with a mediocre but real classifier."""
    rng = np.random.default_rng(seed)
    names = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
    counts = [670, 111, 110, 51, 33, 14, 11]
    y = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    logits = rng.standard_normal((y.size, 7)) * 0.8
    logits[np.arange(y.size), y] += 1.8  # informative but far from perfect
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = e / e.sum(axis=1, keepdims=True)
    return y, prob, names


def test_clinical_report_shape_and_summary():
    y, prob, names = _ham_like_report()
    rep = clinical_report(
        y,
        prob,
        class_names=names,
        positive_classes=["mel"],
        threshold=0.2,
        target_sensitivity=0.95,
        provenance={"split": "test", "model": "toy"},
    )
    assert rep["clinical_schema_version"] == 1
    assert rep["n"] == y.size and rep["num_classes"] == 7
    assert rep["baselines"]["majority_class_rate"] == pytest.approx(670 / y.size)
    assert rep["multiclass"]["balanced_accuracy"] is not None
    assert rep["binary"]["label"] == "mel"
    assert rep["binary"]["at_threshold"]["threshold"] == 0.2
    assert rep["binary"]["roc_auc"] > 0.5
    assert rep["binary"]["target_sensitivity_operating_point"]["achieved"] is True
    assert rep["focus_class"]["class"] == "mel"
    assert rep["provenance"]["split"] == "test"

    s = rep["summary"]
    assert "mel sensitivity" in s
    assert "95% CI" in s
    assert "at threshold 0.20" in s
    assert "balanced accuracy" in s
    assert "majority-class baseline" in s


def test_summary_never_shows_accuracy_without_the_majority_baseline():
    y, prob, names = _ham_like_report()
    rep = clinical_report(y, prob, class_names=names)
    s = format_honest_summary(rep)
    assert "accuracy" in s
    assert "majority-class baseline" in s
    # accuracy and its floor appear in the same clause, in that order.
    assert s.index("accuracy ") < s.index("majority-class baseline")


def test_summary_example_shape():
    # A fixture engineered to produce the documented sentence shape.
    y = np.array([0] * 60 + [1] * 40)
    prob = np.zeros((100, 2))
    prob[:60, 0] = 0.9
    prob[:60, 1] = 0.1
    prob[60:94, 1] = 0.9
    prob[60:94, 0] = 0.1
    prob[94:, 1] = 0.05
    prob[94:, 0] = 0.95
    rep = clinical_report(
        y, prob, class_names=["nevus", "melanoma"], positive_classes=["melanoma"],
        threshold=0.2,
    )
    s = rep["summary"]
    assert s.startswith("melanoma sensitivity 0.85 (95% CI ")
    assert "n=40)" in s
    assert "/ specificity 1.00 at threshold 0.20" in s


def test_summary_tolerates_missing_sections():
    assert format_honest_summary({}) == "no metrics available"
    assert "majority-class baseline" in format_honest_summary(
        {"baselines": baselines(HAND_TRUE)}
    )


def test_clinical_report_binary_group_defaults_to_malignancy_label():
    y, prob, names = _ham_like_report()
    rep = clinical_report(
        y, prob, class_names=names, positive_classes=["mel", "bcc", "akiec"], threshold=0.2
    )
    assert rep["binary"]["label"] == "malignancy"
    assert rep["binary"]["positive_classes"] == ["mel", "bcc", "akiec"]
    assert rep["binary"]["at_threshold"]["n_positive"] == 111 + 51 + 33


def test_clinical_report_rejects_shape_mismatch():
    y, prob, names = _ham_like_report()
    with pytest.raises(ValueError):
        clinical_report(y[:-1], prob, class_names=names)
    with pytest.raises(ValueError):
        clinical_report(y, prob[:, :3], class_names=names)


# --------------------------------------------------------------------------- #
# serialization + import discipline
# --------------------------------------------------------------------------- #


def _assert_json_native(obj, path="root"):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        assert not isinstance(obj, np.generic), f"numpy scalar leaked at {path}"
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
    y, prob, names = _ham_like_report()
    rep = clinical_report(
        y, prob, class_names=names, positive_classes=["mel"], threshold=0.2
    )
    for blob in (
        rep,
        baselines(y, class_names=names),
        multiclass_metrics(y, argmax_predictions(prob), class_names=names),
        binary_metrics_at_threshold(BIN_POS, BIN_SCORE, 0.5),
        threshold_sweep(BIN_POS, BIN_SCORE, num_points=11),
        threshold_for_target_sensitivity(BIN_POS, BIN_SCORE, 0.9),
        rate_with_ci(3, 12),
        class_recall(HAND_TRUE, HAND_PRED, 0, class_names=HAND_NAMES),
    ):
        _assert_json_native(blob)
        assert json.loads(json.dumps(blob)) == json.loads(json.dumps(blob))


def test_numpy_inputs_accepted():
    # numpy label arrays, numpy bool masks, numpy score arrays all work.
    y = np.asarray(HAND_TRUE, dtype=np.int32)
    p = np.asarray(HAND_PRED, dtype=np.int64)
    m = multiclass_metrics(y, p, class_names=HAND_NAMES)
    assert m["accuracy"]["value"] == pytest.approx(6 / 9)
    r = binary_metrics_at_threshold(
        np.asarray(BIN_POS), np.asarray(BIN_SCORE), np.float64(0.5)
    )
    assert r["counts"]["tp"] == 4


def test_module_does_not_import_torch_or_sklearn():
    import subprocess
    import sys

    code = (
        "import sys; import vitreous.clinical; "
        "print('torch' in sys.modules, 'sklearn' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False False", out.stdout


def test_z_constant_is_95_percent():
    assert rate_with_ci(1, 2, z=Z_95)["ci_level"] == pytest.approx(0.95, abs=1e-5)
