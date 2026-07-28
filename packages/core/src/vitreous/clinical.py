"""Clinical metrics — the honest readout of a cancer classifier (numpy-only).

This module exists because the project was reporting the *wrong numbers*. A
7-way HAM10000 accuracy of ``0.79`` was being quoted against ``"chance": 0.14285``
(uniform ``1/7``) while ~67% of the dataset is a single class (``nv``,
melanocytic nevi). On an imbalanced dataset the meaningful floor is the
**majority-class rate** (``≈ 0.669`` for HAM10000), not uniform chance; quoting
the latter inflates the apparent margin by more than 4x. Separately, the number
that actually matters clinically — **melanoma sensitivity at the chosen decision
threshold** — was never computed anywhere, even though the workbench ships a
hardcoded ``0.2`` threshold labelled "high-sensitivity default".

Everything here is measured from two arrays and nothing else:

- ``y_true``  ``[N]``     integer class labels;
- ``y_prob``  ``[N, K]``  per-class probabilities (or ``y_pred`` ``[N]`` labels).

What is provided:

1. :func:`baselines` — *all* the reference points at once (uniform chance,
   majority-class rate, per-class prevalence) so a caller cannot quote only the
   flattering one.
2. :func:`multiclass_metrics` — confusion matrix, per-class recall / precision /
   specificity / F1, overall accuracy, and **balanced accuracy**.
3. :func:`binary_metrics_at_threshold`, :func:`threshold_sweep`, :func:`roc_auc`,
   :func:`threshold_for_target_sensitivity` — the clinically load-bearing binary
   readout of a malignancy score at an operating point, plus the curve that
   answers "what does threshold 0.2 actually buy me".
4. :func:`class_recall` / :func:`melanoma_sensitivity` — a named class's recall
   as a first-class number, because a missed melanoma is the worst possible
   error (``docs/MALIGNANCY-LENS.md`` §8 states the project's intent to treat
   that cost asymmetrically).
5. :func:`wilson_interval` — every rate carries a Wilson score interval **and**
   its support count. With 115 dermatofibroma images in all of HAM10000, a
   per-class recall from a 10% test split rests on ~12 examples; a bare point
   estimate there is exactly the dishonesty this project's own rules forbid.
6. :func:`clinical_report` + :func:`format_honest_summary` — the whole thing as
   one JSON-serializable blob for bundle provenance, and the single plain
   sentence a UI or notebook should print.

**Import discipline (M0 rule):** numpy only. No torch, no scikit-learn. Every
metric — including ROC AUC, via the rank / Mann-Whitney identity with correct
tie handling — is implemented here directly.

**Serialization:** every returned value is a plain ``float`` / ``int`` / ``bool``
/ ``str`` / ``list`` / ``dict`` / ``None``. No numpy scalars leak out, so any
result can be dropped straight into a pack manifest's provenance and survive
``json.dumps``.

**Undefined vs NaN:** a rate with no support (specificity with zero negatives,
precision with zero predicted positives, recall of a class absent from the split)
is reported as ``None`` with an ``undefined_reason`` string — never ``NaN``,
never a silent ``0.0``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "CLINICAL_SCHEMA_VERSION",
    "Z_95",
    "MELANOMA_ALIASES",
    "wilson_interval",
    "rate_with_ci",
    "baselines",
    "confusion_matrix",
    "multiclass_metrics",
    "argmax_predictions",
    "group_probability",
    "roc_auc",
    "binary_metrics_at_threshold",
    "threshold_sweep",
    "threshold_for_target_sensitivity",
    "resolve_class_index",
    "melanoma_index",
    "class_recall",
    "melanoma_sensitivity",
    "clinical_report",
    "format_honest_summary",
]

#: Version of the dict shape emitted by :func:`clinical_report`.
CLINICAL_SCHEMA_VERSION = 1

#: Two-sided standard-normal quantile for a 95% interval.
Z_95 = 1.959963984540054

#: Names HAM10000 sources use for the melanoma class (spec name and raw ``dx``).
MELANOMA_ALIASES: Tuple[str, ...] = ("melanoma", "mel")


# --------------------------------------------------------------------------- #
# input coercion
# --------------------------------------------------------------------------- #


def _as_labels(y: Sequence[int], name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D [N], got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty; no metric is defined on zero examples")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(np.mod(arr.astype(np.float64), 1), 0)):
            raise ValueError(f"{name} must contain integer class labels")
        arr = arr.astype(np.int64)
    arr = arr.astype(np.int64)
    if arr.min() < 0:
        raise ValueError(f"{name} contains a negative class label")
    return arr


def _as_scores(y_score: Sequence[float], name: str = "y_score") -> np.ndarray:
    arr = np.asarray(y_score, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty; no metric is defined on zero examples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/inf")
    return arr


def _as_positive(y_true_positive: Sequence[Any], n_expected: int) -> np.ndarray:
    arr = np.asarray(y_true_positive)
    if arr.ndim != 1:
        raise ValueError(f"y_true_positive must be 1-D [N], got shape {arr.shape}")
    if arr.size != n_expected:
        raise ValueError(
            f"y_true_positive has {arr.size} entries but y_score has {n_expected}"
        )
    return arr.astype(bool)


def _num_classes(
    *label_arrays: np.ndarray,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> int:
    if num_classes is not None:
        k = int(num_classes)
    elif class_names is not None:
        k = len(class_names)
    else:
        k = int(max(int(a.max()) for a in label_arrays)) + 1
    if k < 1:
        raise ValueError(f"num_classes must be >= 1, got {k}")
    for a in label_arrays:
        if int(a.max()) >= k:
            raise ValueError(f"label {int(a.max())} is out of range for {k} classes")
    return k


def _names(class_names: Optional[Sequence[str]], k: int) -> List[str]:
    if class_names is None:
        return [f"class_{i}" for i in range(k)]
    if len(class_names) != k:
        raise ValueError(f"class_names has {len(class_names)} entries but K = {k}")
    return [str(c) for c in class_names]


# --------------------------------------------------------------------------- #
# (e) Wilson score intervals — attached to every rate
# --------------------------------------------------------------------------- #


def _ci_level(z: float) -> float:
    """Two-sided coverage implied by ``z`` (0.95 for :data:`Z_95`)."""
    return round(math.erf(float(z) / math.sqrt(2.0)), 6)


def wilson_interval(successes: int, n: int, *, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion ``successes / n``.

    The Wilson interval is used rather than the textbook normal ("Wald")
    interval because Wald is badly wrong exactly where this project needs an
    interval most: tiny per-class supports and proportions near 0 or 1, where
    Wald produces impossible bounds outside ``[0, 1]`` and collapses to zero
    width at ``0/n`` and ``n/n``. Wilson stays inside ``[0, 1]`` by construction
    and stays informative at the extremes (``0/10`` → ``(0.000, 0.278)``).

    Parameters
    ----------
    successes:
        Number of successes observed (``0 <= successes <= n``).
    n:
        Number of trials — the *support*. Must be reported alongside the rate.
    z:
        Two-sided standard-normal quantile; :data:`Z_95` (default) gives 95%.

    Returns
    -------
    (lo, hi)
        Plain floats, always within ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If ``n < 0``, ``successes < 0``, ``successes > n``, or ``z <= 0``. With
        ``n == 0`` the interval is the whole ``[0.0, 1.0]``: zero observations
        constrain the rate not at all (callers should prefer
        :func:`rate_with_ci`, which reports the *value* as ``None`` there).
    """
    s, m = int(successes), int(n)
    if m < 0:
        raise ValueError(f"n must be >= 0, got {m}")
    if s < 0 or s > m:
        raise ValueError(f"successes must satisfy 0 <= successes <= n, got {s}/{m}")
    if z <= 0:
        raise ValueError(f"z must be > 0, got {z}")
    if m == 0:
        return (0.0, 1.0)
    p = s / m
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / m
    center = (p + z2 / (2.0 * m)) / denom
    half = (float(z) / denom) * math.sqrt(p * (1.0 - p) / m + z2 / (4.0 * m * m))
    lo = float(min(max(center - half, 0.0), 1.0))
    hi = float(min(max(center + half, 0.0), 1.0))
    # At the extremes the algebra collapses exactly: 0/n → lo == 0, n/n → hi == 1.
    # Snap them so float round-off cannot report an interval that excludes its own
    # point estimate (0.9999999999999999 for a rate of 1.0).
    if s == 0:
        lo = 0.0
    if s == m:
        hi = 1.0
    return (lo, hi)


def rate_with_ci(
    successes: int,
    n: int,
    *,
    z: float = Z_95,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """One rate as a JSON-serializable dict: value + Wilson CI + support.

    This is the only shape in which this module reports a proportion. A bare
    point estimate is never returned, because "melanoma sensitivity 0.86" and
    "melanoma sensitivity 0.86 (95% CI 0.55-0.97, n=12)" are different claims
    and only the second one is honest.

    Returns
    -------
    dict
        ``{"name", "value", "lo", "hi", "n", "successes", "ci_level", "z",
        "undefined_reason"}``. When ``n == 0`` the rate is **undefined**:
        ``value``/``lo``/``hi`` are ``None`` (JSON ``null``, never ``NaN``) and
        ``undefined_reason`` explains why.
    """
    s, m = int(successes), int(n)
    if m == 0:
        return {
            "name": name,
            "value": None,
            "lo": None,
            "hi": None,
            "n": 0,
            "successes": 0,
            "ci_level": _ci_level(z),
            "z": float(z),
            "undefined_reason": "no support (n=0)",
        }
    lo, hi = wilson_interval(s, m, z=z)
    return {
        "name": name,
        "value": float(s / m),
        "lo": lo,
        "hi": hi,
        "n": m,
        "successes": s,
        "ci_level": _ci_level(z),
        "z": float(z),
        "undefined_reason": None,
    }


# --------------------------------------------------------------------------- #
# (a) baselines — the fix for the misleading "chance" framing
# --------------------------------------------------------------------------- #


def baselines(
    y_true: Sequence[int],
    *,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """All the reference points for an accuracy number, together, in one dict.

    **On an imbalanced dataset the majority-class rate is the meaningful floor
    and uniform chance (1/K) is not.** A model that always predicts ``nv``
    scores ``0.669`` on HAM10000 while knowing nothing; against uniform chance
    (``0.143``) that same do-nothing model would look like a 4.7x win. Reporting
    accuracy against ``1/K`` on a 67%-majority dataset overstates the result by
    more than a factor of four, and no practitioner would ever use ``1/K`` as
    the comparison. This function returns *every* baseline at once precisely so
    that a caller cannot quote only the flattering one.

    Returns
    -------
    dict
        ``{"n", "num_classes", "uniform_chance", "majority_class_rate",
        "majority_class_index", "majority_class_name", "prevalence", "support",
        "class_names", "meaningful_floor", "note"}``. ``prevalence`` is the
        per-class fraction of ``y_true``; ``support`` the raw counts.
        ``meaningful_floor`` names which baseline a reader should compare
        against (``"majority_class_rate"`` whenever the data is imbalanced).

    Raises
    ------
    ValueError
        On empty ``y_true`` or labels outside ``[0, num_classes)``.
    """
    yt = _as_labels(y_true, "y_true")
    k = _num_classes(yt, num_classes=num_classes, class_names=class_names)
    names = _names(class_names, k)
    counts = np.bincount(yt, minlength=k).astype(np.int64)
    n = int(yt.size)
    prevalence = counts / float(n)
    maj = int(counts.argmax())
    maj_rate = float(prevalence[maj])
    uniform = 1.0 / k
    imbalanced = maj_rate > uniform * 1.5
    return {
        "n": n,
        "num_classes": k,
        "class_names": names,
        "uniform_chance": float(uniform),
        "majority_class_rate": maj_rate,
        "majority_class_index": maj,
        "majority_class_name": names[maj],
        "prevalence": [float(x) for x in prevalence],
        "support": [int(x) for x in counts],
        "imbalanced": bool(imbalanced),
        "meaningful_floor": "majority_class_rate" if imbalanced else "uniform_chance",
        "note": (
            "Compare accuracy against majority_class_rate, not uniform_chance: a "
            "constant majority-class predictor already scores majority_class_rate "
            "while learning nothing."
        ),
    }


# --------------------------------------------------------------------------- #
# (b) multi-class metrics
# --------------------------------------------------------------------------- #


def argmax_predictions(y_prob: Sequence[Sequence[float]]) -> np.ndarray:
    """``[N, K]`` probabilities → ``[N]`` argmax class labels (ties → lowest index)."""
    p = np.asarray(y_prob, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"y_prob must be [N, K], got shape {p.shape}")
    if p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError(f"y_prob is empty, got shape {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("y_prob contains NaN/inf")
    return p.argmax(axis=1).astype(np.int64)


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """``[K, K]`` counts, **rows = true class, columns = predicted class**.

    Returned as a numpy int array (the raw numeric helper);
    :func:`multiclass_metrics` embeds it as a nested list for JSON.
    """
    yt = _as_labels(y_true, "y_true")
    yp = _as_labels(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} must be the same length")
    k = _num_classes(yt, yp, num_classes=num_classes)
    flat = np.bincount(yt * k + yp, minlength=k * k)
    return flat.reshape(k, k).astype(np.int64)


def multiclass_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    z: float = Z_95,
) -> Dict[str, Any]:
    """Full multi-class readout in one JSON-serializable dict.

    Every quantity is read straight off the confusion matrix ``C`` (rows = true,
    cols = predicted), with per-class one-vs-rest definitions:

    - ``recall``/sensitivity ``= C[i,i] / row_i``   — of the class's true cases, how
      many were caught. This is the clinically load-bearing per-class number.
    - ``precision``/PPV ``= C[i,i] / col_i``        — of the cases flagged as class
      ``i``, how many really were.
    - ``specificity`` ``= TN_i / (TN_i + FP_i)``    — of the cases that are *not*
      class ``i``, how many were correctly not flagged.
    - ``f1`` — harmonic mean of precision and recall (``None`` if either is
      undefined).
    - ``accuracy`` ``= trace(C) / N``, reported **with the baselines attached**
      (see :func:`baselines`) so it can never be quoted alone.
    - ``balanced_accuracy`` — the unweighted mean of per-class recall over the
      classes that have support. This is the number that does not collapse under
      class imbalance: a constant majority-class predictor on HAM10000 scores
      ``0.669`` accuracy but ``0.143`` balanced accuracy.

    Balanced accuracy is a mean of rates, not a binomial proportion, so no
    Wilson interval is attached to it; each per-class recall carries its own.

    Classes with zero support (e.g. ``df`` missing from a small split) get
    ``recall = None`` with an ``undefined_reason``, are excluded from balanced
    accuracy, and the number of classes actually counted is reported in
    ``balanced_accuracy_classes_counted`` — no NaN, no silent zero.
    """
    yt = _as_labels(y_true, "y_true")
    yp = _as_labels(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} must be the same length")
    k = _num_classes(yt, yp, num_classes=num_classes, class_names=class_names)
    names = _names(class_names, k)
    cm = confusion_matrix(yt, yp, num_classes=k)
    n = int(yt.size)

    row = cm.sum(axis=1)  # support per true class
    col = cm.sum(axis=0)  # predicted count per class
    diag = np.diag(cm)

    per_class: List[Dict[str, Any]] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for i in range(k):
        tp = int(diag[i])
        fn = int(row[i] - tp)
        fp = int(col[i] - tp)
        tn = int(n - tp - fn - fp)
        rec = rate_with_ci(tp, tp + fn, z=z, name=f"recall[{names[i]}]")
        prec = rate_with_ci(tp, tp + fp, z=z, name=f"precision[{names[i]}]")
        spec = rate_with_ci(tn, tn + fp, z=z, name=f"specificity[{names[i]}]")
        if rec["value"] is None:
            rec["undefined_reason"] = f"class '{names[i]}' has no true examples in this split"
        if prec["value"] is None:
            prec["undefined_reason"] = f"class '{names[i]}' was never predicted"
        if spec["value"] is None:
            spec["undefined_reason"] = f"every example is class '{names[i]}'"
        f1: Optional[float] = None
        if rec["value"] is not None and prec["value"] is not None:
            denom = rec["value"] + prec["value"]
            f1 = float(2.0 * rec["value"] * prec["value"] / denom) if denom > 0 else 0.0
            f1s.append(f1)
        if rec["value"] is not None:
            recalls.append(float(rec["value"]))
        per_class.append(
            {
                "index": i,
                "class": names[i],
                "support": int(row[i]),
                "predicted": int(col[i]),
                "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "recall": rec,
                "precision": prec,
                "specificity": spec,
                "f1": f1,
            }
        )

    correct = int(diag.sum())
    base = baselines(yt, num_classes=k, class_names=names)
    return {
        "n": n,
        "num_classes": k,
        "class_names": names,
        "confusion_matrix": [[int(v) for v in r] for r in cm],
        "confusion_matrix_layout": "rows = true class, columns = predicted class",
        "accuracy": rate_with_ci(correct, n, z=z, name="accuracy"),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "balanced_accuracy_classes_counted": len(recalls),
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per_class,
        "baselines": base,
    }


# --------------------------------------------------------------------------- #
# (c) binary readout at a threshold — the clinically load-bearing one
# --------------------------------------------------------------------------- #


def group_probability(
    y_prob: Sequence[Sequence[float]], group_idx: Sequence[int]
) -> np.ndarray:
    """``[N, K]`` probabilities → ``[N]`` summed probability over a class group.

    The batched sibling of :func:`vitreous.malignancy.malignant_probability`:
    ``Σ P(class)`` over e.g. the malignant group ``{mel, bcc, akiec}``, clipped
    to ``[0, 1]``. This is the score the decision threshold is applied to.
    """
    p = np.asarray(y_prob, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"y_prob must be [N, K], got shape {p.shape}")
    idx = [int(i) for i in group_idx]
    if not idx:
        return np.zeros(p.shape[0], dtype=np.float64)
    if min(idx) < 0 or max(idx) >= p.shape[1]:
        raise ValueError(f"group_idx {idx} out of range for {p.shape[1]} classes")
    return np.clip(p[:, idx].sum(axis=1), 0.0, 1.0)


def _midranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks with tied values sharing their average rank."""
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)  # last 1-based rank of each unique value
    first = last - counts + 1
    mid = (first + last) / 2.0
    return mid[inv]


def roc_auc(
    y_true_positive: Sequence[Any], y_score: Sequence[float]
) -> Optional[float]:
    """ROC AUC via the rank / Mann-Whitney U identity — numpy only, ties handled.

    ``AUC = (Σ rank(positives) - n_pos(n_pos+1)/2) / (n_pos · n_neg)`` where ranks
    are **midranks** (tied scores share their average rank). That tie handling is
    what makes the identity equal the trapezoidal ROC area: a tied
    positive/negative pair contributes exactly ``0.5``, so a constant scorer gets
    ``0.5`` rather than ``0.0`` or ``1.0``.

    Returns
    -------
    float or None
        ``None`` — never ``NaN`` — when one of the two classes is absent, since
        AUC is undefined without both a positive and a negative example.
    """
    s = _as_scores(y_score)
    y = _as_positive(y_true_positive, s.size)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _midranks(s)
    u = float(ranks[y].sum()) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def binary_metrics_at_threshold(
    y_true_positive: Sequence[Any],
    y_score: Sequence[float],
    threshold: float,
    *,
    z: float = Z_95,
    label: str = "positive",
) -> Dict[str, Any]:
    """The operating-point readout: sensitivity / specificity / PPV / NPV + counts.

    The decision rule is ``score >= threshold`` (inclusive — a score exactly at
    the threshold is flagged). This matters: ``LensExplorer`` defaults the
    threshold to ``0.2`` and calls it a "high-sensitivity default"; this function
    is what turns that claim into a measured number.

    Parameters
    ----------
    y_true_positive:
        ``[N]`` booleans — is this image truly in the positive group (e.g. truly
        malignant, or truly melanoma)?
    y_score:
        ``[N]`` the positive-group probability per image (see
        :func:`group_probability`).
    threshold:
        Decision threshold in the score's units.
    label:
        Name of the positive group, used in the returned rate names and by
        :func:`format_honest_summary` (e.g. ``"melanoma"``, ``"malignancy"``).

    Returns
    -------
    dict
        ``{"label", "threshold", "decision_rule", "n", "n_positive",
        "n_negative", "counts": {tp, fp, tn, fn}, "prevalence",
        "flagged_fraction", "sensitivity", "specificity", "ppv", "npv",
        "accuracy", "youden_j"}``. Each of sensitivity / specificity / ppv / npv
        / accuracy is a :func:`rate_with_ci` dict carrying its own Wilson
        interval and support ``n``. Rates with no support are ``None`` with an
        ``undefined_reason`` (e.g. PPV when nothing was flagged), never ``NaN``.
    """
    s = _as_scores(y_score)
    y = _as_positive(y_true_positive, s.size)
    thr = float(threshold)
    flagged = s >= thr

    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    tn = int(np.sum(~flagged & ~y))
    n = int(s.size)
    n_pos, n_neg = tp + fn, tn + fp

    sens = rate_with_ci(tp, n_pos, z=z, name=f"sensitivity[{label}]")
    spec = rate_with_ci(tn, n_neg, z=z, name=f"specificity[{label}]")
    ppv = rate_with_ci(tp, tp + fp, z=z, name=f"ppv[{label}]")
    npv = rate_with_ci(tn, tn + fn, z=z, name=f"npv[{label}]")
    if sens["value"] is None:
        sens["undefined_reason"] = f"no true {label} examples in this split"
    if spec["value"] is None:
        spec["undefined_reason"] = f"every example is {label}"
    if ppv["value"] is None:
        ppv["undefined_reason"] = f"nothing flagged at threshold {thr:g}"
    if npv["value"] is None:
        npv["undefined_reason"] = f"everything flagged at threshold {thr:g}"

    youden: Optional[float] = None
    if sens["value"] is not None and spec["value"] is not None:
        youden = float(sens["value"] + spec["value"] - 1.0)

    return {
        "label": label,
        "threshold": thr,
        "decision_rule": "score >= threshold",
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "prevalence": float(n_pos / n),
        "flagged_fraction": float((tp + fp) / n),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "accuracy": rate_with_ci(tp + tn, n, z=z, name=f"accuracy[{label}]"),
        "youden_j": youden,
    }


def _default_grid(scores: np.ndarray, num_points: int) -> np.ndarray:
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")
    lo, hi = float(scores.min()), float(scores.max())
    if lo >= 0.0 and hi <= 1.0:
        return np.linspace(0.0, 1.0, int(num_points))
    return np.linspace(lo, hi, int(num_points))


def threshold_sweep(
    y_true_positive: Sequence[Any],
    y_score: Sequence[float],
    *,
    thresholds: Optional[Sequence[float]] = None,
    num_points: int = 101,
    z: float = Z_95,
    label: str = "positive",
) -> Dict[str, Any]:
    """Sensitivity/specificity across a grid of thresholds → an operating curve.

    This is what lets a UI answer "what does threshold 0.2 actually buy me": one
    compact point per threshold, plus the threshold-free :func:`roc_auc`.

    Sensitivity is non-increasing and specificity non-decreasing as the threshold
    rises (the rule is ``score >= threshold``, so raising it can only un-flag
    images) — a property the tests assert directly.

    Points are deliberately compact (plain floats, with ``sensitivity_lo/hi`` and
    ``specificity_lo/hi`` Wilson bounds so a chart can band the curve) rather
    than full rate dicts; use :func:`binary_metrics_at_threshold` for the
    fully-annotated readout at the *chosen* operating point. Undefined rates
    appear as ``None``.

    ``thresholds`` defaults to ``num_points`` evenly spaced values over
    ``[0, 1]`` when the scores are probabilities, else over the observed score
    range.
    """
    s = _as_scores(y_score)
    y = _as_positive(y_true_positive, s.size)
    grid = (
        _default_grid(s, num_points)
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float64).ravel()
    )
    if grid.size == 0:
        raise ValueError("thresholds is empty")

    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    points: List[Dict[str, Any]] = []
    for thr in grid:
        flagged = s >= thr
        tp = int(np.sum(flagged & y))
        fp = int(np.sum(flagged & ~y))
        fn = n_pos - tp
        tn = n_neg - fp
        sens = rate_with_ci(tp, n_pos, z=z)
        spec = rate_with_ci(tn, n_neg, z=z)
        ppv = rate_with_ci(tp, tp + fp, z=z)
        npv = rate_with_ci(tn, tn + fn, z=z)
        points.append(
            {
                "threshold": float(thr),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "sensitivity": sens["value"],
                "sensitivity_lo": sens["lo"],
                "sensitivity_hi": sens["hi"],
                "specificity": spec["value"],
                "specificity_lo": spec["lo"],
                "specificity_hi": spec["hi"],
                "ppv": ppv["value"],
                "npv": npv["value"],
                "youden_j": (
                    None
                    if sens["value"] is None or spec["value"] is None
                    else float(sens["value"] + spec["value"] - 1.0)
                ),
            }
        )

    return {
        "label": label,
        "decision_rule": "score >= threshold",
        "n": int(s.size),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "roc_auc": roc_auc(y, s),
        "thresholds": [float(t) for t in grid],
        "points": points,
    }


def threshold_for_target_sensitivity(
    y_true_positive: Sequence[Any],
    y_score: Sequence[float],
    target_sensitivity: float = 0.95,
    *,
    thresholds: Optional[Sequence[float]] = None,
    z: float = Z_95,
    label: str = "positive",
) -> Dict[str, Any]:
    """The operating point that buys a target sensitivity at the least cost.

    Returns the **highest** threshold whose sensitivity is still
    ``>= target_sensitivity`` — i.e. the best specificity available subject to
    the sensitivity floor. (Asking for the *lowest* such threshold is degenerate:
    a threshold of 0 flags every image and trivially attains sensitivity 1.0, at
    zero specificity. The clinically meaningful question is always "how selective
    can I be while still catching 95% of the melanomas?")

    ``thresholds`` defaults to the observed unique scores plus ``0.0``, which are
    exactly the values at which the decision changes.

    Returns
    -------
    dict
        The full :func:`binary_metrics_at_threshold` readout at the chosen
        threshold, plus ``"target_sensitivity"`` and ``"achieved"``. If no
        threshold reaches the target (only possible when the split contains no
        positive examples), ``{"achieved": False, "threshold": None,
        "target_sensitivity": ..., "reason": ...}`` is returned instead of a
        misleading point estimate.
    """
    t = float(target_sensitivity)
    if not (0.0 < t <= 1.0):
        raise ValueError(f"target_sensitivity must be in (0, 1], got {t}")
    s = _as_scores(y_score)
    y = _as_positive(y_true_positive, s.size)
    n_pos = int(y.sum())
    if n_pos == 0:
        return {
            "label": label,
            "target_sensitivity": t,
            "achieved": False,
            "threshold": None,
            "reason": f"no true {label} examples in this split; sensitivity is undefined",
        }

    if thresholds is None:
        cand = np.unique(np.concatenate([s, np.array([0.0])]))
    else:
        cand = np.unique(np.asarray(thresholds, dtype=np.float64).ravel())
    if cand.size == 0:
        raise ValueError("thresholds is empty")

    pos_scores = s[y]
    best: Optional[float] = None
    for thr in cand:  # ascending → last qualifying candidate is the highest
        sens = float(np.sum(pos_scores >= thr)) / n_pos
        if sens >= t:
            best = float(thr)
    if best is None:
        return {
            "label": label,
            "target_sensitivity": t,
            "achieved": False,
            "threshold": None,
            "reason": f"no threshold in the candidate set reaches sensitivity {t:g}",
        }
    out = binary_metrics_at_threshold(y, s, best, z=z, label=label)
    out["target_sensitivity"] = t
    out["achieved"] = True
    return out


# --------------------------------------------------------------------------- #
# (d) melanoma-specific recall
# --------------------------------------------------------------------------- #


def resolve_class_index(
    class_names: Sequence[str], ref: Union[int, str]
) -> int:
    """Resolve a class reference (index or name, case-insensitive) to an index."""
    if isinstance(ref, (int, np.integer)) and not isinstance(ref, bool):
        i = int(ref)
        if not (0 <= i < len(class_names)):
            raise ValueError(f"class index {i} out of range for {len(class_names)} classes")
        return i
    name = str(ref)
    names = [str(c) for c in class_names]
    if name in names:
        return names.index(name)
    lowered = [c.lower() for c in names]
    if name.lower() in lowered:
        return lowered.index(name.lower())
    raise ValueError(f"class {name!r} not found in {names}")


def melanoma_index(class_names: Sequence[str]) -> Optional[int]:
    """Index of the melanoma class, or ``None`` if this dataset has none.

    Matches HAM10000's spec name (``"Melanoma"``) and its raw ``dx`` code
    (``"mel"``), case-insensitively — see :data:`MELANOMA_ALIASES`.
    """
    lowered = [str(c).lower() for c in class_names]
    for alias in MELANOMA_ALIASES:
        if alias in lowered:
            return lowered.index(alias)
    return None


def class_recall(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_ref: Union[int, str],
    *,
    class_names: Optional[Sequence[str]] = None,
    num_classes: Optional[int] = None,
    z: float = Z_95,
) -> Dict[str, Any]:
    """Recall (sensitivity) for **one named class**, as a first-class number.

    Per-class recall already lives inside :func:`multiclass_metrics`, but buried
    in an array indexed by class id. A bundle should be able to carry "melanoma
    sensitivity" as its own field — melanoma is the deadliest class and a missed
    melanoma is the worst error this system can make
    (``docs/MALIGNANCY-LENS.md`` §8) — so this is the convenience path that lifts
    it out.

    Returns
    -------
    dict
        A :func:`rate_with_ci` dict with ``"class"`` and ``"index"`` added.
    """
    yt = _as_labels(y_true, "y_true")
    yp = _as_labels(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} must be the same length")
    k = _num_classes(yt, yp, num_classes=num_classes, class_names=class_names)
    names = _names(class_names, k)
    i = resolve_class_index(names, class_ref)
    support = int(np.sum(yt == i))
    hits = int(np.sum((yt == i) & (yp == i)))
    out = rate_with_ci(hits, support, z=z, name=f"recall[{names[i]}]")
    if out["value"] is None:
        out["undefined_reason"] = f"class '{names[i]}' has no true examples in this split"
    out["class"] = names[i]
    out["index"] = i
    return out


def melanoma_sensitivity(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    *,
    z: float = Z_95,
) -> Optional[Dict[str, Any]]:
    """Melanoma recall from argmax predictions, or ``None`` if no melanoma class.

    Note this is the *argmax* sensitivity. When a decision threshold is in play
    (the workbench's malignancy lens), the honest number is the thresholded one
    from :func:`binary_metrics_at_threshold` with the melanoma class as the
    positive group — argmax is just the special case of "whatever class won".
    """
    i = melanoma_index(class_names)
    if i is None:
        return None
    return class_recall(y_true, y_pred, i, class_names=class_names, z=z)


# --------------------------------------------------------------------------- #
# (f) the bundle-ready report + the one-line honest summary
# --------------------------------------------------------------------------- #


def clinical_report(
    y_true: Sequence[int],
    y_prob: Sequence[Sequence[float]],
    *,
    class_names: Optional[Sequence[str]] = None,
    positive_classes: Optional[Sequence[Union[int, str]]] = None,
    positive_label: Optional[str] = None,
    threshold: float = 0.5,
    target_sensitivity: Optional[float] = 0.95,
    focus_class: Optional[Union[int, str]] = None,
    sweep_points: int = 101,
    z: float = Z_95,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything above, computed once, as one JSON-serializable provenance blob.

    This is the function a notebook or a pack builder should call. It bundles the
    baselines (so accuracy can never be quoted without its floor), the full
    multi-class metrics, the binary operating-point readout + sweep + AUC for a
    class group (e.g. the malignant group, or melanoma alone), the focus class's
    recall, and the one-line :func:`format_honest_summary` sentence.

    Parameters
    ----------
    y_true:
        ``[N]`` true class labels.
    y_prob:
        ``[N, K]`` per-class probabilities.
    positive_classes:
        Class names or indices forming the binary positive group (e.g.
        ``["Melanoma", "Basal cell carcinoma", "Actinic keratoses"]``, or just
        ``["Melanoma"]``). Omit to skip the binary section entirely.
    positive_label:
        Human label for that group; defaults to the single class's name when the
        group has one member, else ``"malignancy"``.
    threshold:
        The decision threshold actually in use by the product (the workbench
        ships ``0.2``). Pass the real one — the point of this module is to
        measure what the shipped threshold buys.
    focus_class:
        A class whose recall should be lifted out as its own field; defaults to
        melanoma when the dataset has one.
    """
    yt = _as_labels(y_true, "y_true")
    p = np.asarray(y_prob, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"y_prob must be [N, K], got shape {p.shape}")
    if p.shape[0] != yt.size:
        raise ValueError(f"y_prob has {p.shape[0]} rows but y_true has {yt.size} labels")
    k = _num_classes(yt, num_classes=p.shape[1], class_names=class_names)
    names = _names(class_names, k)
    yp = argmax_predictions(p)

    multi = multiclass_metrics(yt, yp, num_classes=k, class_names=names, z=z)

    binary: Optional[Dict[str, Any]] = None
    if positive_classes:
        idx = [resolve_class_index(names, c) for c in positive_classes]
        label = positive_label or (names[idx[0]].lower() if len(idx) == 1 else "malignancy")
        score = group_probability(p, idx)
        is_pos = np.isin(yt, idx)
        binary = {
            "label": label,
            "positive_classes": [names[i] for i in idx],
            "positive_class_indices": idx,
            "score": "sum of P(class) over positive_classes",
            "at_threshold": binary_metrics_at_threshold(
                is_pos, score, threshold, z=z, label=label
            ),
            "sweep": threshold_sweep(
                is_pos, score, num_points=sweep_points, z=z, label=label
            ),
            "roc_auc": roc_auc(is_pos, score),
        }
        if target_sensitivity is not None:
            binary["target_sensitivity_operating_point"] = (
                threshold_for_target_sensitivity(
                    is_pos, score, target_sensitivity, z=z, label=label
                )
            )

    if focus_class is None:
        mi = melanoma_index(names)
        focus_idx = mi
    else:
        focus_idx = resolve_class_index(names, focus_class)
    focus = (
        class_recall(yt, yp, focus_idx, class_names=names, z=z)
        if focus_idx is not None
        else None
    )

    report: Dict[str, Any] = {
        "clinical_schema_version": CLINICAL_SCHEMA_VERSION,
        "n": int(yt.size),
        "num_classes": k,
        "class_names": names,
        "baselines": multi["baselines"],
        "multiclass": multi,
        "binary": binary,
        "focus_class": focus,
        "provenance": dict(provenance or {}),
    }
    report["summary"] = format_honest_summary(report)
    return report


def _fmt(v: Optional[float], places: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{places}f}"


def format_honest_summary(report: Dict[str, Any]) -> str:
    """The one plain sentence a UI or notebook should print.

    Example::

        melanoma sensitivity 0.86 (95% CI 0.78-0.92, n=111) / specificity 0.61
        at threshold 0.20; balanced accuracy 0.63; accuracy 0.79 vs
        majority-class baseline 0.67 (uniform chance 0.14)

    **Overall accuracy is never emitted without the majority-class baseline
    beside it.** That pairing is the whole point of this formatter: on a dataset
    that is 67% one class, an accuracy number alone is not a result, and the
    project already published one ("0.7922 vs chance 0.14285") that read as far
    stronger than it was. The uniform-chance figure is kept only in third place,
    in parentheses, so it can be seen for what it is.

    Accepts the dict from :func:`clinical_report`; tolerates missing sections.
    """
    parts: List[str] = []

    binary = report.get("binary") or {}
    at = binary.get("at_threshold") or {}
    if at:
        sens = at.get("sensitivity") or {}
        spec = at.get("specificity") or {}
        lvl = int(round(float(sens.get("ci_level", 0.95)) * 100))
        head = f"{at.get('label', 'positive')} sensitivity {_fmt(sens.get('value'))}"
        if sens.get("value") is not None:
            head += (
                f" ({lvl}% CI {_fmt(sens.get('lo'))}-{_fmt(sens.get('hi'))},"
                f" n={sens.get('n')})"
            )
        head += (
            f" / specificity {_fmt(spec.get('value'))}"
            f" at threshold {_fmt(at.get('threshold'))}"
        )
        parts.append(head)

    focus = report.get("focus_class") or {}
    if focus and not at:
        lvl = int(round(float(focus.get("ci_level", 0.95)) * 100))
        line = f"{focus.get('class', 'focus')} recall {_fmt(focus.get('value'))}"
        if focus.get("value") is not None:
            line += (
                f" ({lvl}% CI {_fmt(focus.get('lo'))}-{_fmt(focus.get('hi'))},"
                f" n={focus.get('n')})"
            )
        parts.append(line)

    multi = report.get("multiclass") or {}
    if multi.get("balanced_accuracy") is not None:
        parts.append(f"balanced accuracy {_fmt(multi.get('balanced_accuracy'))}")

    base = report.get("baselines") or multi.get("baselines") or {}
    acc = (multi.get("accuracy") or {}).get("value")
    if acc is not None and base:
        tail = (
            f"accuracy {_fmt(acc)} vs majority-class baseline"
            f" {_fmt(base.get('majority_class_rate'))}"
        )
        if base.get("uniform_chance") is not None:
            tail += f" (uniform chance {_fmt(base.get('uniform_chance'))})"
        parts.append(tail)
    elif base:
        parts.append(
            f"majority-class baseline {_fmt(base.get('majority_class_rate'))}"
        )

    return "; ".join(parts) if parts else "no metrics available"
