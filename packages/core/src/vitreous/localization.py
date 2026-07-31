"""Localization metrics — *did it look in the right place?* (numpy-only).

This module is the measurement substrate for the project's pivot from
**CADx** (computer-aided diagnosis — "classify the lesion") to **CADe**
(computer-aided detection — "mark the region a clinician should examine"). The
distinction is regulatory and statistical, not cosmetic (``docs/VITREOUS-TRIAL.md``
§1): the FDA evaluates a CADe device on **localization accuracy** and a CADx
device on the accuracy of its likelihood-of-disease output.

**Why the substitution matters.** At the published operating point for
skin-lesion AI (sensitivity 0.91 / specificity 0.64), positive predictive value
collapses with prevalence: ~37.9% on HAM10000 as-is, ~2.5% at a 1% consumer
prevalence — about **39 false alarms per true finding**. PPV is
``f(sensitivity, specificity, prevalence)``. Every quantity in this module is
computed **conditional on a lesion being present in the image**, so none of them
has a prevalence term at all. IoU, Dice, pointing-game hit rate and
attribution-mass-within-mask are properties of a single image that already
contains a lesion. That escape is real, and it holds *only* under that
condition — whole-body screening ("find the concerning spots on this back")
puts the burden back, which is exactly why :func:`froc_curve` exists.

**Relationship to :mod:`vitreous.xai.eval`.** That module measures
*faithfulness*: :func:`~vitreous.xai.eval.deletion_insertion` masks the
highest-attributed patches first and watches the class probability collapse —
does the attribution reflect what the model actually used? This module measures
*correctness of place*: does the attribution land on the pixels a dermatologist
annotated? **They are complementary and they can disagree.** An explanation can
be perfectly faithful to a model that is looking at a ruler. Nothing here
duplicates deletion/insertion; report both.

What is provided:

(a) **Overlap** — :func:`iou`, :func:`dice`, :func:`overlap_at_threshold`
    (adds pixel precision/recall at one binarization threshold) and
    :func:`overlap_threshold_sweep` / :func:`overlap_dataset`, which sweep the
    threshold and report the best achievable Dice/IoU *and* the threshold that
    achieved it — flagged as optimistically biased, see below.
(b) **Pointing game** — :func:`pointing_game_hit` (one image) and
    :func:`pointing_game` (a dataset), the standard weakly-supervised
    localization metric: does the argmax of the saliency map fall inside the
    ground-truth mask? With a tolerance variant (hit if the peak is within
    ``k`` pixels of the mask).
(c) **Attribution mass** — :func:`mass_within_mask` / :func:`mass_within_mask_dataset`,
    the fraction of total positive attribution mass falling inside the lesion.
    This is the direct answer to "is it looking in the right place", and it
    doubles as the **shortcut detector** of ``docs/UX-VISION.md`` lay question #2
    (:func:`shortcut_report`): for dermoscopy, mass landing outside the lesion —
    rulers, ink marks, vignetting, hair — is the classic artifact-driven failure
    mode, and this quantifies it instead of leaving it suspected.
(d) **FROC** — :func:`froc_curve`, sensitivity as a function of false marks per
    image, for the multi-lesion case. FROC replaces ROC when the unit of
    analysis is a *mark* rather than an image, which is precisely why it suits
    CADe.
(e) **Report** — :func:`localization_report` bundles all of the above plus
    provenance into one JSON-serializable dict, and
    :func:`format_localization_summary` renders the single sentence a UI or
    notebook should print. These mirror
    :func:`vitreous.clinical.clinical_report` / ``format_honest_summary``.

**Ground truth this needs.** Two public sources, already loadable by this
package: **ISIC 2018 Task 1** ships binary *lesion segmentation* masks ("did the
model look at the lesion at all"), and **ISIC 2018 Task 2** ships pixel-level
masks for five dermoscopic *attributes* over 2,594 HAM10000 images ("did it look
at the diagnostic structures") — see
:func:`vitreous.dermoscopy.load_isic2018_task2_attributes`.
Task 2 is the stronger endpoint. :func:`load_binary_mask` reads either.

**Honesty requirements enforced here:**

* Every *rate* (a count over a countable denominator) carries a Wilson score
  interval and its support ``n``, via :func:`vitreous.clinical.rate_with_ci` —
  imported, not reimplemented.
* Quantities that are **not** binomial proportions — IoU, Dice, a mean of
  per-image mass fractions — get no Wilson interval. They are reported with
  ``n``, spread (sd/median/min/max), and a ``statistic`` string saying so.
  Each is also paired with a genuine proportion (e.g. "fraction of images whose
  attribution mass is *majority*-inside the lesion") which does carry an
  interval.
* Pixel-level rates (precision/recall over pixels) carry an explicit
  ``interval_caveat``: neighbouring pixels are highly spatially correlated, so a
  binomial interval over pixels is optimistically narrow. Image-level rates
  (the pointing game) do not have this problem.
* **Nothing NaN escapes.** An undefined quantity is ``None`` plus an
  ``undefined_reason`` string.
* **An empty ground-truth mask is an explicit, documented case** — a lesion-free
  image. Recall, mass-fraction and the pointing game are *undefined* there (not
  zero, not a miss); such images are counted and excluded from aggregates rather
  than silently scored.
* **A best-over-threshold figure is optimistically biased.** Choosing the
  threshold that maximizes Dice *on the data you are reporting* is model
  selection on the test set. Every such number is stamped
  ``"threshold_selection": "best-on-this-data (optimistic)"`` and carries a
  ``warning``, so a caller cannot quote it as if it were held out. To report an
  honest figure, pick the threshold on one split and evaluate on another.

**Import discipline (M0 rule):** numpy + stdlib only. No torch, no scikit-learn,
no scipy, no Pillow at import time (Pillow is a lazy fallback for non-PNG masks,
matching :mod:`vitreous.dermoscopy`). PNG masks are decoded with that module's
existing stdlib reader — imported, not forked.

**Serialization:** every returned value is a plain ``float`` / ``int`` / ``bool``
/ ``str`` / ``list`` / ``dict`` / ``None``; no numpy scalars leak out, so any
result survives ``json.dumps`` into a pack manifest's provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .clinical import Z_95, rate_with_ci, wilson_interval

__all__ = [
    "LOCALIZATION_SCHEMA_VERSION",
    "Z_95",
    "OPTIMISTIC_THRESHOLD_NOTE",
    "PIXEL_INTERVAL_CAVEAT",
    "wilson_interval",
    "rate_with_ci",
    # coercion + geometry helpers
    "as_binary_mask",
    "as_saliency_map",
    "binarize",
    "resize_map",
    "normalize_map",
    "border_mask",
    "load_binary_mask",
    "peak_location",
    "distance_to_mask",
    # (a) overlap
    "iou",
    "dice",
    "overlap_at_threshold",
    "overlap_threshold_sweep",
    "overlap_dataset",
    # (b) pointing game
    "pointing_game_hit",
    "pointing_game",
    # (c) attribution mass / shortcut detection
    "attribution_mass",
    "mass_within_mask",
    "mass_within_mask_dataset",
    "shortcut_report",
    # (d) FROC
    "assign_marks_to_lesions",
    "froc_curve",
    # (e) the bundle report
    "localization_report",
    "format_localization_summary",
]

#: Version of the dict shape emitted by :func:`localization_report`.
LOCALIZATION_SCHEMA_VERSION = 1

#: Stamped onto every best-over-threshold figure. See the module docstring.
OPTIMISTIC_THRESHOLD_NOTE = "best-on-this-data (optimistic)"

#: Attached to every rate whose denominator counts *pixels* rather than images.
PIXEL_INTERVAL_CAVEAT = (
    "denominator counts pixels, which are strongly spatially correlated; this "
    "binomial interval is therefore optimistically narrow. Image-level rates "
    "(e.g. the pointing-game hit rate) do not have this problem."
)

#: The standard false-marks-per-image operating points a FROC curve is read at.
FROC_OPERATING_POINTS: Tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


# --------------------------------------------------------------------------- #
# input coercion + geometry
# --------------------------------------------------------------------------- #


def as_saliency_map(smap: Any, *, name: str = "saliency") -> np.ndarray:
    """``[H, W]`` continuous attribution/saliency map → float64, validated.

    Raises :class:`ValueError` on a non-2-D array, an empty array, or any
    NaN/inf — a NaN in an attribution map would silently poison every mass
    fraction downstream, and this module's contract is that nothing NaN escapes.
    """
    a = np.asarray(smap, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D [H, W] map, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty; no localization metric is defined on it")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/inf")
    return a


def as_binary_mask(
    mask: Any, *, threshold: float = 0.0, name: str = "mask"
) -> np.ndarray:
    """``[H, W]`` ground-truth mask → bool array (``value > threshold``).

    Boolean input is passed through unchanged (``threshold`` is then irrelevant).
    Numeric input is thresholded with a strict ``>``: an ISIC mask is encoded
    0/255, so with the default ``threshold=0`` **any non-zero pixel is
    annotated** — the same presence rule
    :func:`vitreous.dermoscopy.load_isic2018_task2_attributes` uses, so the two
    modules cannot disagree about what "in the mask" means.

    An **all-zero mask is legal and meaningful**: it says "this image has no
    annotated region" (a lesion-free image, or a dermoscopic criterion that is
    absent). It is not an error here; the metric functions each document what
    they do with it, and none of them silently scores it as 0.
    """
    a = np.asarray(mask)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D [H, W] mask, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty; no localization metric is defined on it")
    if a.dtype == np.bool_:
        return a
    b = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(b)):
        raise ValueError(f"{name} contains NaN/inf")
    return b > float(threshold)


def binarize(smap: Any, threshold: float, *, name: str = "saliency") -> np.ndarray:
    """Binarize a saliency map with the rule ``value >= threshold`` (inclusive).

    The rule is ``>=``, matching
    :func:`vitreous.clinical.binary_metrics_at_threshold` so that "threshold" has
    one meaning across this codebase. A threshold at or below the map's minimum
    therefore selects **every** pixel (the sensitivity=1 endpoint of a sweep).
    """
    a = as_saliency_map(smap, name=name)
    return a >= float(threshold)


def _check_same_shape(
    a: np.ndarray, b: np.ndarray, name_a: str = "saliency", name_b: str = "mask"
) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} has shape {tuple(a.shape)} but {name_b} has shape "
            f"{tuple(b.shape)}; localization metrics require matching shapes. "
            f"This module never resizes silently — a silent resize hides which "
            f"side was interpolated and how. Use "
            f"resize_map({name_a}, {tuple(b.shape)}) to lift e.g. a [14, 14] "
            f"patch grid onto a [224, 224] mask."
        )


def resize_map(
    smap: Any, shape: Sequence[int], *, mode: str = "nearest"
) -> np.ndarray:
    """Resize a ``[H, W]`` map to ``shape`` — deterministic, numpy-only.

    The common case this exists for is the ViT patch grid: an attribution of
    shape ``[14, 14]`` (one value per 16×16 patch) has to be compared against a
    ``[224, 224]`` pixel mask. Rather than downsample the *mask* (which destroys
    the annotation), the map is upsampled.

    Parameters
    ----------
    smap:
        ``[H, W]`` map (any finite float/int content).
    shape:
        Target ``(H_out, W_out)``, both >= 1.
    mode:
        ``"nearest"`` (default) — each output pixel takes the value of the source
        pixel whose cell contains its centre:
        ``src = floor((i + 0.5) * H_in / H_out)``. Exact for integer upsampling
        (a ``[2, 2]`` map → ``[4, 4]`` becomes each value repeated 2×2), and it
        never invents values that were not in the source. This is the honest
        default for a patch grid: the model really did produce one number per
        16×16 patch, and pretending otherwise smooths a blockiness that is real.

        ``"bilinear"`` — half-pixel-aligned linear interpolation
        (``align_corners=False``, the torch/PIL convention):
        ``x = (i + 0.5) * H_in / H_out - 0.5``, clamped to the source range.
        Prettier for display; it *does* invent intermediate values, so a
        pointing-game peak computed on a bilinear map can land on a pixel whose
        value the model never produced.

    Notes
    -----
    **No mass renormalization.** Upsampling replicates (or interpolates) values,
    so the *sum* of the map scales roughly with the area ratio. That is harmless
    for every metric here — mass fractions are ratios taken within one resized
    map — but do not compare raw sums across resolutions.

    Returns
    -------
    numpy.ndarray
        float64 ``[H_out, W_out]``.
    """
    a = as_saliency_map(smap, name="map")
    if len(shape) != 2:
        raise ValueError(f"shape must be (H, W), got {tuple(shape)}")
    h_out, w_out = int(shape[0]), int(shape[1])
    if h_out < 1 or w_out < 1:
        raise ValueError(f"target shape must be positive, got {(h_out, w_out)}")
    h_in, w_in = a.shape

    if mode == "nearest":
        rows = np.minimum(((np.arange(h_out) + 0.5) * h_in / h_out).astype(np.int64), h_in - 1)
        cols = np.minimum(((np.arange(w_out) + 0.5) * w_in / w_out).astype(np.int64), w_in - 1)
        return a[np.ix_(rows, cols)].astype(np.float64)

    if mode == "bilinear":
        ys = np.clip((np.arange(h_out) + 0.5) * h_in / h_out - 0.5, 0.0, h_in - 1.0)
        xs = np.clip((np.arange(w_out) + 0.5) * w_in / w_out - 0.5, 0.0, w_in - 1.0)
        y0 = np.floor(ys).astype(np.int64)
        x0 = np.floor(xs).astype(np.int64)
        y1 = np.minimum(y0 + 1, h_in - 1)
        x1 = np.minimum(x0 + 1, w_in - 1)
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]
        top = a[np.ix_(y0, x0)] * (1 - wx) + a[np.ix_(y0, x1)] * wx
        bot = a[np.ix_(y1, x0)] * (1 - wx) + a[np.ix_(y1, x1)] * wx
        return (top * (1 - wy) + bot * wy).astype(np.float64)

    raise ValueError(f"unknown resize mode {mode!r}; expected 'nearest' or 'bilinear'")


def normalize_map(smap: Any) -> np.ndarray:
    """Min-max scale a map into ``[0, 1]`` so thresholds mean the same thing.

    Threshold sweeps pooled across images are only meaningful if the images'
    saliency maps share a scale; different attribution methods (and different
    images under the same method) do not. A **constant** map carries no ranking
    information at all and is mapped to all-zeros, so that any threshold above 0
    selects nothing from it — rather than fabricating a ranking.
    """
    a = as_saliency_map(smap, name="map")
    lo, hi = float(a.min()), float(a.max())
    if hi - lo <= 0.0:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def border_mask(shape: Sequence[int], *, border_fraction: float = 0.1) -> np.ndarray:
    """A bool ``[H, W]`` frame around the image edge — the default distractor.

    ``docs/UX-VISION.md`` lay question #2 names the border and corners as where
    dermoscopy's classic shortcuts live: rulers, ink marks, sticker labels,
    vignetting from the dermatoscope's contact plate, and hair sweeping in from
    the edge. The frame is ``max(1, round(border_fraction * min(H, W)))`` pixels
    wide. Pass your own mask to :func:`shortcut_report` when you have a better
    one (e.g. a segmented ruler).
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be (H, W), got {tuple(shape)}")
    h, w = int(shape[0]), int(shape[1])
    if h < 1 or w < 1:
        raise ValueError(f"shape must be positive, got {(h, w)}")
    f = float(border_fraction)
    if not (0.0 < f < 0.5):
        raise ValueError(f"border_fraction must be in (0, 0.5), got {f}")
    b = max(1, int(round(f * min(h, w))))
    m = np.ones((h, w), dtype=bool)
    if 2 * b < h and 2 * b < w:
        m[b : h - b, b : w - b] = False
    return m


def load_binary_mask(path: Union[str, Path], *, threshold: int = 0) -> np.ndarray:
    """Read a ground-truth mask file into a bool ``[H, W]`` array.

    Reads ISIC 2018 **Task 1** lesion-segmentation masks
    (``ISIC_0000000_segmentation.png``) and **Task 2** attribute masks
    (``ISIC_0000000_attribute_globules.png``) alike — both are 8-bit 0/255 PNGs.

    PNG decoding is delegated to :mod:`vitreous.dermoscopy`'s existing stdlib
    ``zlib``/``struct`` reader (``decode_png_gray``), imported lazily. That
    helper is **private by name**; it is reused deliberately rather than forked
    so this package has exactly one PNG decoder to keep correct. Anything that is
    not a plain PNG falls back to Pillow, imported lazily so neither this module
    nor :mod:`vitreous.dermoscopy` needs PIL at import time.
    """
    p = Path(path)
    if p.suffix.lower() == ".png":
        from .dermoscopy import decode_png_gray  # the package's single PNG reader

        try:
            return np.asarray(decode_png_gray(p.read_bytes())) > int(threshold)
        except ValueError:
            pass  # exotic encoding (e.g. interlaced) — fall through to Pillow
    try:
        from PIL import Image  # lazy, optional
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"cannot read mask {p}: not a plain PNG and Pillow is not installed. "
            "Install the [ml] extra (or Pillow) to read non-PNG masks."
        ) from exc
    with Image.open(p) as im:  # pragma: no cover - requires Pillow
        return np.asarray(im.convert("L")) > int(threshold)


def peak_location(smap: Any) -> Tuple[int, int]:
    """``(row, col)`` of the saliency map's maximum.

    Ties are broken deterministically by row-major order (the first maximum),
    matching :func:`numpy.argmax`. Determinism matters: the pointing game is a
    hit/miss decision, and a tie broken at random would make the metric
    irreproducible on flat or saturated maps.
    """
    a = as_saliency_map(smap)
    r, c = np.unravel_index(int(np.argmax(a)), a.shape)
    return (int(r), int(c))


def distance_to_mask(point: Sequence[int], mask: Any) -> Optional[float]:
    """Euclidean pixel distance from ``point`` to the nearest annotated pixel.

    ``0.0`` when the point is inside the mask. ``None`` — never ``NaN`` — when
    the mask is empty, because "distance to nothing" has no value.
    """
    m = as_binary_mask(mask)
    pts = np.argwhere(m)
    if pts.size == 0:
        return None
    r, c = int(point[0]), int(point[1])
    d = np.hypot(pts[:, 0] - r, pts[:, 1] - c)
    return float(d.min())


def _fraction(num: float, den: float) -> Optional[float]:
    return float(num / den) if den > 0 else None


def _summary_stats(
    values: Sequence[Optional[float]], *, name: Optional[str] = None
) -> Dict[str, Any]:
    """Spread of a per-image quantity that is **not** a binomial proportion.

    IoU, Dice and mass fractions are ratios of continuous quantities, not
    successes over trials, so a Wilson interval does not apply to their mean.
    Rather than attach a fake interval, this reports ``n``, mean, median, sd,
    min and max, plus a ``statistic`` string saying what the number is.
    ``None`` entries (undefined on that image) are dropped and the surviving
    ``n`` is reported, so a mean can never be diluted by images where the
    quantity did not exist.
    """
    vals = [float(v) for v in values if v is not None]
    stat = (
        "mean of per-image ratios — not a binomial proportion, so no Wilson "
        "interval applies; spread is reported instead"
    )
    if not vals:
        return {
            "name": name,
            "n": 0,
            "mean": None,
            "median": None,
            "sd": None,
            "min": None,
            "max": None,
            "statistic": stat,
            "undefined_reason": "no image contributed a defined value",
        }
    a = np.asarray(vals, dtype=np.float64)
    return {
        "name": name,
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "statistic": stat,
        "undefined_reason": None,
    }


# --------------------------------------------------------------------------- #
# (a) overlap — IoU / Dice / pixel precision + recall
# --------------------------------------------------------------------------- #


def iou(pred: Any, true: Any, *, mask_threshold: float = 0.0) -> Optional[float]:
    """Intersection over union of two binary regions (Jaccard index).

    ``|A ∩ B| / |A ∪ B|``.

    Returns ``None`` — with the reason available from :func:`overlap_at_threshold`
    — when **both** regions are empty, i.e. ``0/0``. Note that some libraries
    return ``1.0`` for that case ("two empty sets agree"); this module refuses
    to, because scoring a perfect 1.0 for predicting nothing on an image with
    nothing annotated would silently inflate any dataset mean. When the
    ground truth is empty but something *was* predicted, IoU is a well-defined
    ``0.0`` and is returned as such.
    """
    a = as_binary_mask(pred, threshold=mask_threshold, name="pred")
    b = as_binary_mask(true, threshold=mask_threshold, name="true")
    _check_same_shape(a, b, "pred", "true")
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    return _fraction(inter, union)


def dice(pred: Any, true: Any, *, mask_threshold: float = 0.0) -> Optional[float]:
    """Dice / F1 overlap of two binary regions: ``2|A ∩ B| / (|A| + |B|)``.

    Dice is the harmonic mean of pixel precision and pixel recall, and is
    monotonically related to IoU (``dice = 2·iou / (1 + iou)``) — they rank
    predictions identically, so reporting both is a presentation choice, not two
    pieces of evidence. Segmentation literature quotes Dice; detection
    literature quotes IoU; this module emits both so neither audience has to
    convert.

    ``None`` when both regions are empty (see :func:`iou`).
    """
    a = as_binary_mask(pred, threshold=mask_threshold, name="pred")
    b = as_binary_mask(true, threshold=mask_threshold, name="true")
    _check_same_shape(a, b, "pred", "true")
    inter = float(np.count_nonzero(a & b))
    total = float(np.count_nonzero(a) + np.count_nonzero(b))
    return _fraction(2.0 * inter, total)


def overlap_at_threshold(
    saliency: Any,
    gt_mask: Any,
    threshold: float,
    *,
    mask_threshold: float = 0.0,
    z: float = Z_95,
) -> Dict[str, Any]:
    """Overlap of a binarized saliency map with a ground-truth mask, one image.

    The saliency map is binarized with ``value >= threshold`` (see
    :func:`binarize`); the ground-truth mask with ``value > mask_threshold`` (see
    :func:`as_binary_mask`).

    Returns
    -------
    dict
        ``{"threshold", "decision_rule", "counts": {tp, fp, fn, tn},
        "n_pixels", "predicted_pixels", "true_pixels", "predicted_fraction",
        "true_fraction", "iou", "dice", "precision", "recall",
        "empty_ground_truth", "undefined_reason", "interval_caveat"}``.

        ``precision`` and ``recall`` are :func:`vitreous.clinical.rate_with_ci`
        dicts (value + Wilson interval + support). Their denominators count
        **pixels**, so both carry :data:`PIXEL_INTERVAL_CAVEAT`: neighbouring
        pixels are not independent and the interval is therefore optimistically
        narrow. ``iou``/``dice`` are plain floats (or ``None``) with no interval —
        they are not binomial proportions.

    Empty ground truth
    ------------------
    A lesion-free image (or an absent dermoscopic criterion) has an all-zero
    mask. Then ``empty_ground_truth`` is ``True``, ``recall`` is ``None`` with
    an ``undefined_reason`` ("ground-truth mask is empty..."), and IoU/Dice are
    ``0.0`` if anything was predicted or ``None`` if nothing was. Never a crash,
    never a silent zero.

    Raises
    ------
    ValueError
        If the shapes differ (the message points at :func:`resize_map`), or if
        the saliency map contains NaN/inf.
    """
    s = as_saliency_map(saliency)
    g = as_binary_mask(gt_mask, threshold=mask_threshold, name="gt_mask")
    _check_same_shape(s, g, "saliency", "gt_mask")
    thr = float(threshold)
    p = s >= thr

    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(np.count_nonzero(~p & ~g))
    n_pix = int(s.size)
    n_pred, n_true = tp + fp, tp + fn

    prec = rate_with_ci(tp, n_pred, z=z, name="pixel_precision")
    rec = rate_with_ci(tp, n_true, z=z, name="pixel_recall")
    if prec["value"] is None:
        prec["undefined_reason"] = f"no pixel reaches threshold {thr:g}"
    if rec["value"] is None:
        rec["undefined_reason"] = (
            "ground-truth mask is empty (a lesion-free image): recall is undefined"
        )
    prec["interval_caveat"] = PIXEL_INTERVAL_CAVEAT
    rec["interval_caveat"] = PIXEL_INTERVAL_CAVEAT

    union = float(tp + fp + fn)
    i_val = _fraction(float(tp), union)
    d_val = _fraction(2.0 * tp, float(n_pred + n_true))
    undefined: Optional[str] = None
    if i_val is None:
        undefined = (
            f"nothing predicted at threshold {thr:g} and nothing annotated: "
            f"IoU/Dice are 0/0. Reporting 1.0 here (as some libraries do) would "
            f"credit a perfect score for predicting nothing"
        )

    return {
        "threshold": thr,
        "decision_rule": "pixel is predicted iff saliency >= threshold",
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n_pixels": n_pix,
        "predicted_pixels": n_pred,
        "true_pixels": n_true,
        "predicted_fraction": float(n_pred / n_pix),
        "true_fraction": float(n_true / n_pix),
        "iou": i_val,
        "dice": d_val,
        "precision": prec,
        "recall": rec,
        "empty_ground_truth": bool(n_true == 0),
        "undefined_reason": undefined,
        "interval_caveat": PIXEL_INTERVAL_CAVEAT,
    }


def _threshold_grid(
    maps: Sequence[np.ndarray],
    thresholds: Optional[Sequence[float]],
    num_points: int,
) -> np.ndarray:
    if thresholds is not None:
        grid = np.asarray(thresholds, dtype=np.float64).ravel()
        if grid.size == 0:
            raise ValueError("thresholds is empty")
        if not np.all(np.isfinite(grid)):
            raise ValueError("thresholds contains NaN/inf")
        return grid
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")
    lo = min(float(m.min()) for m in maps)
    hi = max(float(m.max()) for m in maps)
    if lo >= 0.0 and hi <= 1.0:
        return np.linspace(0.0, 1.0, int(num_points))
    if hi <= lo:
        return np.linspace(lo, lo + 1.0, int(num_points))
    return np.linspace(lo, hi, int(num_points))


def _best_over_threshold(
    grid: np.ndarray,
    values: Sequence[Optional[float]],
    criterion: str,
) -> Dict[str, Any]:
    """Argmax over a threshold sweep, stamped with its own optimism."""
    best_i: Optional[int] = None
    best_v: Optional[float] = None
    for i, v in enumerate(values):
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_i, best_v = i, float(v)
    return {
        "criterion": criterion,
        "value": best_v,
        "threshold": float(grid[best_i]) if best_i is not None else None,
        "threshold_selection": OPTIMISTIC_THRESHOLD_NOTE,
        "warning": (
            f"this {criterion} is the maximum over {len(grid)} thresholds chosen "
            f"on the very data it is reported on. That is model selection on the "
            f"test set: it is optimistically biased upward and must be quoted as "
            f"'best achievable', never as an operating-point result. For an "
            f"honest figure, select the threshold on one split and evaluate it on "
            f"a held-out split."
        ),
        "undefined_reason": (
            None if best_v is not None else "no threshold produced a defined value"
        ),
    }


def overlap_threshold_sweep(
    saliency: Any,
    gt_mask: Any,
    *,
    thresholds: Optional[Sequence[float]] = None,
    num_points: int = 51,
    mask_threshold: float = 0.0,
    z: float = Z_95,
) -> Dict[str, Any]:
    """Sweep the binarization threshold for one image; report the best overlap.

    Attributions are continuous, so any single IoU/Dice figure is really a
    statement about a threshold that somebody chose. This sweeps a grid and
    returns the whole curve **plus** the best achievable Dice and IoU and the
    thresholds that achieved them.

    .. warning::
       **A best-over-threshold figure is optimistically biased.** Picking the
       threshold that maximizes Dice on the same data you report it on is model
       selection on the test set; the number is an upper bound on what a fixed
       threshold would deliver, not an achievable operating point. Both
       ``best_dice`` and ``best_iou`` are stamped
       ``"threshold_selection": "best-on-this-data (optimistic)"`` and carry a
       ``warning`` string, so the caller cannot quote them as held-out results.
       To report honestly, choose the threshold on one split and evaluate on
       another (see :func:`overlap_dataset`, which does the same at dataset
       level, and note that a threshold chosen there is still in-sample).

    ``thresholds`` defaults to ``num_points`` evenly spaced values over
    ``[0, 1]`` when the map lies in ``[0, 1]``, else over its observed range.
    Sweeping is ``O(num_points · H · W)``.
    """
    s = as_saliency_map(saliency)
    g = as_binary_mask(gt_mask, threshold=mask_threshold, name="gt_mask")
    _check_same_shape(s, g, "saliency", "gt_mask")
    grid = _threshold_grid([s], thresholds, num_points)

    points: List[Dict[str, Any]] = []
    dices: List[Optional[float]] = []
    ious: List[Optional[float]] = []
    for thr in grid:
        r = overlap_at_threshold(s, g, float(thr), z=z)
        dices.append(r["dice"])
        ious.append(r["iou"])
        points.append(
            {
                "threshold": float(thr),
                "iou": r["iou"],
                "dice": r["dice"],
                "precision": r["precision"]["value"],
                "recall": r["recall"]["value"],
                "predicted_fraction": r["predicted_fraction"],
            }
        )

    return {
        "n_thresholds": int(grid.size),
        "thresholds": [float(t) for t in grid],
        "points": points,
        "best_dice": _best_over_threshold(grid, dices, "dice"),
        "best_iou": _best_over_threshold(grid, ious, "iou"),
        "threshold_selection": OPTIMISTIC_THRESHOLD_NOTE,
        "empty_ground_truth": bool(not g.any()),
    }


def overlap_dataset(
    saliency_maps: Sequence[Any],
    gt_masks: Sequence[Any],
    *,
    threshold: float = 0.5,
    thresholds: Optional[Sequence[float]] = None,
    num_points: int = 51,
    mask_threshold: float = 0.0,
    z: float = Z_95,
) -> Dict[str, Any]:
    """Overlap across a dataset: at a fixed threshold, **and** swept.

    Two readouts, deliberately side by side:

    * ``at_threshold`` — mean/median IoU and Dice at the caller's fixed
      ``threshold``, i.e. what a shipped, pre-committed operating point
      actually delivers. This is the number to report.
    * ``best_over_threshold`` — the single shared threshold that maximizes mean
      Dice (and, separately, mean IoU) **over this same data**, with the
      achieved value. Stamped optimistic (see
      :func:`overlap_threshold_sweep`); it is an upper bound, useful for
      answering "is a better threshold even available?", not a result.

    Images whose ground-truth mask is empty contribute no IoU/Dice to the means
    (the quantity is undefined there, not zero); they are counted in
    ``n_empty_ground_truth``. Per-image values are returned in
    ``per_image_at_threshold`` so a caller can see the distribution rather than
    only its mean.

    Thresholds are applied in each map's own units — if the maps do not share a
    scale, pass them through :func:`normalize_map` first, or the swept threshold
    is meaningless across images.
    """
    maps, masks = _as_pairs(saliency_maps, gt_masks, mask_threshold=mask_threshold)
    grid = _threshold_grid(maps, thresholds, num_points)

    per_image: List[Dict[str, Any]] = []
    fixed_iou: List[Optional[float]] = []
    fixed_dice: List[Optional[float]] = []
    for s, g in zip(maps, masks):
        r = overlap_at_threshold(s, g, float(threshold), z=z)
        fixed_iou.append(r["iou"] if g.any() else None)
        fixed_dice.append(r["dice"] if g.any() else None)
        per_image.append(
            {
                "iou": r["iou"],
                "dice": r["dice"],
                "precision": r["precision"]["value"],
                "recall": r["recall"]["value"],
                "empty_ground_truth": r["empty_ground_truth"],
            }
        )

    mean_dice: List[Optional[float]] = []
    mean_iou: List[Optional[float]] = []
    curve: List[Dict[str, Any]] = []
    for thr in grid:
        d_vals: List[Optional[float]] = []
        i_vals: List[Optional[float]] = []
        for s, g in zip(maps, masks):
            if not g.any():
                continue
            p = s >= float(thr)
            tp = float(np.count_nonzero(p & g))
            n_pred = float(np.count_nonzero(p))
            n_true = float(np.count_nonzero(g))
            d_vals.append(_fraction(2.0 * tp, n_pred + n_true))
            i_vals.append(_fraction(tp, n_pred + n_true - tp))
        d_stat = _summary_stats(d_vals)
        i_stat = _summary_stats(i_vals)
        mean_dice.append(d_stat["mean"])
        mean_iou.append(i_stat["mean"])
        curve.append(
            {
                "threshold": float(thr),
                "mean_dice": d_stat["mean"],
                "mean_iou": i_stat["mean"],
                "n": d_stat["n"],
            }
        )

    n_empty = int(sum(1 for g in masks if not g.any()))
    return {
        "n_images": len(maps),
        "n_empty_ground_truth": n_empty,
        "empty_ground_truth_policy": (
            "images with an all-zero ground-truth mask are excluded from the "
            "IoU/Dice means (both are undefined there, not zero) and counted here"
        ),
        "at_threshold": {
            "threshold": float(threshold),
            "decision_rule": "pixel is predicted iff saliency >= threshold",
            "threshold_selection": "fixed by the caller (not tuned on this data)",
            "iou": _summary_stats(fixed_iou, name="iou"),
            "dice": _summary_stats(fixed_dice, name="dice"),
        },
        "per_image_at_threshold": per_image,
        "sweep": {
            "n_thresholds": int(grid.size),
            "thresholds": [float(t) for t in grid],
            "points": curve,
        },
        "best_over_threshold": {
            "dice": _best_over_threshold(grid, mean_dice, "mean dice"),
            "iou": _best_over_threshold(grid, mean_iou, "mean iou"),
            "threshold_selection": OPTIMISTIC_THRESHOLD_NOTE,
        },
    }


# --------------------------------------------------------------------------- #
# (b) pointing game — the standard weakly-supervised localization metric
# --------------------------------------------------------------------------- #


def pointing_game_hit(
    saliency: Any,
    gt_mask: Any,
    *,
    tolerance: float = 0.0,
    mask_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Does the saliency peak land in (or within ``tolerance`` px of) the mask?

    The pointing game is the standard weakly-supervised localization metric
    (Zhang et al. 2016): reduce the whole attribution to its single most-salient
    pixel and ask whether that point falls inside the annotation. It is coarse
    on purpose — it is insensitive to the map's scale, needs no binarization
    threshold, and answers exactly the CADe question "would the mark this system
    drops be on the lesion?".

    Parameters
    ----------
    tolerance:
        Hit if the peak is within this many pixels (**Euclidean**) of the nearest
        annotated pixel. ``0.0`` (default) means strictly inside the mask. A
        tolerance is the honest way to score a coarse map: a ``[14, 14]`` patch
        grid upsampled to ``[224, 224]`` cannot localize better than ±8 px, so
        scoring it at 0 px tolerance penalizes the grid, not the model. Report
        the tolerance you used, always.

    Returns
    -------
    dict
        ``{"hit", "peak_row", "peak_col", "peak_value", "distance",
        "tolerance", "empty_ground_truth", "undefined_reason"}``. On an empty
        ground-truth mask ``hit`` is ``None`` (**not** ``False``) with a reason:
        an image with nothing annotated can be neither hit nor missed, and
        scoring it as a miss would drag a dataset hit rate down for images that
        do not belong in it at all.
    """
    s = as_saliency_map(saliency)
    g = as_binary_mask(gt_mask, threshold=mask_threshold, name="gt_mask")
    _check_same_shape(s, g, "saliency", "gt_mask")
    tol = float(tolerance)
    if tol < 0:
        raise ValueError(f"tolerance must be >= 0, got {tol}")

    r, c = peak_location(s)
    peak_value = float(s[r, c])
    if not g.any():
        return {
            "hit": None,
            "peak_row": r,
            "peak_col": c,
            "peak_value": peak_value,
            "distance": None,
            "tolerance": tol,
            "empty_ground_truth": True,
            "undefined_reason": (
                "ground-truth mask is empty (a lesion-free image): the peak can "
                "neither hit nor miss it, so this image is undefined for the "
                "pointing game rather than scored as a miss"
            ),
        }
    if bool(g[r, c]):
        dist = 0.0
    else:
        nearest = distance_to_mask((r, c), g)  # never None: the mask is non-empty
        dist = float(nearest) if nearest is not None else 0.0
    return {
        "hit": bool(dist <= tol),
        "peak_row": r,
        "peak_col": c,
        "peak_value": peak_value,
        "distance": dist,
        "tolerance": tol,
        "empty_ground_truth": False,
        "undefined_reason": None,
    }


def pointing_game(
    saliency_maps: Sequence[Any],
    gt_masks: Sequence[Any],
    *,
    tolerance: float = 0.0,
    mask_threshold: float = 0.0,
    z: float = Z_95,
) -> Dict[str, Any]:
    """Pointing-game hit rate over a dataset, with a Wilson interval and ``n``.

    The hit rate is a genuine binomial proportion over **images** (one
    independent hit/miss per image), so unlike IoU/Dice it takes a Wilson
    interval honestly — see :func:`vitreous.clinical.rate_with_ci`, which this
    calls rather than reimplementing. ``n`` is the number of images that
    *could* be scored: images with an empty ground-truth mask are excluded and
    reported separately in ``n_empty_ground_truth``.

    Returns
    -------
    dict
        ``{"hit_rate", "n_images", "n_scored", "n_empty_ground_truth",
        "tolerance", "distance", "per_image", "metric", "note"}`` where
        ``hit_rate`` is the rate dict (value + Wilson lo/hi + n + successes) and
        ``distance`` summarizes the peak-to-mask distances (in pixels) across
        scored images — a hit rate of 0.6 with a median miss distance of 3 px is
        a very different result from one with a median of 90 px.
    """
    maps, masks = _as_pairs(saliency_maps, gt_masks, mask_threshold=mask_threshold)
    per_image = [
        pointing_game_hit(s, g, tolerance=tolerance, mask_threshold=mask_threshold)
        for s, g in zip(maps, masks)
    ]
    scored = [r for r in per_image if r["hit"] is not None]
    hits = int(sum(1 for r in scored if r["hit"]))
    rate = rate_with_ci(hits, len(scored), z=z, name="pointing_game_hit_rate")
    if rate["value"] is None:
        rate["undefined_reason"] = (
            "no image had a non-empty ground-truth mask; the pointing game is "
            "undefined on a dataset with nothing annotated"
        )
    return {
        "metric": "pointing game",
        "definition": (
            "hit iff the argmax of the saliency map is within `tolerance` pixels "
            "(Euclidean) of the ground-truth mask; tolerance 0 means strictly "
            "inside it. Argmax ties break row-major (deterministic)."
        ),
        "tolerance": float(tolerance),
        "hit_rate": rate,
        "n_images": len(maps),
        "n_scored": len(scored),
        "n_empty_ground_truth": len(per_image) - len(scored),
        "distance": _summary_stats(
            [r["distance"] for r in scored], name="peak_to_mask_distance_px"
        ),
        "per_image": per_image,
        "note": (
            "this rate counts images, not pixels, so its Wilson interval is not "
            "subject to the spatial-correlation caveat that pixel-level "
            "precision/recall carry"
        ),
    }


# --------------------------------------------------------------------------- #
# (c) attribution mass / concentration — the shortcut detector
# --------------------------------------------------------------------------- #


def attribution_mass(
    saliency: Any,
    region: Any,
    *,
    mask_threshold: float = 0.0,
    region_name: str = "region",
) -> Dict[str, Any]:
    """Fraction of total **positive** attribution mass falling inside ``region``.

    ``fraction_inside = Σ max(saliency, 0) over region / Σ max(saliency, 0)``.

    Only positive attribution is counted. A signed method (Integrated Gradients,
    LRP) also produces negative attribution — evidence *against* the target
    class — and summing signed values would let a strong negative region cancel a
    strong positive one and report a meaningless ratio. The discarded negative
    mass is still reported (``negative_mass``) so the caller can see how much was
    set aside.

    **``fraction_inside`` alone is not evidence.** A lesion filling 65% of the
    frame collects 65% of the mass from a uniform map that knows nothing. That is
    why ``concentration`` is returned beside it:

    ``concentration = fraction_inside / area_fraction``

    — the mass multiple relative to chance. ``1.0`` means "exactly as much mass
    as its area would get anyway" (no localization at all); ``> 1`` means the
    attribution really is concentrated there. Quote the pair, never the fraction
    alone.

    Returns
    -------
    dict
        ``{"region", "mass_inside", "mass_total", "fraction_inside",
        "area_pixels", "n_pixels", "area_fraction", "concentration",
        "negative_mass", "empty_region", "undefined_reason"}``. ``None`` (never
        ``NaN``) whenever the map has no positive mass at all, or the region is
        empty, each with its own reason.
    """
    s = as_saliency_map(saliency)
    r = as_binary_mask(region, threshold=mask_threshold, name="region")
    _check_same_shape(s, r, "saliency", "region")

    pos = np.clip(s, 0.0, None)
    total = float(pos.sum())
    inside = float(pos[r].sum())
    negative = float(np.clip(-s, 0.0, None).sum())
    n_pix = int(s.size)
    area = int(np.count_nonzero(r))
    area_fraction = float(area / n_pix)

    frac = _fraction(inside, total)
    conc: Optional[float] = None
    undefined: Optional[str] = None
    if area == 0:
        # 0.0 would be arithmetically true and epistemically false: it would read
        # as "the model put no mass on the lesion" when there is no lesion.
        frac = None
        undefined = (
            f"{region_name} is empty (no annotated pixels): the fraction of mass "
            f"inside it is undefined, not 0.0 — there is no region to be inside"
        )
    elif total <= 0.0:
        undefined = (
            "the saliency map has no positive mass (all values <= 0): the "
            "fraction of a zero total is undefined"
        )
    if frac is not None and area_fraction > 0.0:
        conc = float(frac / area_fraction)

    return {
        "region": region_name,
        "mass_inside": inside,
        "mass_total": total,
        "fraction_inside": frac,
        "area_pixels": area,
        "n_pixels": n_pix,
        "area_fraction": area_fraction,
        "concentration": conc,
        "concentration_meaning": (
            "fraction_inside / area_fraction — 1.0 is exactly chance for a "
            "uniform map; > 1 means the attribution is concentrated in the region"
        ),
        "negative_mass": negative,
        "positive_only": True,
        "empty_region": bool(area == 0),
        "undefined_reason": undefined,
    }


def mass_within_mask(
    saliency: Any, gt_mask: Any, *, mask_threshold: float = 0.0
) -> Dict[str, Any]:
    """:func:`attribution_mass` against the ground-truth lesion mask.

    This is the **primary endpoint** of the Tier-0 retrospective localization
    study in ``docs/VITREOUS-TRIAL.md`` §3: "X% of this model's attribution mass
    falls inside the dermatologist-annotated lesion". An empty mask (lesion-free
    image) yields ``fraction_inside = None`` with a reason, not ``0.0``.
    """
    out = attribution_mass(
        saliency, gt_mask, mask_threshold=mask_threshold, region_name="lesion"
    )
    if out["empty_region"]:
        out["undefined_reason"] = (
            "ground-truth mask is empty (a lesion-free image): there is no "
            "region for attribution mass to fall inside, so the fraction is "
            "undefined rather than 0.0"
        )
    return out


def mass_within_mask_dataset(
    saliency_maps: Sequence[Any],
    gt_masks: Sequence[Any],
    *,
    mask_threshold: float = 0.0,
    majority_bar: float = 0.5,
    z: float = Z_95,
) -> Dict[str, Any]:
    """Attribution-mass-within-mask across a dataset — mean **and** a proportion.

    Two numbers, because they answer different questions and only one of them
    takes a confidence interval:

    * ``fraction`` — the mean/median/spread of the per-image mass fractions. A
      mean of ratios, so no Wilson interval (see :func:`_summary_stats`).
    * ``majority_inside_rate`` — the fraction of **images** on which at least
      ``majority_bar`` (default 0.5) of the positive attribution mass lands
      inside the lesion. That *is* a binomial proportion over images, so it
      carries a Wilson interval and ``n``. It is also the number the UX vision
      asks for directly: "on N% of cases the mass falls predominantly on the
      image border, and there the explanation is flagged rather than shown".

    ``concentration`` (mass fraction ÷ area fraction) is summarized too, because
    a mean mass fraction cannot be read without knowing how much of the frame
    the lesions occupy.
    """
    maps, masks = _as_pairs(saliency_maps, gt_masks, mask_threshold=mask_threshold)
    bar = float(majority_bar)
    if not (0.0 <= bar <= 1.0):
        raise ValueError(f"majority_bar must be in [0, 1], got {bar}")

    per_image = [
        mass_within_mask(s, g, mask_threshold=mask_threshold)
        for s, g in zip(maps, masks)
    ]
    fracs = [r["fraction_inside"] for r in per_image]
    defined = [f for f in fracs if f is not None]
    n_major = int(sum(1 for f in defined if f >= bar))
    rate = rate_with_ci(n_major, len(defined), z=z, name="majority_inside_rate")
    if rate["value"] is None:
        rate["undefined_reason"] = (
            "no image had both a non-empty ground-truth mask and positive "
            "attribution mass"
        )
    return {
        "metric": "attribution mass within ground-truth mask",
        "n_images": len(maps),
        "n_scored": len(defined),
        "n_undefined": len(fracs) - len(defined),
        "fraction": _summary_stats(fracs, name="mass_fraction_inside_lesion"),
        "concentration": _summary_stats(
            [r["concentration"] for r in per_image], name="concentration"
        ),
        "area_fraction": _summary_stats(
            [r["area_fraction"] for r in per_image], name="lesion_area_fraction"
        ),
        "majority_bar": bar,
        "majority_inside_rate": rate,
        "per_image": [
            {
                "fraction_inside": r["fraction_inside"],
                "concentration": r["concentration"],
                "area_fraction": r["area_fraction"],
                "undefined_reason": r["undefined_reason"],
            }
            for r in per_image
        ],
        "note": (
            "fraction_inside must be read against area_fraction: a lesion filling "
            "65% of the frame collects 65% of the mass from a map that knows "
            "nothing. concentration is that normalization"
        ),
    }


def shortcut_report(
    saliency: Any,
    gt_mask: Any,
    distractor: Optional[Any] = None,
    *,
    border_fraction: float = 0.1,
    mask_threshold: float = 0.0,
    min_inside_fraction: float = 0.5,
    flag_at: float = 0.5,
) -> Dict[str, Any]:
    """Shortcut detector — lay question #2, "is that the right place to look?".

    ``docs/UX-VISION.md`` asks the UI to show plainly "✓ focused on the lesion"
    or "⚠ distracted by the image border — treat with caution". This computes
    both sides of that judgment: the attribution mass inside the annotated
    lesion, and the mass in a **distractor** region where dermoscopy's classic
    artifacts live — rulers, ink marks, sticker labels, vignetting from the
    contact plate, hair sweeping in from the edge.

    Parameters
    ----------
    distractor:
        Bool mask of the distractor region. Defaults to
        :func:`border_mask` with ``border_fraction``. **Any overlap with the
        lesion is removed before measuring**, so no pixel's mass is counted on
        both sides of the comparison (a lesion touching the frame would
        otherwise be scored as its own distractor); the removed pixel count is
        reported as ``distractor_overlap_pixels``.

    Two independent, separately documented bars — each testable on its own:

    * ``focused`` — ``fraction_inside_lesion >= min_inside_fraction`` (default
      0.5): the majority of the mass is on the lesion.
    * ``flagged`` — ``shortcut_score > flag_at`` (default 0.5) where
      ``shortcut_score = distractor_mass / (lesion_mass + distractor_mass)``:
      more mass lands in the distractor than on the lesion. ``0`` = all lesion,
      ``1`` = all distractor, ``0.5`` = parity. ``None`` (and never flagged)
      when neither region received any mass.

    ``message`` renders the verdict in plain language for the UI. Note the two
    bars can both be false (mass is elsewhere — hair, skin texture, nothing in
    particular), which is reported as an explicitly inconclusive verdict rather
    than as reassurance.
    """
    s = as_saliency_map(saliency)
    g = as_binary_mask(gt_mask, threshold=mask_threshold, name="gt_mask")
    _check_same_shape(s, g, "saliency", "gt_mask")
    if distractor is None:
        d = border_mask(s.shape, border_fraction=border_fraction)
        distractor_source = f"image border (border_fraction={float(border_fraction)})"
        distractor_label = "image border"
    else:
        d = as_binary_mask(distractor, threshold=mask_threshold, name="distractor")
        _check_same_shape(s, d, "saliency", "distractor")
        distractor_source = "caller-supplied"
        distractor_label = "distractor region"
    overlap = int(np.count_nonzero(d & g))
    d_eff = d & ~g

    lesion = attribution_mass(s, g, region_name="lesion")
    if lesion["empty_region"]:
        lesion["undefined_reason"] = (
            "ground-truth mask is empty (a lesion-free image): the fraction of "
            "mass inside it is undefined rather than 0.0"
        )
    distract = attribution_mass(s, d_eff, region_name="distractor")

    l_mass, d_mass = lesion["mass_inside"], distract["mass_inside"]
    denom = l_mass + d_mass
    score = _fraction(d_mass, denom)
    inside = lesion["fraction_inside"]
    focused = bool(inside is not None and inside >= float(min_inside_fraction))
    flagged = bool(score is not None and score > float(flag_at))

    if lesion["empty_region"]:
        message = (
            "no lesion is annotated on this image, so there is nothing to be "
            "focused on — the shortcut check does not apply"
        )
        verdict = "not applicable"
    elif score is None:
        message = (
            "the attribution map places no positive mass on either the lesion or "
            "the distractor region — nothing can be judged from it"
        )
        verdict = "inconclusive"
    elif flagged:
        message = (
            f"distracted by the {distractor_label}: {d_mass / denom:.0%} of the "
            f"attribution mass compared here landed outside the lesion — treat "
            f"this explanation with caution"
        )
        verdict = "distracted"
    elif focused:
        message = (
            f"focused on the lesion: {inside:.0%} of the model's positive "
            f"attribution mass falls inside the annotated lesion "
            f"({lesion['concentration']:.1f}x its area share)"
            if lesion.get("concentration") is not None
            else f"focused on the lesion: {inside:.0%} of the mass falls inside it"
        )
        verdict = "focused"
    else:
        message = (
            "neither clearly focused on the lesion nor clearly distracted by the "
            "distractor region — most of the mass is somewhere else entirely"
        )
        verdict = "inconclusive"

    return {
        "metric": "shortcut detector (attribution mass: lesion vs distractor)",
        "lay_question": "Is that the right place to look?",
        "lesion": lesion,
        "distractor": distract,
        "distractor_source": distractor_source,
        "distractor_overlap_pixels": overlap,
        "distractor_excludes_lesion": True,
        "fraction_inside_lesion": inside,
        "fraction_in_distractor": distract["fraction_inside"],
        "shortcut_score": score,
        "shortcut_score_meaning": (
            "distractor_mass / (lesion_mass + distractor_mass): 0 = all on the "
            "lesion, 1 = all in the distractor, 0.5 = parity"
        ),
        "min_inside_fraction": float(min_inside_fraction),
        "flag_at": float(flag_at),
        "focused": focused,
        "flagged": flagged,
        "verdict": verdict,
        "message": message,
    }


# --------------------------------------------------------------------------- #
# (d) FROC — detection *with* localization, for the multi-lesion case
# --------------------------------------------------------------------------- #


def assign_marks_to_lesions(
    points: Sequence[Sequence[int]],
    lesion_label_map: Any,
    *,
    tolerance: float = 0.0,
) -> List[int]:
    """Map candidate marks onto lesion ids — the input :func:`froc_curve` needs.

    ``lesion_label_map`` is an integer ``[H, W]`` map where ``0`` is background
    and each distinct positive value is one lesion (an ISIC Task 1 mask with a
    single lesion is just ``mask.astype(int)``). Each point ``(row, col)``
    returns the label it lands on, the nearest label within ``tolerance``
    pixels, or ``-1`` for a **false mark** (lands on background).

    Ties (two lesions equidistant) resolve to the smaller label, so the mapping
    is deterministic.
    """
    lab = np.asarray(lesion_label_map)
    if lab.ndim != 2:
        raise ValueError(f"lesion_label_map must be 2-D [H, W], got {lab.shape}")
    if not np.issubdtype(lab.dtype, np.integer):
        if lab.dtype == np.bool_:
            lab = lab.astype(np.int64)
        else:
            f = np.asarray(lab, dtype=np.float64)
            if not np.all(np.isfinite(f)) or not np.all(np.equal(np.mod(f, 1), 0)):
                raise ValueError("lesion_label_map must contain integer labels")
            lab = f.astype(np.int64)
    tol = float(tolerance)
    if tol < 0:
        raise ValueError(f"tolerance must be >= 0, got {tol}")
    h, w = lab.shape
    fg = np.argwhere(lab > 0)

    out: List[int] = []
    for pt in points:
        r, c = int(pt[0]), int(pt[1])
        if 0 <= r < h and 0 <= c < w and lab[r, c] > 0:
            out.append(int(lab[r, c]))
            continue
        if tol > 0 and fg.size:
            d = np.hypot(fg[:, 0] - r, fg[:, 1] - c)
            near = d <= tol
            if near.any():
                cand = lab[fg[near, 0], fg[near, 1]]
                dd = d[near]
                best = np.lexsort((cand, dd))[0]
                out.append(int(cand[best]))
                continue
        out.append(-1)
    return out


def froc_curve(
    mark_scores: Sequence[float],
    mark_lesion_ids: Sequence[int],
    *,
    n_images: int,
    n_lesions: int,
    thresholds: Optional[Sequence[float]] = None,
    operating_points: Sequence[float] = FROC_OPERATING_POINTS,
    z: float = Z_95,
) -> Dict[str, Any]:
    """FROC: sensitivity as a function of **false marks per image**.

    **Why FROC and not ROC.** ROC's unit of analysis is the *image* — one score,
    one binary truth, and specificity is well defined because "the negatives" are
    a fixed set of images. When the system emits *marks* (regions), an image can
    carry several true findings and any number of false ones, and there is no
    meaningful denominator for specificity: you cannot count the true negatives,
    because you cannot count the non-marks. FROC replaces the false-positive
    *rate* with a false-mark *count per image*, which is exactly the quantity a
    clinician experiences. That is why CADe devices are evaluated on FROC.

    Parameters
    ----------
    mark_scores:
        ``[M]`` confidence of each candidate mark, across the whole dataset.
    mark_lesion_ids:
        ``[M]`` the lesion each mark hits — any hashable-as-int label, with
        ``-1`` (or any negative value) meaning **false mark**. Build it with
        :func:`assign_marks_to_lesions`. Ids must be unique *across the
        dataset* (e.g. ``image_index * 1000 + lesion_index``), since a lesion is
        counted as detected once, globally.
    n_images:
        Number of images the marks came from — the denominator of "false marks
        per image". Includes images that produced no marks at all.
    n_lesions:
        Total number of annotated lesions — the denominator of sensitivity.
        Pass the count from the ground truth, **not** the number of distinct ids
        seen in ``mark_lesion_ids``: lesions that no mark ever touched are
        misses and must be in the denominator.
    operating_points:
        The false-marks-per-image budgets to read the curve at. Defaults to
        :data:`FROC_OPERATING_POINTS`, the conventional
        ``1/8 … 8`` octave ladder.

    Counting rules (stated because implementations differ)
    -----------------------------------------------------
    * A mark is kept iff ``score >= threshold``.
    * A lesion counts as detected **once**, however many kept marks land on it.
    * Duplicate marks on an already-detected lesion are **not** counted as false
      marks (the standard FROC convention). Only marks with a negative lesion id
      are false.

    Returns
    -------
    dict
        ``{"points": [...], "sensitivity_at_fp_per_image": [...], "froc_score",
        ...}``. Each point's ``sensitivity`` is a
        :func:`vitreous.clinical.rate_with_ci` dict over lesions.
        ``froc_score`` is the mean of the sensitivities achievable at the
        operating points (the conventional single-number summary); it is
        ``None`` if none of them is achievable, and the operating points that
        were not achievable are marked ``"achieved": False`` rather than
        silently dropped.

    Note on the interval: sensitivity's denominator counts lesions, and two
    lesions on the same image are not independent, so its Wilson interval is
    mildly optimistic when images carry multiple lesions.
    """
    scores = np.asarray(mark_scores, dtype=np.float64).ravel()
    ids = np.asarray(mark_lesion_ids).ravel()
    if scores.size != ids.size:
        raise ValueError(
            f"mark_scores has {scores.size} entries but mark_lesion_ids has {ids.size}"
        )
    if scores.size and not np.all(np.isfinite(scores)):
        raise ValueError("mark_scores contains NaN/inf")
    if not np.issubdtype(ids.dtype, np.integer):
        ids = np.asarray(ids, dtype=np.int64)
    n_img = int(n_images)
    n_les = int(n_lesions)
    if n_img < 1:
        raise ValueError(f"n_images must be >= 1, got {n_img}")
    if n_les < 0:
        raise ValueError(f"n_lesions must be >= 0, got {n_les}")

    if thresholds is None:
        grid = (
            np.unique(scores)
            if scores.size
            else np.array([0.0], dtype=np.float64)
        )
    else:
        grid = np.unique(np.asarray(thresholds, dtype=np.float64).ravel())
        if grid.size == 0:
            raise ValueError("thresholds is empty")
    grid = grid[::-1]  # strictest first → the curve runs left-to-right in FP/image

    points: List[Dict[str, Any]] = []
    for thr in grid:
        kept = scores >= float(thr)
        kept_ids = ids[kept]
        detected = int(np.unique(kept_ids[kept_ids >= 0]).size)
        false_marks = int(np.count_nonzero(kept_ids < 0))
        sens = rate_with_ci(detected, n_les, z=z, name="froc_sensitivity")
        if sens["value"] is None:
            sens["undefined_reason"] = "no annotated lesions: sensitivity is undefined"
        points.append(
            {
                "threshold": float(thr),
                "n_marks_kept": int(np.count_nonzero(kept)),
                "n_lesions_detected": detected,
                "n_false_marks": false_marks,
                "false_marks_per_image": float(false_marks / n_img),
                "sensitivity": sens["value"],
                "sensitivity_lo": sens["lo"],
                "sensitivity_hi": sens["hi"],
                "sensitivity_n": sens["n"],
            }
        )

    at_fp: List[Dict[str, Any]] = []
    achieved_sens: List[float] = []
    for target in operating_points:
        t = float(target)
        best: Optional[Dict[str, Any]] = None
        for p in points:
            if p["false_marks_per_image"] <= t and p["sensitivity"] is not None:
                if best is None or p["sensitivity"] > best["sensitivity"]:
                    best = p
        if best is None:
            at_fp.append(
                {
                    "fp_per_image": t,
                    "achieved": False,
                    "sensitivity": None,
                    "threshold": None,
                    "undefined_reason": (
                        "no threshold keeps false marks per image at or below "
                        f"{t:g} with a defined sensitivity"
                    ),
                }
            )
        else:
            achieved_sens.append(float(best["sensitivity"]))
            at_fp.append(
                {
                    "fp_per_image": t,
                    "achieved": True,
                    "sensitivity": best["sensitivity"],
                    "sensitivity_lo": best["sensitivity_lo"],
                    "sensitivity_hi": best["sensitivity_hi"],
                    "threshold": best["threshold"],
                    "false_marks_per_image": best["false_marks_per_image"],
                    "undefined_reason": None,
                }
            )

    return {
        "metric": "FROC (free-response ROC)",
        "why_froc": (
            "FROC replaces ROC when the unit of analysis is a mark rather than "
            "an image: an image can contain several true findings and any number "
            "of false marks, so there is no countable set of true negatives and "
            "no specificity. The x-axis is false marks per image — the quantity a "
            "clinician actually experiences — which is why CADe devices are "
            "evaluated this way."
        ),
        "counting_rule": (
            "a mark is kept iff score >= threshold; a lesion counts as detected "
            "once however many marks land on it; duplicate marks on an "
            "already-detected lesion are not counted as false marks"
        ),
        "n_images": n_img,
        "n_lesions": n_les,
        "n_marks": int(scores.size),
        "n_false_marks_total": int(np.count_nonzero(ids < 0)),
        "operating_points": [float(t) for t in operating_points],
        "points": points,
        "sensitivity_at_fp_per_image": at_fp,
        "froc_score": float(np.mean(achieved_sens)) if achieved_sens else None,
        "froc_score_definition": (
            "mean sensitivity over the achievable operating points listed in "
            "operating_points; None when none of them is achievable"
        ),
        "interval_caveat": (
            "sensitivity's denominator counts lesions; two lesions in the same "
            "image are not independent, so the Wilson interval is mildly "
            "optimistic when images carry multiple lesions"
        ),
    }


# --------------------------------------------------------------------------- #
# shared sequence coercion
# --------------------------------------------------------------------------- #


def _as_item_list(items: Any) -> List[Any]:
    """A sequence of per-image arrays, accepting a stacked ``[N, H, W]`` array."""
    if isinstance(items, np.ndarray):
        if items.ndim != 3:
            raise ValueError(
                f"a stacked array must be [N, H, W], got shape {items.shape}"
            )
        return list(items)
    return list(items)


def _as_pairs(
    saliency_maps: Sequence[Any],
    gt_masks: Sequence[Any],
    *,
    mask_threshold: float = 0.0,
    resize: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Coerce aligned sequences of maps + masks, checking shapes pairwise."""
    maps_in = _as_item_list(saliency_maps)
    masks_in = _as_item_list(gt_masks)
    if len(maps_in) != len(masks_in):
        raise ValueError(
            f"got {len(maps_in)} saliency maps but {len(masks_in)} ground-truth masks"
        )
    if not maps_in:
        raise ValueError("no images given; no localization metric is defined on zero images")
    maps: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for i, (s, g) in enumerate(zip(maps_in, masks_in)):
        sm = as_saliency_map(s, name=f"saliency[{i}]")
        gm = as_binary_mask(g, threshold=mask_threshold, name=f"gt_mask[{i}]")
        if resize is not None and sm.shape != gm.shape:
            sm = resize_map(sm, gm.shape, mode=resize)
        _check_same_shape(sm, gm, f"saliency[{i}]", f"gt_mask[{i}]")
        maps.append(sm)
        masks.append(gm)
    return maps, masks


# --------------------------------------------------------------------------- #
# (e) the bundle-ready report + the one-line summary
# --------------------------------------------------------------------------- #


def localization_report(
    saliency_maps: Sequence[Any],
    gt_masks: Sequence[Any],
    *,
    threshold: float = 0.5,
    thresholds: Optional[Sequence[float]] = None,
    num_points: int = 51,
    tolerance: float = 0.0,
    mask_threshold: float = 0.0,
    distractor_masks: Optional[Sequence[Any]] = None,
    border_fraction: float = 0.1,
    majority_bar: float = 0.5,
    resize: Optional[str] = None,
    marks: Optional[Mapping[str, Any]] = None,
    z: float = Z_95,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything above, computed once, as one JSON-serializable provenance blob.

    This is the function a notebook or pack builder should call for a Tier-0
    localization study (``docs/VITREOUS-TRIAL.md`` §3). It mirrors
    :func:`vitreous.clinical.clinical_report`: one dict, all sections, plus the
    one-line :func:`format_localization_summary` sentence under ``"summary"``.

    Parameters
    ----------
    saliency_maps, gt_masks:
        Aligned sequences (or ``[N, H, W]`` arrays). Shapes must match pairwise
        unless ``resize`` is set.
    threshold:
        The **fixed, pre-committed** binarization threshold for the overlap
        section. Pass the one your product actually uses.
    tolerance:
        Pointing-game tolerance in pixels. The report always includes the strict
        (0 px) hit rate as well, so a tolerant figure can never be quoted without
        its strict counterpart.
    distractor_masks:
        Per-image distractor regions for the shortcut detector. Defaults to the
        image border (:func:`border_mask` with ``border_fraction``).
    resize:
        ``None`` (default) requires matching shapes and raises otherwise;
        ``"nearest"`` / ``"bilinear"`` resizes each saliency map to its mask's
        shape via :func:`resize_map`. Whatever is used is recorded in
        ``provenance["resize"]`` — a resize is a processing decision and must
        travel with the numbers.
    marks:
        Optional mapping forwarded to :func:`froc_curve` — keys
        ``mark_scores``, ``mark_lesion_ids``, ``n_images``, ``n_lesions`` (plus
        optional ``thresholds`` / ``operating_points``). Omit for the
        single-region case; FROC only says something when marks are the unit of
        analysis.

    Returns
    -------
    dict
        ``{"localization_schema_version", "task", "n_images",
        "n_empty_ground_truth", "mass_within_mask", "shortcut", "pointing_game",
        "pointing_game_tolerant", "overlap", "froc", "provenance", "summary",
        "caveats"}``.
    """
    maps, masks = _as_pairs(
        saliency_maps, gt_masks, mask_threshold=mask_threshold, resize=resize
    )
    n = len(maps)

    if distractor_masks is None:
        distractors: List[Optional[np.ndarray]] = [None] * n
    else:
        dl = _as_item_list(distractor_masks)
        if len(dl) != n:
            raise ValueError(f"got {len(dl)} distractor masks but {n} images")
        distractors = [
            as_binary_mask(d, threshold=mask_threshold, name=f"distractor[{i}]")
            for i, d in enumerate(dl)
        ]

    shortcuts = [
        shortcut_report(
            s,
            g,
            d,
            border_fraction=border_fraction,
            mask_threshold=mask_threshold,
        )
        for s, g, d in zip(maps, masks, distractors)
    ]
    n_flagged = int(sum(1 for r in shortcuts if r["flagged"]))
    n_judgeable = int(sum(1 for r in shortcuts if r["verdict"] != "not applicable"))
    flag_rate = rate_with_ci(n_flagged, n_judgeable, z=z, name="shortcut_flag_rate")
    if flag_rate["value"] is None:
        flag_rate["undefined_reason"] = "no image had an annotated lesion to judge"

    strict = pointing_game(maps, masks, tolerance=0.0, mask_threshold=mask_threshold, z=z)
    tolerant = (
        pointing_game(maps, masks, tolerance=tolerance, mask_threshold=mask_threshold, z=z)
        if tolerance > 0
        else None
    )

    froc = None
    if marks is not None:
        kwargs = dict(marks)
        required = ("mark_scores", "mark_lesion_ids", "n_lesions")
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(
                f"marks is missing required key(s) {missing}; it needs "
                f"'mark_scores', 'mark_lesion_ids' and 'n_lesions' (see froc_curve)"
            )
        froc = froc_curve(
            kwargs.pop("mark_scores"),
            kwargs.pop("mark_lesion_ids"),
            n_images=int(kwargs.pop("n_images", n)),
            n_lesions=int(kwargs.pop("n_lesions")),
            thresholds=kwargs.pop("thresholds", None),
            operating_points=kwargs.pop("operating_points", FROC_OPERATING_POINTS),
            z=z,
        )
        if kwargs:
            raise ValueError(f"unknown keys in marks: {sorted(kwargs)}")

    report: Dict[str, Any] = {
        "localization_schema_version": LOCALIZATION_SCHEMA_VERSION,
        "task": (
            "CADe — computer-aided detection: does the mark land on the region a "
            "clinician should examine? This is not CADx: nothing here is a "
            "statement about whether the lesion is malignant, and none of these "
            "quantities has a prevalence term, because all of them are computed "
            "conditional on a lesion being present in the image."
        ),
        "n_images": n,
        "n_empty_ground_truth": int(sum(1 for g in masks if not g.any())),
        "mass_within_mask": mass_within_mask_dataset(
            maps, masks, mask_threshold=mask_threshold, majority_bar=majority_bar, z=z
        ),
        "shortcut": {
            "metric": "shortcut detector (attribution mass: lesion vs distractor)",
            "lay_question": "Is that the right place to look?",
            "n_images": n,
            "n_judged": n_judgeable,
            "n_flagged": n_flagged,
            "flag_rate": flag_rate,
            "distractor_source": shortcuts[0]["distractor_source"] if shortcuts else None,
            "shortcut_score": _summary_stats(
                [r["shortcut_score"] for r in shortcuts], name="shortcut_score"
            ),
            "fraction_in_distractor": _summary_stats(
                [r["fraction_in_distractor"] for r in shortcuts],
                name="mass_fraction_in_distractor",
            ),
            "per_image": [
                {
                    "verdict": r["verdict"],
                    "focused": r["focused"],
                    "flagged": r["flagged"],
                    "fraction_inside_lesion": r["fraction_inside_lesion"],
                    "fraction_in_distractor": r["fraction_in_distractor"],
                    "shortcut_score": r["shortcut_score"],
                    "message": r["message"],
                }
                for r in shortcuts
            ],
        },
        "pointing_game": strict,
        "pointing_game_tolerant": tolerant,
        "overlap": overlap_dataset(
            maps,
            masks,
            threshold=threshold,
            thresholds=thresholds,
            num_points=num_points,
            mask_threshold=mask_threshold,
            z=z,
        ),
        "froc": froc,
        "caveats": [
            "localization measures where the model looked, not whether its answer "
            "was right; report it alongside classification metrics "
            "(vitreous.clinical), never instead of them",
            "this is complementary to faithfulness (vitreous.xai.eval."
            "deletion_insertion): an explanation can be perfectly faithful to a "
            "model that is looking at a ruler",
            "the prevalence escape holds only conditional on a lesion being "
            "present; whole-body screening puts false marks per image back in "
            "play, which is what the froc section measures",
            f"best-over-threshold figures are {OPTIMISTIC_THRESHOLD_NOTE} and are "
            "labelled as such wherever they appear",
            "pixel-level precision/recall intervals assume independent pixels and "
            "are optimistically narrow; image-level rates are not affected",
        ],
        "provenance": {
            **dict(provenance or {}),
            "resize": resize,
            "mask_threshold": float(mask_threshold),
            "binarization_threshold": float(threshold),
            "pointing_game_tolerance_px": float(tolerance),
            "border_fraction": float(border_fraction),
            "majority_bar": float(majority_bar),
            "ground_truth_expected": (
                "binary lesion masks aligned pixel-for-pixel with the saliency "
                "maps — e.g. ISIC 2018 Task 1 (lesion segmentation) or Task 2 "
                "(per-attribute masks over 2,594 HAM10000 images)"
            ),
        },
    }
    report["summary"] = format_localization_summary(report)
    return report


def _fmt(v: Optional[float], places: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{places}f}"


def format_localization_summary(report: Dict[str, Any]) -> str:
    """The one plain sentence a UI or notebook should print.

    Example::

        attribution mass inside lesion 0.72 (mean over 40 images, lesion area
        0.18 of frame, 4.0x chance); majority-inside on 0.85 of images (95% CI
        0.71-0.93, n=40); pointing-game hit rate 0.90 (95% CI 0.77-0.96, n=40);
        border-distracted on 0.05 of images; Dice 0.41 at fixed threshold 0.50
        (best achievable 0.48 at 0.35 — best-on-this-data (optimistic))

    Two pairings are load-bearing and always emitted together when available:

    * **mass fraction with the lesion's area share.** A mass fraction is
      uninterpretable alone — a lesion filling most of the frame collects most
      of the mass from a map that knows nothing — so the area fraction and the
      chance multiple travel with it.
    * **the fixed-threshold Dice with the best-over-threshold one, and the
      best one is explicitly marked optimistic.** A caller copying this
      sentence cannot accidentally present a tuned number as an operating point.

    Accepts the dict from :func:`localization_report`; tolerates missing
    sections and returns ``"no localization metrics available"`` for an empty
    one.
    """
    parts: List[str] = []

    mass = report.get("mass_within_mask") or {}
    frac = mass.get("fraction") or {}
    if frac.get("mean") is not None:
        area = (mass.get("area_fraction") or {}).get("mean")
        conc = (mass.get("concentration") or {}).get("mean")
        head = (
            f"attribution mass inside lesion {_fmt(frac.get('mean'))} "
            f"(mean over {frac.get('n')} images"
        )
        if area is not None:
            head += f", lesion area {_fmt(area)} of frame"
        if conc is not None:
            head += f", {conc:.1f}x chance"
        head += ")"
        parts.append(head)
    maj = mass.get("majority_inside_rate") or {}
    if maj.get("value") is not None:
        lvl = int(round(float(maj.get("ci_level", 0.95)) * 100))
        parts.append(
            f"majority-inside on {_fmt(maj.get('value'))} of images "
            f"({lvl}% CI {_fmt(maj.get('lo'))}-{_fmt(maj.get('hi'))}, n={maj.get('n')})"
        )

    pg = report.get("pointing_game") or {}
    hr = pg.get("hit_rate") or {}
    if hr.get("value") is not None:
        lvl = int(round(float(hr.get("ci_level", 0.95)) * 100))
        tol = pg.get("tolerance") or 0.0
        line = (
            f"pointing-game hit rate {_fmt(hr.get('value'))} "
            f"({lvl}% CI {_fmt(hr.get('lo'))}-{_fmt(hr.get('hi'))}, n={hr.get('n')})"
        )
        if float(tol) > 0:
            line += f" at {float(tol):g} px tolerance"
        parts.append(line)

    sc = report.get("shortcut") or {}
    fr = sc.get("flag_rate") or {}
    if fr.get("value") is not None:
        where = (
            "border"
            if str(sc.get("distractor_source") or "").startswith("image border")
            else "distractor"
        )
        parts.append(f"{where}-distracted on {_fmt(fr.get('value'))} of images")

    ov = report.get("overlap") or {}
    at = (ov.get("at_threshold") or {})
    fixed = (at.get("dice") or {}).get("mean")
    if fixed is not None:
        tail = f"Dice {_fmt(fixed)} at fixed threshold {_fmt(at.get('threshold'))}"
        best = ((ov.get("best_over_threshold") or {}).get("dice") or {})
        if best.get("value") is not None:
            tail += (
                f" (best achievable {_fmt(best.get('value'))} at "
                f"{_fmt(best.get('threshold'))} — "
                f"{best.get('threshold_selection', OPTIMISTIC_THRESHOLD_NOTE)})"
            )
        parts.append(tail)

    froc = report.get("froc") or {}
    if froc.get("froc_score") is not None:
        parts.append(f"FROC score {_fmt(froc.get('froc_score'))}")

    return "; ".join(parts) if parts else "no localization metrics available"
