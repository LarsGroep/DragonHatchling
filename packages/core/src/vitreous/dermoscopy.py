"""Dermoscopic concept grounding — name neurons after what is *in the crop*.

Concept grounding answers "what does this neuron cluster look for?" by measuring
how much more strongly a concept fires on images that **have** a human-annotated
attribute than on images that do not, then naming the concept after its most
discriminative attributes.

The grounding is only as honest as its vocabulary. The legacy HAM10000 pipeline
(``hatchvision/explain/attributes.py`` + ``hatchvision/data/skin_lesion.py``)
grounded visual concepts in **patient metadata** — sex, age bucket, body site —
and shipped concept names like ``"location: foot · location: abdomen"``. That is
wrong on the merits and dangerous in a clinical frame:

* a dermatoscopic crop does not contain the body site; nothing in the pixels
  encodes "foot". A name derived from it is an explanation the model never gave;
* body site correlates with diagnosis in HAM10000 (acral nevi on feet, actinic
  keratoses on face/scalp in older patients), so a metadata-grounded namer will
  *reliably* surface confounds dressed up as learned visual features.

This module fixes both halves:

**(a) The correct vocabulary — ISIC 2018 Task 2 (Lesion Attribute Detection).**
Task 2 is derived from the same HAM10000 images and ships **pixel-level binary
masks** for five classical dermoscopic criteria on **2,594 training images**:
``pigment_network``, ``negative_network``, ``streaks``, ``milia_like_cyst``,
``globules``. These are properties of the lesion surface — things a dermatologist
points at through the dermatoscope — so a concept named after one of them is
making a claim that can be checked against the crop.
See :data:`DERMOSCOPIC_ATTRIBUTES` and :func:`load_isic2018_task2_attributes`.

**(b) Metadata becomes a confound *probe*, not a name.** The metadata attributes
are kept (they are informative!) but routed through
:func:`probe_metadata_confound`, which returns a *warning* — "this concept tracks
body-site metadata more strongly than any dermoscopic feature — treat its
explanation with caution". This is lay judgment question #2 of
``docs/UX-VISION.md`` ("Is that the right place to look?" — the shortcut
detector) at the concept tier: the key output is a side-by-side comparison of the
dermoscopic-attribute effect vs. the metadata-attribute effect for the same
concept (:func:`grounding_report`).

**Naming discipline.** A concept gets a human-readable name only when a
*dermoscopic* attribute clears **both** the effect-size bar and the
minimum-support bar; otherwise the concept keeps its ID. A metadata attribute can
never become a name — there is no code path from :data:`METADATA_ATTRIBUTES` to a
label. Effect sizes are reported with their support and with a sample-size-honest
lower bound (:attr:`AttributeEffect.effect_lo`), because a raw activation ratio
over three positive images is noise, and the legacy implementation had no defence
against exactly that (it gated on a raw effect with a small default support).

**Import discipline (M0 rule):** numpy + stdlib only, no torch, no PIL. Mask PNGs
are decoded by a small stdlib ``zlib``/``struct`` reader (:func:`decode_png_gray`);
Pillow is used only as a lazy fallback for non-PNG masks.
"""

from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    # vocabulary + provenance
    "DERMOSCOPIC_ATTRIBUTES",
    "DERMOSCOPIC_DISPLAY_NAMES",
    "METADATA_ATTRIBUTES",
    "METADATA_DISPLAY_NAMES",
    "ISIC2018_TASK2",
    # containers
    "AttributeTable",
    "EffectStats",
    "AttributeEffect",
    "ConceptGrounding",
    "ConfoundReport",
    # png decoding (shared with vitreous.localization)
    "decode_png_gray",
    # loaders
    "load_isic2018_task2_attributes",
    "build_metadata_attributes",
    # measurement
    "attribute_effects",
    "ground_concepts",
    "probe_metadata_confound",
    "grounding_report",
]


# --------------------------------------------------------------------------- #
# (a) The dermoscopic vocabulary — ISIC 2018 Task 2.
# --------------------------------------------------------------------------- #

#: The five lesion attributes annotated by ISIC 2018 Task 2, in the canonical
#: order used for every ``[N, 5]`` attribute matrix this module produces.
DERMOSCOPIC_ATTRIBUTES: Tuple[str, ...] = (
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
)

#: Human-readable labels — what a concept named after an attribute shows in the UI.
DERMOSCOPIC_DISPLAY_NAMES: Dict[str, str] = {
    "pigment_network": "pigment network",
    "negative_network": "negative network",
    "streaks": "streaks",
    "milia_like_cyst": "milia-like cysts",
    "globules": "globules",
}

#: Provenance of the grounding vocabulary. Emitted into every
#: :class:`AttributeTable` so a pack can state where its concept names came from.
ISIC2018_TASK2: Dict[str, Any] = {
    "dataset": "ISIC 2018 Task 2 — Lesion Attribute Detection",
    "derived_from": "HAM10000 (Tschandl, Rosendahl & Kittler 2018)",
    "n_training_images": 2594,
    "n_attributes": len(DERMOSCOPIC_ATTRIBUTES),
    "annotation": "pixel-level binary masks — one PNG per (image, attribute)",
    "mask_filename_pattern": "ISIC_<image_id>_attribute_<attribute>.png",
    "presence_rule": "an attribute is present for an image iff its mask is non-empty",
    "license": "CC BY-NC 4.0",
    "url": "https://challenge.isic-archive.com/landing/2018/",
}

# Filename spellings seen in the wild (plural/singular drift, hyphens) that map
# onto the canonical vocabulary.
_ATTRIBUTE_ALIASES: Dict[str, str] = {
    "pigment_network": "pigment_network",
    "pigment_networks": "pigment_network",
    "pigmentnetwork": "pigment_network",
    "negative_network": "negative_network",
    "negative_networks": "negative_network",
    "negativenetwork": "negative_network",
    "streaks": "streaks",
    "streak": "streaks",
    "milia_like_cyst": "milia_like_cyst",
    "milia_like_cysts": "milia_like_cyst",
    "milialikecyst": "milia_like_cyst",
    "globules": "globules",
    "globule": "globules",
}


# --------------------------------------------------------------------------- #
# (c) The metadata vocabulary — kept, but only ever as a confound probe.
# --------------------------------------------------------------------------- #

_AGE_BUCKETS: Tuple[Tuple[float, float], ...] = (
    (0, 20), (20, 40), (40, 60), (60, 80), (80, 200),
)
_AGE_ATTRS: Tuple[str, ...] = (
    "age: <20", "age: 20-40", "age: 40-60", "age: 60-80", "age: 80+",
)
_SEX_ATTRS: Tuple[str, ...] = ("sex: male", "sex: female")
_LOCATION_ATTRS: Tuple[str, ...] = (
    "location: abdomen", "location: acral", "location: back",
    "location: chest", "location: ear", "location: face",
    "location: foot", "location: genital", "location: hand",
    "location: lower extremity", "location: neck", "location: scalp",
    "location: trunk", "location: upper extremity",
)

#: The HAM10000 patient-metadata attributes (identical vocabulary to the legacy
#: ``hatchvision.data.skin_lesion`` loader). These are **never** concept names —
#: they are the input to :func:`probe_metadata_confound`.
METADATA_ATTRIBUTES: Tuple[str, ...] = _SEX_ATTRS + _AGE_ATTRS + _LOCATION_ATTRS

#: Plain-language rendering of a metadata attribute, used inside warnings only.
METADATA_DISPLAY_NAMES: Dict[str, str] = {
    a: a.split(": ", 1)[1] if ": " in a else a for a in METADATA_ATTRIBUTES
}

# family key (text before the colon) -> plain-language phrase for the warning.
_METADATA_FAMILY_PHRASE: Dict[str, str] = {
    "location": "body-site",
    "age": "patient-age",
    "sex": "patient-sex",
}


def _metadata_family(name: str) -> str:
    return name.split(":", 1)[0].strip().lower() if ":" in name else "metadata"


def _family_phrase(name: str) -> str:
    return _METADATA_FAMILY_PHRASE.get(_metadata_family(name), "patient-metadata")


# --------------------------------------------------------------------------- #
# Containers.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttributeTable:
    """A per-image binary attribute matrix plus the names of its columns.

    ``matrix`` is ``[N, A]`` with ``matrix[i, j] == 1`` when image ``image_ids[i]``
    has attribute ``attribute_names[j]``. ``provenance`` records how the table was
    built (dataset, file counts, what was skipped) so a pack can be audited.
    """

    image_ids: List[str]
    attribute_names: List[str]
    matrix: np.ndarray
    display_names: Dict[str, str] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix)
        if m.ndim != 2:
            raise ValueError(f"attribute matrix must be [N, A], got {m.shape}")
        if m.shape[0] != len(self.image_ids):
            raise ValueError(
                f"matrix has {m.shape[0]} rows but {len(self.image_ids)} image ids"
            )
        if m.shape[1] != len(self.attribute_names):
            raise ValueError(
                f"matrix has {m.shape[1]} columns but "
                f"{len(self.attribute_names)} attribute names"
            )

    @property
    def n_images(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_attributes(self) -> int:
        return int(self.matrix.shape[1])

    def support(self) -> np.ndarray:
        """``[A]`` count of images carrying each attribute."""
        return (np.asarray(self.matrix) > 0).sum(axis=0).astype(np.int64)

    def to_json(self) -> Dict[str, Any]:
        """JSON-serializable payload (matrix as a list of 0/1 rows)."""
        return {
            "image_ids": list(self.image_ids),
            "attribute_names": list(self.attribute_names),
            "display_names": dict(self.display_names),
            "support": [int(x) for x in self.support()],
            "matrix": [[int(v) for v in row] for row in np.asarray(self.matrix) > 0],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class EffectStats:
    """Raw per-(attribute, concept) measurements — the shared low-level readout.

    All matrices are ``[A, C]`` (attributes × concepts); ``support`` /
    ``support_negative`` / ``valid`` are ``[A]``. Attributes failing the support
    bar have ``effect`` and ``effect_lo`` set to ``-inf`` so they can never win a
    ranking.
    """

    effect: np.ndarray
    effect_lo: np.ndarray
    support: np.ndarray
    support_negative: np.ndarray
    valid: np.ndarray
    mean_with: np.ndarray
    mean_without: np.ndarray


@dataclass(frozen=True)
class AttributeEffect:
    """One (concept, attribute) association with its sample size attached."""

    attribute: str
    display_name: str
    effect: float          # Cohen's d — standardized mean activation difference
    effect_lo: float       # sample-size-honest lower bound on d
    support: int           # images WITH the attribute
    support_negative: int  # images WITHOUT it
    mean_with: float
    mean_without: float
    qualifies: bool        # cleared BOTH the support and the effect bar

    def to_json(self) -> Dict[str, Any]:
        return {
            "attribute": self.attribute,
            "display_name": self.display_name,
            "effect": round(self.effect, 6),
            "effect_lo": round(self.effect_lo, 6),
            "support": self.support,
            "support_negative": self.support_negative,
            "mean_with": round(self.mean_with, 6),
            "mean_without": round(self.mean_without, 6),
            "qualifies": self.qualifies,
        }


@dataclass(frozen=True)
class ConceptGrounding:
    """What a concept may honestly be called, and the measurement behind it.

    ``label`` is ``None`` — and ``named`` is ``False`` — whenever no dermoscopic
    attribute cleared both bars. Callers must fall back to ``concept_id`` in that
    case: an unnamed concept is a true statement, an invented name is not.
    """

    concept_index: int
    concept_id: str
    label: Optional[str]
    named: bool
    attributes: List[AttributeEffect]
    reason: str

    @property
    def top_effect(self) -> float:
        return self.attributes[0].effect_lo if self.attributes else float("-inf")

    def display(self) -> str:
        """The string the UI should show: the label when earned, else the ID."""
        return self.label if self.label else self.concept_id

    def to_json(self) -> Dict[str, Any]:
        return {
            "concept_index": self.concept_index,
            "concept_id": self.concept_id,
            "label": self.label,
            "named": self.named,
            "display": self.display(),
            "reason": self.reason,
            "attributes": [a.to_json() for a in self.attributes],
        }


@dataclass(frozen=True)
class ConfoundReport:
    """Shortcut detector for one concept: metadata effect vs. dermoscopic effect.

    ``confound_score`` is ``metadata / (metadata + dermoscopic)`` over the clipped
    non-negative effects — ``0`` when only a dermoscopic attribute explains the
    concept, ``1`` when only metadata does, ``0.5`` at parity. ``flagged`` is the
    boolean the UI turns into a caution badge; ``message`` is its plain-language
    rendering.
    """

    concept_index: int
    concept_id: str
    metadata_attribute: Optional[str]
    metadata_effect: float
    metadata_support: int
    dermoscopic_attribute: Optional[str]
    dermoscopic_effect: float
    dermoscopic_support: int
    margin: float
    confound_score: float
    flagged: bool
    message: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "concept_index": self.concept_index,
            "concept_id": self.concept_id,
            "metadata_attribute": self.metadata_attribute,
            "metadata_effect": round(self.metadata_effect, 6),
            "metadata_support": self.metadata_support,
            "dermoscopic_attribute": self.dermoscopic_attribute,
            "dermoscopic_effect": round(self.dermoscopic_effect, 6),
            "dermoscopic_support": self.dermoscopic_support,
            "margin": round(self.margin, 6),
            "confound_score": round(self.confound_score, 6),
            "flagged": self.flagged,
            "message": self.message,
        }


# --------------------------------------------------------------------------- #
# Stdlib PNG mask reader (no PIL at import time — M0 rule).
# --------------------------------------------------------------------------- #

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
MASK_EXTS: Tuple[str, ...] = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, height: int, stride: int, bpp: int) -> bytes:
    """Reverse the PNG per-scanline filters (spec §9.2)."""
    # Fast path: masks are usually stored with filter 0 on every row.
    if height and all(raw[y * (stride + 1)] == 0 for y in range(height)):
        buf = np.frombuffer(raw, dtype=np.uint8, count=height * (stride + 1))
        return buf.reshape(height, stride + 1)[:, 1:].tobytes()

    out = bytearray(height * stride)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        pos += stride
        if ftype == 0:
            pass
        elif ftype == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(a, prev[i], c)) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter type {ftype}")
        out[y * stride : (y + 1) * stride] = line
        prev = line
    return bytes(out)


def _unpack_samples(data: bytes, width: int, height: int, channels: int, depth: int) -> np.ndarray:
    """Unfiltered scanline bytes → ``[H, W, channels]`` uint16 samples (0..255 scale)."""
    if depth == 8:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint16)
        return arr.reshape(height, width * channels)[:, : width * channels].reshape(
            height, width, channels
        )
    if depth == 16:
        arr = np.frombuffer(data, dtype=">u2").astype(np.uint32)
        arr = (arr // 257).astype(np.uint16)  # 0..65535 -> 0..255
        return arr.reshape(height, width * channels).reshape(height, width, channels)
    if depth in (1, 2, 4):
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8).reshape(height, -1), axis=1)
        per_row = width * channels
        vals = np.zeros((height, per_row), dtype=np.uint16)
        for k in range(depth):
            vals = (vals << 1) | bits[:, k::depth][:, :per_row].astype(np.uint16)
        scale = 255 // ((1 << depth) - 1)
        return (vals * scale).reshape(height, width, channels)
    raise ValueError(f"unsupported PNG bit depth {depth}")


def decode_png_gray(data: bytes) -> np.ndarray:
    """Decode a non-interlaced PNG to a ``[H, W]`` uint16 intensity map (0..255).

    Public because :mod:`vitreous.localization` reads the same ISIC mask files
    for its overlap metrics. Two stdlib PNG readers in one package would be one
    too many, so this is the single decoder — keep it that way.

    Supports colour types 0/2/3/4/6 and bit depths 1/2/4/8/16 — everything the
    ISIC mask releases and the common encoders (OpenCV, ImageMagick, PIL) emit.
    Alpha, when present, zeroes fully transparent pixels; palette indices are
    resolved through PLTE. Adam7-interlaced PNGs raise :class:`ValueError`.
    """
    if not data.startswith(_PNG_MAGIC):
        raise ValueError("not a PNG file (bad magic)")
    pos = len(_PNG_MAGIC)
    width = height = depth = colour = interlace = -1
    palette: Optional[np.ndarray] = None
    idat = bytearray()
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length  # length + tag + data + crc
        if tag == b"IHDR":
            width, height, depth, colour, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif tag == b"PLTE":
            palette = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, 3)
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    if width < 0:
        raise ValueError("PNG has no IHDR chunk")
    if interlace:
        raise ValueError("interlaced (Adam7) PNGs are not supported")
    if colour not in _CHANNELS:
        raise ValueError(f"unsupported PNG colour type {colour}")

    channels = _CHANNELS[colour]
    stride = (width * channels * depth + 7) // 8
    raw = zlib.decompress(bytes(idat))
    expected = height * (stride + 1)
    if len(raw) < expected:
        raise ValueError("truncated PNG image data")
    px = _unpack_samples(_unfilter(raw[:expected], height, stride, max(1, channels * depth // 8)),
                         width, height, channels, depth)

    if colour == 3:  # palette
        idx = px[:, :, 0].astype(np.int64)
        if palette is None:
            return idx.astype(np.uint16)
        idx = np.clip(idx, 0, len(palette) - 1)
        return palette[idx].max(axis=2).astype(np.uint16)
    if colour == 0:
        return px[:, :, 0]
    if colour == 4:  # gray + alpha
        return (px[:, :, 0] * (px[:, :, 1] > 0)).astype(np.uint16)
    if colour == 2:  # RGB
        return px.max(axis=2)
    return (px[:, :, :3].max(axis=2) * (px[:, :, 3] > 0)).astype(np.uint16)  # RGBA


def _mask_positive_pixels(path: Path, threshold: int = 0) -> int:
    """Count pixels above ``threshold`` in a binary mask file.

    PNGs are decoded with the stdlib reader above; anything else falls back to
    Pillow, imported lazily so this module stays PIL-free at import time.
    """
    if path.suffix.lower() == ".png":
        try:
            arr = decode_png_gray(path.read_bytes())
            return int((arr > threshold).sum())
        except ValueError:
            pass  # fall through to Pillow for exotic encodings
    try:
        from PIL import Image  # lazy, optional
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"cannot read mask {path}: not a plain PNG and Pillow is not installed. "
            "Install the [ml] extra (or Pillow) to read non-PNG attribute masks."
        ) from exc
    with Image.open(path) as im:  # pragma: no cover - requires Pillow
        return int((np.asarray(im.convert("L")) > threshold).sum())


# --------------------------------------------------------------------------- #
# ISIC 2018 Task 2 loader.
# --------------------------------------------------------------------------- #

# ``ISIC_0000000_attribute_pigment_network.png`` — image id, then the literal
# ``_attribute_`` separator, then the criterion. Matched case-insensitively; a
# hyphen is accepted in place of the separator underscore.
_MASK_STEM_RE = re.compile(r"^(?P<image>.+?)_attribute[_-](?P<attr>.+)$", re.IGNORECASE)


def _normalise_attribute(raw: str) -> Optional[str]:
    key = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    key = re.sub(r"_+", "_", key).strip("_")
    return _ATTRIBUTE_ALIASES.get(key)


def load_isic2018_task2_attributes(
    root: Union[str, Path],
    *,
    attributes: Sequence[str] = DERMOSCOPIC_ATTRIBUTES,
    threshold: int = 0,
    image_ids: Optional[Sequence[str]] = None,
    max_images: Optional[int] = None,
) -> AttributeTable:
    """Read an ISIC 2018 Task 2 ground-truth tree into an ``[N, 5]`` binary matrix.

    On-disk layout (the official release; only the mask filenames matter, they may
    sit at any depth under ``root``)::

        <root>/
            ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg          (images)
            ISIC2018_Task2_Training_GroundTruth_v3/
                ISIC_0000000_attribute_globules.png
                ISIC_0000000_attribute_milia_like_cyst.png
                ISIC_0000000_attribute_negative_network.png
                ISIC_0000000_attribute_pigment_network.png
                ISIC_0000000_attribute_streaks.png
                ...

    **Presence rule:** an attribute is present for an image iff its mask contains
    at least one pixel above ``threshold``. An all-black mask (the common encoding
    for "criterion absent") is therefore *absent*, not missing.

    The scan is deliberately tolerant, in the style of
    :class:`vitreous.data.HAM10000Adapter`: filenames are matched
    case-insensitively, plural/hyphenated spellings are normalised, masks are
    found recursively, images whose criteria are only partially annotated get
    zeros for the rest, and images with no mask files at all are skipped (and
    counted in ``provenance``).

    Parameters
    ----------
    root:
        Directory containing the Task 2 masks (at any depth).
    attributes:
        Column vocabulary — defaults to :data:`DERMOSCOPIC_ATTRIBUTES`.
    threshold:
        Pixel intensity (0..255) above which a mask pixel counts as annotated.
    image_ids:
        Restrict (and order) the rows to these image ids. Ids with no mask files
        are skipped rather than fabricated as all-zero rows.
    max_images:
        Keep only the first ``max_images`` rows (after ordering) — for smoke runs.

    Returns
    -------
    AttributeTable
        ``matrix`` is ``[N, len(attributes)]`` uint8.

    Raises
    ------
    FileNotFoundError
        If ``root`` is not a directory, or if it contains no file matching the
        ``*_attribute_*`` mask convention (the error names the expected pattern).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"ISIC 2018 Task 2 root not found: {root}")

    wanted = list(attributes)
    unknown_cols = [a for a in wanted if a not in DERMOSCOPIC_ATTRIBUTES]
    if unknown_cols:
        raise ValueError(
            f"unknown dermoscopic attribute(s) {unknown_cols}; "
            f"the ISIC 2018 Task 2 vocabulary is {list(DERMOSCOPIC_ATTRIBUTES)}"
        )

    # image_id -> attribute -> mask path (first match wins, sorted for determinism)
    found: Dict[str, Dict[str, Path]] = {}
    n_mask_files = 0
    unknown_attrs: set = set()
    for p in sorted(root_path.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in MASK_EXTS:
            continue
        m = _MASK_STEM_RE.match(p.stem)
        if not m:
            continue
        attr = _normalise_attribute(m.group("attr"))
        if attr is None:
            unknown_attrs.add(m.group("attr"))
            continue
        n_mask_files += 1
        if attr in wanted:
            found.setdefault(m.group("image"), {}).setdefault(attr, p)

    if not found:
        raise FileNotFoundError(
            f"no ISIC 2018 Task 2 attribute masks found under {root}. Expected files "
            f"named '{ISIC2018_TASK2['mask_filename_pattern']}' (e.g. "
            f"'ISIC_0000000_attribute_pigment_network.png') anywhere below that "
            f"directory, for the criteria {list(DERMOSCOPIC_ATTRIBUTES)}. "
            f"Scanned {n_mask_files} mask-shaped file(s)"
            + (f"; unrecognised criteria seen: {sorted(unknown_attrs)}" if unknown_attrs else "")
            + "."
        )

    if image_ids is None:
        ordered = sorted(found)
        requested_missing: List[str] = []
    else:
        ordered = [i for i in image_ids if i in found]
        requested_missing = [i for i in image_ids if i not in found]
    if max_images is not None:
        ordered = ordered[: max_images]

    n_missing_masks = 0
    n_empty_masks = 0
    per_attr_files = {a: 0 for a in wanted}
    rows = np.zeros((len(ordered), len(wanted)), dtype=np.uint8)
    for i, img in enumerate(ordered):
        masks = found[img]
        for j, attr in enumerate(wanted):
            path = masks.get(attr)
            if path is None:
                n_missing_masks += 1
                continue
            per_attr_files[attr] += 1
            positive = _mask_positive_pixels(path, threshold)
            if positive:
                rows[i, j] = 1
            else:
                n_empty_masks += 1

    provenance = {
        **ISIC2018_TASK2,
        "root": str(root_path),
        "threshold": int(threshold),
        "n_images": len(ordered),
        "n_mask_files": n_mask_files,
        "n_mask_files_per_attribute": per_attr_files,
        "n_missing_masks": n_missing_masks,
        "n_empty_masks": n_empty_masks,
        "n_images_skipped_no_masks": len(requested_missing),
        "images_skipped_no_masks": requested_missing[:20],
        "unrecognised_criteria": sorted(unknown_attrs),
    }
    return AttributeTable(
        image_ids=list(ordered),
        attribute_names=list(wanted),
        matrix=rows,
        display_names={a: DERMOSCOPIC_DISPLAY_NAMES[a] for a in wanted},
        provenance=provenance,
    )


def build_metadata_attributes(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_id_key: str = "image_id",
    sex_key: str = "sex",
    age_key: str = "age",
    localization_key: str = "localization",
) -> AttributeTable:
    """HAM10000 metadata rows → the ``[N, 21]`` **confound** attribute matrix.

    Accepts the rows of ``HAM10000_metadata.csv`` (``vitreous.data._read_csv_rows``
    output). Unknown/blank/unparseable values simply produce zeros. The result is
    only ever fed to :func:`probe_metadata_confound` — nothing in this module can
    turn one of these columns into a concept name.
    """
    ids: List[str] = []
    mat = np.zeros((len(rows), len(METADATA_ATTRIBUTES)), dtype=np.uint8)
    index = {a: i for i, a in enumerate(METADATA_ATTRIBUTES)}
    for i, row in enumerate(rows):
        ids.append(str(row.get(image_id_key, f"row_{i}")))
        sex = str(row.get(sex_key, "") or "").strip().lower()
        if f"sex: {sex}" in index:
            mat[i, index[f"sex: {sex}"]] = 1
        loc = str(row.get(localization_key, "") or "").strip().lower()
        if f"location: {loc}" in index:
            mat[i, index[f"location: {loc}"]] = 1
        raw_age = row.get(age_key, "")
        try:
            age = float(str(raw_age).strip())
        except (TypeError, ValueError):
            age = float("nan")
        if age == age:  # not NaN
            for b, (lo, hi) in enumerate(_AGE_BUCKETS):
                if lo <= age < hi:
                    mat[i, index[_AGE_ATTRS[b]]] = 1
                    break
            else:
                mat[i, index[_AGE_ATTRS[-1]]] = 1
    return AttributeTable(
        image_ids=ids,
        attribute_names=list(METADATA_ATTRIBUTES),
        matrix=mat,
        display_names=dict(METADATA_DISPLAY_NAMES),
        provenance={
            "dataset": "HAM10000 metadata (patient sex / age bucket / body site)",
            "role": "confound probe only — never a concept name",
            "n_images": len(ids),
        },
    )


# --------------------------------------------------------------------------- #
# (b) Measurement: effect sizes, grounding, confound probe.
# --------------------------------------------------------------------------- #


def _as_matrix_and_names(
    attributes: Union[AttributeTable, np.ndarray, Sequence[Sequence[int]]],
    names: Optional[Sequence[str]],
    display: Optional[Mapping[str, str]],
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    if isinstance(attributes, AttributeTable):
        mat = np.asarray(attributes.matrix)
        out_names = list(names) if names is not None else list(attributes.attribute_names)
        out_display = dict(display) if display is not None else dict(attributes.display_names)
    else:
        mat = np.asarray(attributes)
        if names is None:
            raise ValueError("attribute_names is required when passing a raw matrix")
        out_names = list(names)
        out_display = dict(display or {})
    if mat.ndim != 2:
        raise ValueError(f"attribute matrix must be [N, A], got {mat.shape}")
    if mat.shape[1] != len(out_names):
        raise ValueError(
            f"attribute matrix has {mat.shape[1]} columns but "
            f"{len(out_names)} attribute names"
        )
    # Canonical display names are filled in for any known attribute the caller
    # did not override, so a raw matrix + the standard vocabulary still renders
    # "pigment network" rather than the on-disk key.
    for name in out_names:
        if name not in out_display:
            out_display[name] = DERMOSCOPIC_DISPLAY_NAMES.get(
                name, METADATA_DISPLAY_NAMES.get(name, name)
            )
    return mat, out_names, out_display


def attribute_effects(
    activations: np.ndarray,
    attribute_matrix: Union[AttributeTable, np.ndarray],
    *,
    min_support: int = 5,
    z: float = 1.96,
) -> EffectStats:
    """Per-(attribute, concept) standardized effect sizes with honest error bars.

    For every attribute ``a`` and concept ``c``:

    ``effect[a, c] = (mean activation on images WITH a − mean WITHOUT a) / s_pooled``

    i.e. Cohen's *d* against the pooled within-group spread. ``effect_lo`` is the
    lower end of the ``z``-sigma interval on *d*, using the standard large-sample
    error ``sqrt(1/n₊ + 1/n₋ + d²/(2(n₊+n₋)))``. That second number is the point
    of this function: *d* alone is scale-free but not evidence — a *d* of 3
    measured on three positive images has an error bar wider than itself, and
    ``effect_lo`` collapses accordingly. The legacy implementation reported only
    the raw ratio, so a three-image accident could name a neuron.

    Attributes with fewer than ``min_support`` positive **or** negative images are
    marked invalid and get ``-inf`` effects, so they cannot win any ranking.
    """
    acts = np.asarray(activations, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"activations must be [N, C], got {acts.shape}")
    mat = (
        np.asarray(attribute_matrix.matrix)
        if isinstance(attribute_matrix, AttributeTable)
        else np.asarray(attribute_matrix)
    )
    if mat.ndim != 2:
        raise ValueError(f"attribute matrix must be [N, A], got {mat.shape}")
    if mat.shape[0] != acts.shape[0]:
        raise ValueError(
            f"attribute matrix rows ({mat.shape[0]}) must match "
            f"activation rows ({acts.shape[0]})"
        )

    n = acts.shape[0]
    pos = (mat > 0).astype(np.float64)          # [N, A]
    neg = 1.0 - pos
    n_pos = pos.sum(axis=0)                     # [A]
    n_neg = neg.sum(axis=0)
    d_pos = np.maximum(n_pos, 1.0)[:, None]
    d_neg = np.maximum(n_neg, 1.0)[:, None]

    mean_with = (pos.T @ acts) / d_pos           # [A, C]
    mean_without = (neg.T @ acts) / d_neg
    sq = acts * acts
    var_with = np.maximum((pos.T @ sq) / d_pos - mean_with**2, 0.0)
    var_without = np.maximum((neg.T @ sq) / d_neg - mean_without**2, 0.0)

    denom = max(n - 2, 1)
    pooled = np.sqrt((n_pos[:, None] * var_with + n_neg[:, None] * var_without) / denom)
    overall = acts.std(axis=0)[None, :]          # fallback when a group is constant
    spread = np.where(pooled > 1e-12, pooled, np.where(overall > 1e-12, overall, np.inf))

    effect = (mean_with - mean_without) / spread
    se = np.sqrt(
        1.0 / d_pos + 1.0 / d_neg + effect**2 / (2.0 * np.maximum(n, 2))
    )
    effect_lo = effect - z * se

    valid = (n_pos >= min_support) & (n_neg >= min_support)
    effect = np.where(valid[:, None], effect, -np.inf)
    effect_lo = np.where(valid[:, None], effect_lo, -np.inf)

    return EffectStats(
        effect=effect,
        effect_lo=effect_lo,
        support=n_pos.astype(np.int64),
        support_negative=n_neg.astype(np.int64),
        valid=valid,
        mean_with=mean_with,
        mean_without=mean_without,
    )


def _concept_ids(concept_ids: Optional[Sequence[str]], n: int) -> List[str]:
    if concept_ids is None:
        return [f"concept_{i}" for i in range(n)]
    if len(concept_ids) != n:
        raise ValueError(f"concept_ids has {len(concept_ids)} entries but {n} concepts")
    return [str(c) for c in concept_ids]


def _ranked_effects(
    stats: EffectStats,
    ci: int,
    names: Sequence[str],
    display: Mapping[str, str],
    *,
    top_k: int,
    min_effect: float,
    min_support: int,
) -> List[AttributeEffect]:
    """Attributes for concept ``ci``, best first. Deterministic ties by index."""
    order = sorted(
        (a for a in range(len(names)) if bool(stats.valid[a])),
        key=lambda a: (-float(stats.effect_lo[a, ci]), a),
    )
    out: List[AttributeEffect] = []
    for a in order[: max(top_k, 0)]:
        eff = float(stats.effect[a, ci])
        lo = float(stats.effect_lo[a, ci])
        out.append(
            AttributeEffect(
                attribute=names[a],
                display_name=display.get(names[a], names[a]),
                effect=eff,
                effect_lo=lo,
                support=int(stats.support[a]),
                support_negative=int(stats.support_negative[a]),
                mean_with=float(stats.mean_with[a, ci]),
                mean_without=float(stats.mean_without[a, ci]),
                qualifies=bool(int(stats.support[a]) >= min_support and lo >= min_effect),
            )
        )
    return out


def ground_concepts(
    activations: np.ndarray,
    attributes: Union[AttributeTable, np.ndarray],
    attribute_names: Optional[Sequence[str]] = None,
    *,
    concept_ids: Optional[Sequence[str]] = None,
    display_names: Optional[Mapping[str, str]] = None,
    top_k: int = 4,
    min_support: int = 5,
    min_effect: float = 0.25,
    label_parts: int = 2,
    z: float = 1.96,
) -> List[ConceptGrounding]:
    """Name concepts after the dermoscopic attributes they actually discriminate.

    This is the mechanism of the legacy ``hatchvision.explain.attributes``
    grounder — contrast a concept's activation on images with vs. without an
    attribute — made numpy-only, generic over any ``[N, A]`` binary attribute
    matrix, and gated so it cannot invent names.

    A concept is named iff its best attribute clears **both** bars:

    * ``support >= min_support`` — enough positive (and negative) images to measure;
    * ``effect_lo >= min_effect`` — the *sample-size-honest* effect (see
      :func:`attribute_effects`) is above threshold, not merely the point estimate.

    Otherwise ``label`` is ``None`` and the caller must keep showing the concept
    ID. ``reason`` records which bar failed, so the UI can say *why* a concept is
    unnamed instead of pretending it has no story.

    Parameters
    ----------
    activations:
        ``[N, C]`` concept activations (rows aligned with the attribute matrix).
    attributes:
        An :class:`AttributeTable` (names/display come from it) or a raw ``[N, A]``
        binary matrix, in which case ``attribute_names`` is required.
    top_k:
        How many ranked attributes to report per concept (all measured, whether or
        not they qualify — the evidence is shown even when it is too weak to name).
    label_parts:
        How many qualifying attributes are joined (with ``" · "``) into the label.

    Returns
    -------
    list of ConceptGrounding
        One entry per concept, in concept order.
    """
    acts = np.asarray(activations, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"activations must be [N, C], got {acts.shape}")
    mat, names, display = _as_matrix_and_names(attributes, attribute_names, display_names)
    if mat.shape[0] != acts.shape[0]:
        raise ValueError(
            f"attribute matrix rows ({mat.shape[0]}) must match "
            f"activation rows ({acts.shape[0]})"
        )
    ids = _concept_ids(concept_ids, acts.shape[1])
    stats = attribute_effects(acts, mat, min_support=min_support, z=z)

    groundings: List[ConceptGrounding] = []
    for ci in range(acts.shape[1]):
        ranked = _ranked_effects(
            stats, ci, names, display,
            top_k=top_k, min_effect=min_effect, min_support=min_support,
        )
        qualifying = [a for a in ranked if a.qualifies]
        if qualifying:
            label = " · ".join(a.display_name for a in qualifying[: max(label_parts, 1)])
            reason = (
                f"named: {qualifying[0].attribute} effect_lo="
                f"{qualifying[0].effect_lo:.3f} >= {min_effect} on "
                f"{qualifying[0].support} positive image(s)"
            )
            named = True
        else:
            label = None
            named = False
            if not stats.valid.any():
                reason = (
                    f"unnamed: no attribute has >= {min_support} positive and "
                    f"negative images — nothing is measurable at this sample size"
                )
            elif ranked:
                best = ranked[0]
                reason = (
                    f"unnamed: strongest attribute ({best.attribute}) has "
                    f"effect_lo={best.effect_lo:.3f} < {min_effect} on "
                    f"{best.support} positive image(s)"
                )
            else:
                reason = f"unnamed: no attribute cleared min_support={min_support}"
        groundings.append(
            ConceptGrounding(
                concept_index=ci,
                concept_id=ids[ci],
                label=label,
                named=named,
                attributes=ranked,
                reason=reason,
            )
        )
    return groundings


def probe_metadata_confound(
    activations: np.ndarray,
    metadata: Union[AttributeTable, np.ndarray],
    metadata_names: Optional[Sequence[str]] = None,
    *,
    dermoscopic: Optional[Union[AttributeTable, np.ndarray]] = None,
    dermoscopic_names: Optional[Sequence[str]] = None,
    concept_ids: Optional[Sequence[str]] = None,
    min_support: int = 5,
    min_effect: float = 0.25,
    margin: float = 0.1,
    z: float = 1.96,
) -> List[ConfoundReport]:
    """Shortcut detector: does a concept track patient metadata or the lesion?

    Patient metadata (sex, age bucket, body site) is measured with exactly the
    same machinery as the dermoscopic attributes — and then deliberately *not*
    used as a name. What comes back is the comparison the UI needs for lay
    judgment question #2 of ``docs/UX-VISION.md`` ("Is that the right place to
    look?"): the concept's strongest metadata effect next to its strongest
    dermoscopic effect, plus a flag and a plain-language sentence.

    A concept is flagged when its metadata effect clears ``min_effect`` **and**
    beats its best dermoscopic effect by more than ``margin``. A dermatoscopic
    crop cannot contain the body site, so such a concept is tracking a property of
    the *cohort*, not of the lesion — the classic HAM10000 confound (acral nevi on
    feet; actinic keratoses on face/scalp in older patients).

    Passing ``dermoscopic`` is optional but strongly recommended: without it the
    comparison has no counterweight and every metadata association looks decisive.
    """
    acts = np.asarray(activations, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"activations must be [N, C], got {acts.shape}")
    meta_mat, meta_names, meta_display = _as_matrix_and_names(
        metadata, metadata_names, None
    )
    if meta_mat.shape[0] != acts.shape[0]:
        raise ValueError(
            f"metadata rows ({meta_mat.shape[0]}) must match "
            f"activation rows ({acts.shape[0]})"
        )
    ids = _concept_ids(concept_ids, acts.shape[1])
    meta_stats = attribute_effects(acts, meta_mat, min_support=min_support, z=z)

    derm_stats = None
    derm_names: List[str] = []
    derm_display: Dict[str, str] = {}
    if dermoscopic is not None:
        derm_mat, derm_names, derm_display = _as_matrix_and_names(
            dermoscopic, dermoscopic_names, None
        )
        if derm_mat.shape[0] != acts.shape[0]:
            raise ValueError(
                f"dermoscopic rows ({derm_mat.shape[0]}) must match "
                f"activation rows ({acts.shape[0]})"
            )
        derm_stats = attribute_effects(acts, derm_mat, min_support=min_support, z=z)

    reports: List[ConfoundReport] = []
    for ci in range(acts.shape[1]):
        m_best = _ranked_effects(
            meta_stats, ci, meta_names, meta_display,
            top_k=1, min_effect=min_effect, min_support=min_support,
        )
        d_best = (
            _ranked_effects(
                derm_stats, ci, derm_names, derm_display,
                top_k=1, min_effect=min_effect, min_support=min_support,
            )
            if derm_stats is not None
            else []
        )
        m_eff = m_best[0].effect_lo if m_best else float("-inf")
        d_eff = d_best[0].effect_lo if d_best else float("-inf")
        m_pos = max(m_eff, 0.0) if np.isfinite(m_eff) else 0.0
        d_pos = max(d_eff, 0.0) if np.isfinite(d_eff) else 0.0
        total = m_pos + d_pos
        score = float(m_pos / total) if total > 0 else 0.0
        finite_m = m_eff if np.isfinite(m_eff) else 0.0
        finite_d = d_eff if np.isfinite(d_eff) else 0.0
        margin_val = float(finite_m - finite_d)
        flagged = bool(m_best and m_eff >= min_effect and margin_val > margin)

        if flagged:
            attr = m_best[0].attribute
            message = (
                f"this concept tracks {_family_phrase(attr)} metadata "
                f"({attr}) more strongly than any dermoscopic feature — "
                f"treat its explanation with caution"
            )
        elif d_best and d_best[0].qualifies:
            message = (
                f"this concept's strongest association is a dermoscopic feature "
                f"({d_best[0].display_name}), not patient metadata"
            )
        elif m_best and m_eff >= min_effect:
            message = (
                f"this concept is associated with both {_family_phrase(m_best[0].attribute)} "
                f"metadata and lesion appearance; the metadata link is not dominant"
            )
        else:
            message = (
                "no attribute — dermoscopic or metadata — is supported strongly "
                "enough to judge this concept"
            )

        reports.append(
            ConfoundReport(
                concept_index=ci,
                concept_id=ids[ci],
                metadata_attribute=m_best[0].attribute if m_best else None,
                metadata_effect=float(finite_m),
                metadata_support=int(m_best[0].support) if m_best else 0,
                dermoscopic_attribute=d_best[0].attribute if d_best else None,
                dermoscopic_effect=float(finite_d),
                dermoscopic_support=int(d_best[0].support) if d_best else 0,
                margin=margin_val,
                confound_score=score,
                flagged=flagged,
                message=message,
            )
        )
    return reports


def grounding_report(
    activations: np.ndarray,
    dermoscopic: Union[AttributeTable, np.ndarray],
    metadata: Optional[Union[AttributeTable, np.ndarray]] = None,
    *,
    dermoscopic_names: Optional[Sequence[str]] = None,
    metadata_names: Optional[Sequence[str]] = None,
    concept_ids: Optional[Sequence[str]] = None,
    top_k: int = 4,
    min_support: int = 5,
    min_effect: float = 0.25,
    label_parts: int = 2,
    margin: float = 0.1,
    z: float = 1.96,
) -> List[Dict[str, Any]]:
    """One JSON-serializable record per concept: its name *and* its caution flag.

    Combines :func:`ground_concepts` (dermoscopic vocabulary → name) with
    :func:`probe_metadata_confound` (metadata vocabulary → warning) into the shape
    the verdict panel consumes. ``display`` is the label when it was earned and the
    concept ID otherwise — the UI should render it verbatim.
    """
    groundings = ground_concepts(
        activations,
        dermoscopic,
        dermoscopic_names,
        concept_ids=concept_ids,
        top_k=top_k,
        min_support=min_support,
        min_effect=min_effect,
        label_parts=label_parts,
        z=z,
    )
    if metadata is None:
        return [
            {**g.to_json(), "confound": None} for g in groundings
        ]
    probes = probe_metadata_confound(
        activations,
        metadata,
        metadata_names,
        dermoscopic=dermoscopic,
        dermoscopic_names=dermoscopic_names,
        concept_ids=concept_ids,
        min_support=min_support,
        min_effect=min_effect,
        margin=margin,
        z=z,
    )
    return [
        {**g.to_json(), "confound": p.to_json()} for g, p in zip(groundings, probes)
    ]
