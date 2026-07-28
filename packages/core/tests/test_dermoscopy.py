"""Dermoscopic concept grounding (vitreous.dermoscopy) — numpy-only.

Covers the ISIC 2018 Task 2 vocabulary + mask loader, the grounding mechanism
(planted-signal recovery, the two naming bars, no spurious labels), and the
metadata **confound probe** that replaces the legacy metadata-as-a-name
grounding. No torch, no Pillow: mask fixtures are written with stdlib
``zlib``/``struct``, mirroring the M0 discipline of test_malignancy.py.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from vitreous.dermoscopy import (
    DERMOSCOPIC_ATTRIBUTES,
    DERMOSCOPIC_DISPLAY_NAMES,
    ISIC2018_TASK2,
    METADATA_ATTRIBUTES,
    AttributeTable,
    attribute_effects,
    build_metadata_attributes,
    ground_concepts,
    grounding_report,
    load_isic2018_task2_attributes,
    probe_metadata_confound,
)

# --------------------------------------------------------------------------- #
# stdlib PNG fixture writer (no Pillow) — see apps/web/scripts/gen-lens-demo.py
# --------------------------------------------------------------------------- #


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, rows, *, filter_type: int = 0, rgb: bool = False) -> Path:
    """Write an 8-bit PNG mask. ``rows`` is a list of lists of 0..255 intensities."""
    height, width = len(rows), len(rows[0])
    channels = 3 if rgb else 1
    raw = bytearray()
    prev = [0] * (width * channels)
    for r in rows:
        line = [v for v in r for _ in range(channels)] if rgb else list(r)
        raw.append(filter_type)
        if filter_type == 0:
            raw.extend(line)
        elif filter_type == 2:  # Up
            raw.extend((line[i] - prev[i]) & 0xFF for i in range(len(line)))
        elif filter_type == 4:  # Paeth (bpp = channels)
            for i in range(len(line)):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                raw.append((line[i] - pred) & 0xFF)
        else:  # pragma: no cover - fixtures only use 0/2/4
            raise ValueError(filter_type)
        prev = line
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2 if rgb else 0, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    return path


def _mask(filled: bool, size: int = 6):
    """A blank mask, or one with a small annotated blob."""
    rows = [[0] * size for _ in range(size)]
    if filled:
        for y in range(1, 3):
            for x in range(1, 3):
                rows[y][x] = 255
    return rows


def _make_task2_tree(tmp_path: Path, presence, *, gt_dir: str = "ISIC2018_Task2_Training_GroundTruth_v3"):
    """Write ``{image_id: {attribute: bool}}`` as an ISIC 2018 Task 2 mask tree."""
    root = tmp_path / "isic2018"
    gt = root / gt_dir
    for image_id, attrs in presence.items():
        for attr, filled in attrs.items():
            _write_png(gt / f"{image_id}_attribute_{attr}.png", _mask(filled))
    # The official release ships the shared Task1-2 input images next to it.
    for image_id in presence:
        _write_png(root / "ISIC2018_Task1-2_Training_Input" / f"{image_id}.png", _mask(True))
    return root


def _full(**overrides):
    row = {a: False for a in DERMOSCOPIC_ATTRIBUTES}
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- #
# vocabulary + provenance
# --------------------------------------------------------------------------- #


def test_vocabulary_is_isic2018_task2():
    assert DERMOSCOPIC_ATTRIBUTES == (
        "pigment_network",
        "negative_network",
        "streaks",
        "milia_like_cyst",
        "globules",
    )
    assert set(DERMOSCOPIC_DISPLAY_NAMES) == set(DERMOSCOPIC_ATTRIBUTES)
    # Provenance is stated, not implied.
    assert ISIC2018_TASK2["n_training_images"] == 2594
    assert "pixel-level" in ISIC2018_TASK2["annotation"]
    assert "ISIC 2018 Task 2" in ISIC2018_TASK2["dataset"]
    assert "HAM10000" in ISIC2018_TASK2["derived_from"]
    # The metadata vocabulary is kept — but it is a separate list.
    assert "location: foot" in METADATA_ATTRIBUTES
    assert not set(METADATA_ATTRIBUTES) & set(DERMOSCOPIC_ATTRIBUTES)


# --------------------------------------------------------------------------- #
# grounding
# --------------------------------------------------------------------------- #


def _planted(seed: int = 0, n: int = 200):
    """Attributes + activations where concept 0 genuinely tracks pigment_network.

    concept 0 = pigment network signal;  concept 1 = pure noise (no association).
    """
    rng = np.random.default_rng(seed)
    attrs = (rng.random((n, len(DERMOSCOPIC_ATTRIBUTES))) < 0.4).astype(np.uint8)
    acts = rng.standard_normal((n, 2)) * 0.5
    acts[:, 0] += 2.0 * attrs[:, 0]
    return attrs, acts


def test_planted_signal_is_recovered_and_ranked_first():
    attrs, acts = _planted()
    grounded = ground_concepts(acts, attrs, DERMOSCOPIC_ATTRIBUTES,
                               display_names=DERMOSCOPIC_DISPLAY_NAMES)
    assert len(grounded) == 2
    g = grounded[0]
    assert g.named is True
    assert g.attributes[0].attribute == "pigment_network"
    assert g.attributes[0].effect > 1.0
    assert g.attributes[0].support >= 5
    assert g.label == DERMOSCOPIC_DISPLAY_NAMES["pigment_network"]
    assert g.display() == "pigment network"
    assert "pigment_network" in g.reason


def test_unassociated_concept_stays_unnamed():
    attrs, acts = _planted()
    grounded = ground_concepts(acts, attrs, DERMOSCOPIC_ATTRIBUTES)
    noise = grounded[1]
    assert noise.named is False
    assert noise.label is None
    # It falls back to its ID, and says why it has no name.
    assert noise.display() == "concept_1"
    assert noise.reason.startswith("unnamed:")
    # The evidence is still reported — just too weak to be a name.
    assert all(not a.qualifies for a in noise.attributes)


def test_min_support_suppresses_large_effect_tiny_support_attribute():
    """A huge effect measured on 2 images must not become a name."""
    rng = np.random.default_rng(3)
    n = 120
    attrs = np.zeros((n, 5), dtype=np.uint8)
    attrs[:2, 0] = 1                      # tiny support: 2 positive images
    attrs[: n // 2, 1] = 1                # well-supported but uninformative
    acts = rng.standard_normal((n, 1)) * 0.2
    acts[:2, 0] += 12.0                   # enormous, but on 2 images only

    strict = ground_concepts(acts, attrs, DERMOSCOPIC_ATTRIBUTES, min_support=5)[0]
    assert strict.named is False
    assert "pigment_network" not in [a.attribute for a in strict.attributes]

    # Lower the bar and the very same attribute reappears — the bar is what
    # suppressed it, not an accident of the data.
    loose = ground_concepts(acts, attrs, DERMOSCOPIC_ATTRIBUTES, min_support=2)[0]
    assert loose.attributes[0].attribute == "pigment_network"
    assert loose.attributes[0].support == 2


def test_effect_lo_is_honest_about_sample_size():
    """Same point estimate, less data → a smaller sample-size-honest effect."""
    rng = np.random.default_rng(7)
    big_n, small_n = 400, 40
    for n in (big_n, small_n):
        attrs = np.zeros((n, 1), dtype=np.uint8)
        attrs[: n // 2, 0] = 1
        acts = rng.standard_normal((n, 1)) * 0.0
        acts[: n // 2, 0] = 1.0
        acts[n // 2 :, 0] = 0.0
        stats = attribute_effects(acts, attrs, min_support=5)
        if n == big_n:
            big = stats
        else:
            small = stats
    assert big.effect[0, 0] == pytest.approx(small.effect[0, 0], rel=1e-6)
    assert big.effect_lo[0, 0] > small.effect_lo[0, 0]
    assert big.effect_lo[0, 0] < big.effect[0, 0]  # never optimistic


def test_shape_mismatches_raise():
    acts = np.zeros((10, 2))
    with pytest.raises(ValueError):
        ground_concepts(acts, np.zeros((7, 5), dtype=np.uint8), DERMOSCOPIC_ATTRIBUTES)
    with pytest.raises(ValueError):
        ground_concepts(acts, np.zeros((10, 4), dtype=np.uint8), DERMOSCOPIC_ATTRIBUTES)
    with pytest.raises(ValueError):
        ground_concepts(acts, np.zeros((10, 5), dtype=np.uint8))  # names required


# --------------------------------------------------------------------------- #
# confound probe
# --------------------------------------------------------------------------- #


def _confound_fixture(seed: int = 11, n: int = 240):
    """Concept 0 tracks body site (a confound); concept 1 tracks pigment network."""
    rng = np.random.default_rng(seed)
    derm = (rng.random((n, len(DERMOSCOPIC_ATTRIBUTES))) < 0.4).astype(np.uint8)
    meta = np.zeros((n, len(METADATA_ATTRIBUTES)), dtype=np.uint8)
    foot = METADATA_ATTRIBUTES.index("location: foot")
    male = METADATA_ATTRIBUTES.index("sex: male")
    on_foot = rng.random(n) < 0.35
    meta[on_foot, foot] = 1
    meta[rng.random(n) < 0.5, male] = 1
    acts = rng.standard_normal((n, 2)) * 0.5
    acts[:, 0] += 2.5 * on_foot           # a body-site shortcut
    acts[:, 1] += 2.5 * derm[:, 0]        # an honest dermoscopic concept
    return derm, meta, acts


def test_confound_probe_fires_on_metadata_and_stays_quiet_on_dermoscopic():
    derm, meta, acts = _confound_fixture()
    reports = probe_metadata_confound(
        acts, meta, METADATA_ATTRIBUTES, dermoscopic=derm,
        dermoscopic_names=DERMOSCOPIC_ATTRIBUTES,
    )
    shortcut, honest = reports
    assert shortcut.flagged is True
    assert shortcut.metadata_attribute == "location: foot"
    assert shortcut.metadata_effect > shortcut.dermoscopic_effect
    assert shortcut.confound_score > 0.8
    assert "body-site" in shortcut.message and "caution" in shortcut.message

    assert honest.flagged is False
    assert honest.dermoscopic_attribute == "pigment_network"
    assert honest.confound_score < 0.5
    assert "dermoscopic feature" in honest.message


def test_confound_score_is_bounded_and_comparative():
    derm, meta, acts = _confound_fixture()
    for r in probe_metadata_confound(
        acts, meta, METADATA_ATTRIBUTES, dermoscopic=derm,
        dermoscopic_names=DERMOSCOPIC_ATTRIBUTES,
    ):
        assert 0.0 <= r.confound_score <= 1.0
        payload = r.to_json()
        # The UI's comparison: both sides of the same measurement.
        assert {"metadata_effect", "dermoscopic_effect", "margin", "message"} <= set(payload)


def test_metadata_can_never_become_a_concept_name():
    derm, meta, acts = _confound_fixture()
    records = grounding_report(
        acts, derm, meta,
        dermoscopic_names=DERMOSCOPIC_ATTRIBUTES,
        metadata_names=METADATA_ATTRIBUTES,
    )
    shortcut = records[0]
    # The body-site concept is flagged, and stays an ID: no label was invented.
    assert shortcut["confound"]["flagged"] is True
    assert shortcut["label"] is None
    assert shortcut["display"] == "concept_0"
    named = [a["attribute"] for a in shortcut["attributes"]]
    assert all(a in DERMOSCOPIC_ATTRIBUTES for a in named)
    # And the honest concept is named after a dermoscopic criterion only.
    assert records[1]["label"] == "pigment network"
    assert records[1]["confound"]["flagged"] is False


def test_grounding_report_without_metadata_has_no_confound():
    derm, _meta, acts = _confound_fixture()
    records = grounding_report(acts, derm, dermoscopic_names=DERMOSCOPIC_ATTRIBUTES)
    assert all(r["confound"] is None for r in records)


def test_metadata_attribute_matrix_from_ham10000_rows():
    rows = [
        {"image_id": "ISIC_1", "sex": "male", "age": "75", "localization": "foot"},
        {"image_id": "ISIC_2", "sex": "female", "age": "", "localization": "unknown"},
        {"image_id": "ISIC_3", "sex": "unknown", "age": "95", "localization": "scalp"},
    ]
    table = build_metadata_attributes(rows)
    assert table.n_attributes == len(METADATA_ATTRIBUTES)
    idx = {a: i for i, a in enumerate(METADATA_ATTRIBUTES)}
    assert table.matrix[0, idx["sex: male"]] == 1
    assert table.matrix[0, idx["age: 60-80"]] == 1
    assert table.matrix[0, idx["location: foot"]] == 1
    assert table.matrix[1].sum() == 1  # only sex is parseable
    assert table.matrix[2, idx["age: 80+"]] == 1
    assert "confound probe only" in table.provenance["role"]


# --------------------------------------------------------------------------- #
# ISIC 2018 Task 2 loader
# --------------------------------------------------------------------------- #


def test_loader_builds_binary_matrix(tmp_path):
    root = _make_task2_tree(
        tmp_path,
        {
            "ISIC_0000000": _full(pigment_network=True, globules=True),
            "ISIC_0000001": _full(streaks=True),
            "ISIC_0000002": _full(),  # every mask present but empty
        },
    )
    table = load_isic2018_task2_attributes(root)
    assert isinstance(table, AttributeTable)
    assert table.matrix.shape == (3, 5)
    assert table.image_ids == ["ISIC_0000000", "ISIC_0000001", "ISIC_0000002"]
    assert table.attribute_names == list(DERMOSCOPIC_ATTRIBUTES)
    np.testing.assert_array_equal(
        table.matrix,
        np.array(
            [
                [1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],  # empty masks -> absent, not missing
            ],
            dtype=np.uint8,
        ),
    )
    assert list(table.support()) == [1, 0, 1, 0, 1]
    assert table.provenance["n_images"] == 3
    assert table.provenance["n_mask_files"] == 15
    assert table.provenance["n_empty_masks"] == 12
    assert table.provenance["n_missing_masks"] == 0
    assert table.provenance["n_training_images"] == 2594  # provenance carried through
    assert table.to_json()["matrix"][0] == [1, 0, 0, 0, 1]


def test_loader_tolerates_missing_criteria_and_case(tmp_path):
    root = tmp_path / "isic"
    gt = root / "ISIC2018_Task2_Training_GroundTruth_v3"
    # Only two of the five criteria annotated, one of them upper-cased with a
    # hyphen and a plural spelling, one with a .PNG suffix.
    _write_png(gt / "ISIC_0000010_attribute_pigment_network.png", _mask(True))
    _write_png(gt / "ISIC_0000010_ATTRIBUTE_Milia-Like-Cysts.PNG", _mask(True))
    _write_png(gt / "ISIC_0000011_attribute_streaks.png", _mask(False))

    table = load_isic2018_task2_attributes(root)
    assert table.image_ids == ["ISIC_0000010", "ISIC_0000011"]
    idx = {a: i for i, a in enumerate(DERMOSCOPIC_ATTRIBUTES)}
    assert table.matrix[0, idx["pigment_network"]] == 1
    assert table.matrix[0, idx["milia_like_cyst"]] == 1
    assert table.matrix[0, idx["globules"]] == 0       # no mask -> absent
    assert table.matrix[1].sum() == 0
    assert table.provenance["n_missing_masks"] == 7    # 3 + 4 unannotated criteria


def test_loader_reads_filtered_and_rgb_masks(tmp_path):
    """Real releases are adaptively filtered; some masks are stored as RGB."""
    root = tmp_path / "isic"
    gt = root / "gt"
    _write_png(gt / "ISIC_1_attribute_globules.png", _mask(True), filter_type=4)
    _write_png(gt / "ISIC_1_attribute_streaks.png", _mask(False), filter_type=4)
    _write_png(gt / "ISIC_2_attribute_globules.png", _mask(True), rgb=True, filter_type=2)
    _write_png(gt / "ISIC_2_attribute_streaks.png", _mask(False), rgb=True, filter_type=2)

    table = load_isic2018_task2_attributes(root)
    idx = {a: i for i, a in enumerate(DERMOSCOPIC_ATTRIBUTES)}
    assert list(table.matrix[:, idx["globules"]]) == [1, 1]
    assert list(table.matrix[:, idx["streaks"]]) == [0, 0]


def test_loader_skips_requested_images_without_masks(tmp_path):
    root = _make_task2_tree(tmp_path, {"ISIC_A": _full(streaks=True)})
    table = load_isic2018_task2_attributes(root, image_ids=["ISIC_A", "ISIC_MISSING"])
    assert table.image_ids == ["ISIC_A"]
    assert table.provenance["n_images_skipped_no_masks"] == 1
    assert table.provenance["images_skipped_no_masks"] == ["ISIC_MISSING"]


def test_loader_raises_actionable_error_when_nothing_matches(tmp_path):
    empty = tmp_path / "wrong_dir"
    (empty / "images").mkdir(parents=True)
    _write_png(empty / "images" / "ISIC_0000000.png", _mask(True))
    _write_png(empty / "images" / "ISIC_0000000_segmentation.png", _mask(True))

    with pytest.raises(FileNotFoundError) as exc:
        load_isic2018_task2_attributes(empty)
    msg = str(exc.value)
    assert "ISIC_<image_id>_attribute_<attribute>.png" in msg
    assert "pigment_network" in msg

    with pytest.raises(FileNotFoundError):
        load_isic2018_task2_attributes(tmp_path / "does_not_exist")


def test_loader_rejects_attributes_outside_the_vocabulary(tmp_path):
    root = _make_task2_tree(tmp_path, {"ISIC_A": _full(streaks=True)})
    with pytest.raises(ValueError):
        load_isic2018_task2_attributes(root, attributes=["location: foot"])


def test_loader_output_drives_grounding_end_to_end(tmp_path):
    presence = {}
    for i in range(40):
        presence[f"ISIC_{i:07d}"] = _full(pigment_network=(i % 2 == 0), globules=(i % 5 == 0))
    root = _make_task2_tree(tmp_path, presence)
    table = load_isic2018_task2_attributes(root)

    rng = np.random.default_rng(1)
    acts = rng.standard_normal((table.n_images, 1)) * 0.3
    acts[:, 0] += 2.0 * table.matrix[:, 0]
    # The table carries its own names + display names.
    grounded = ground_concepts(acts, table)[0]
    assert grounded.label == "pigment network"


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_grounding_and_loader_are_deterministic(tmp_path):
    derm, meta, acts = _confound_fixture()
    first = grounding_report(acts, derm, meta,
                             dermoscopic_names=DERMOSCOPIC_ATTRIBUTES,
                             metadata_names=METADATA_ATTRIBUTES)
    second = grounding_report(acts, derm, meta,
                              dermoscopic_names=DERMOSCOPIC_ATTRIBUTES,
                              metadata_names=METADATA_ATTRIBUTES)
    assert first == second

    root = _make_task2_tree(
        tmp_path, {"ISIC_B": _full(streaks=True), "ISIC_A": _full(globules=True)}
    )
    t1 = load_isic2018_task2_attributes(root)
    t2 = load_isic2018_task2_attributes(root)
    assert t1.image_ids == t2.image_ids == ["ISIC_A", "ISIC_B"]  # sorted, not FS order
    np.testing.assert_array_equal(t1.matrix, t2.matrix)


def test_import_is_torch_free():
    """The M0 rule: this module must not drag the ML stack in."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import vitreous.dermoscopy as d
        assert "torch" not in sys.modules, "torch was imported at import time"
        assert "PIL" not in sys.modules, "PIL was imported at import time"
        assert len(d.DERMOSCOPIC_ATTRIBUTES) == 5
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
