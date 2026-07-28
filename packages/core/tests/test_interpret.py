"""Interpretability bundle (vitreous.interpret) — numpy-only, no torch.

The three measured views (Hebbian graph, dermoscopic grounding, clinical
metrics) are composed here for the first time, so these tests are about the
*seams* rather than the maths — each sibling module already pins its own
arithmetic down (test_hebbian.py / test_dermoscopy.py / test_clinical.py).

The fixture plants one of each kind of concept in a single synthetic run:

* a concept whose units fire with a **dermoscopic** attribute (pigment network)
  — it must be named;
* a concept whose units fire with **patient metadata** (body site: foot) — it
  must stay unnamed *and* surface in the top-level confound warnings;
* a concept whose units fire with nothing — it must stay unnamed and say why.

so a single bundle exercises "named", "unnamed", and "shortcut" at once. What
is asserted throughout is the honesty contract: absent blocks say why they are
absent, no label is ever written back into the graph, accuracy never travels
without the majority-class baseline, and the whole document is strict JSON.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from vitreous.dermoscopy import (
    DERMOSCOPIC_ATTRIBUTES,
    METADATA_ATTRIBUTES,
    AttributeTable,
)
from vitreous.hebbian import HebbianStats
from vitreous.interpret import (
    BUNDLE_KIND,
    INTERPRET_SCHEMA_VERSION,
    build_interpretability_bundle,
    concept_activations_from_units,
)

BLOCKS = 4                       # planted co-activation blocks == planted classes
BLOCK = 8                        # units per block
U = BLOCKS * BLOCK
LAYER = "neurons"
CLASS_NAMES = [
    "Melanoma",
    "Melanocytic nevi",
    "Basal cell carcinoma",
    "Benign keratosis",
]
TAXONOMY = {
    "Melanoma": True,
    "Melanocytic nevi": False,
    "Basal cell carcinoma": True,
    "Benign keratosis": False,
}

# Which planted block carries which signal, in *unit* space.
DERM_BLOCK = 0        # fires with pigment_network
CONFOUND_BLOCK = 1    # fires with location: foot  (the shortcut)
NOISE_BLOCK = 2       # fires with nothing
GLOBULES_BLOCK = 3    # fires with globules

FOOT = METADATA_ATTRIBUTES.index("location: foot")
PIGMENT = DERMOSCOPIC_ATTRIBUTES.index("pigment_network")
GLOBULES = DERMOSCOPIC_ATTRIBUTES.index("globules")


# --------------------------------------------------------------------------- #
# fixtures — a numpy replica of what the recorder / a probe pass would measure
# --------------------------------------------------------------------------- #


def _planted_stats(seed: int = 0, n_samples: int = 400) -> HebbianStats:
    """Co-activation stats whose units fire in ``BLOCKS`` disjoint blocks."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(BLOCKS), n_samples // BLOCKS)
    acts = np.zeros((n_samples, U), dtype=np.float64)
    for s in range(n_samples):
        b = int(labels[s])
        acts[s, b * BLOCK : (b + 1) * BLOCK] = 1.0 + 0.05 * rng.standard_normal(BLOCK)
    acts = np.maximum(acts, 0.0)
    a_hat = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
    return HebbianStats(
        coact=a_hat.T @ a_hat / n_samples,
        mean_act=a_hat.mean(axis=0),
        class_act=np.stack([a_hat[labels == c].mean(axis=0) for c in range(BLOCKS)]),
        class_count=np.array([(labels == c).sum() for c in range(BLOCKS)], dtype=float),
        unit_index=np.arange(U) * 2 + 1,  # non-identity channel ids
        layer=LAYER,
        num_updates=n_samples // 16,
    )


def _planted_probe(seed: int = 1, n: int = 240):
    """Probe-set attributes + per-image unit activations with planted signals."""
    rng = np.random.default_rng(seed)
    derm = (rng.random((n, len(DERMOSCOPIC_ATTRIBUTES))) < 0.4).astype(np.uint8)
    meta = np.zeros((n, len(METADATA_ATTRIBUTES)), dtype=np.uint8)
    on_foot = rng.random(n) < 0.35
    meta[on_foot, FOOT] = 1
    meta[rng.random(n) < 0.5, METADATA_ATTRIBUTES.index("sex: male")] = 1

    units = rng.standard_normal((n, U)) * 0.4
    units[:, DERM_BLOCK * BLOCK : (DERM_BLOCK + 1) * BLOCK] += 2.5 * derm[:, [PIGMENT]]
    units[:, CONFOUND_BLOCK * BLOCK : (CONFOUND_BLOCK + 1) * BLOCK] += (
        2.5 * on_foot[:, None]
    )
    units[:, GLOBULES_BLOCK * BLOCK : (GLOBULES_BLOCK + 1) * BLOCK] += (
        2.5 * derm[:, [GLOBULES]]
    )
    # NOISE_BLOCK is left as pure noise on purpose.
    return derm, meta, units


def _planted_split(seed: int = 2, n: int = 500):
    """A held-out split with an informative but imperfect classifier."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, BLOCKS, n)
    logits = rng.standard_normal((n, BLOCKS)) * 0.8
    logits[np.arange(n), y] += 1.8
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return y, e / e.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def stats():
    return _planted_stats()


@pytest.fixture(scope="module")
def probe():
    return _planted_probe()


@pytest.fixture(scope="module")
def split():
    return _planted_split()


@pytest.fixture(scope="module")
def full_bundle(stats, probe, split):
    derm, meta, units = probe
    y, p = split
    return build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        y_true=y,
        y_prob=p,
        unit_activations=units,
        dermoscopic=(derm, list(DERMOSCOPIC_ATTRIBUTES)),
        metadata=(meta, list(METADATA_ATTRIBUTES)),
        taxonomy=TAXONOMY,
        threshold=0.2,
        target_sensitivity=0.95,
        focus_class="Melanoma",
        k=4,
        n_concepts=BLOCKS,
        seed=0,
        provenance={"dataset": "planted", "split": "test"},
    )


@pytest.fixture(scope="module")
def graph_only_bundle(stats):
    return build_interpretability_bundle(stats, class_names=CLASS_NAMES, n_concepts=BLOCKS)


def _record_for_block(bundle, block: int):
    """The grounding record of the concept made of ``block``'s units.

    Concept ids are assigned by descending importance, so the fixture must look
    concepts up by their membership rather than assume an order.
    """
    for record in bundle["grounding"]["concepts"]:
        if all(u // BLOCK == block for u in record["units"]):
            return record
    raise AssertionError(f"no concept covers planted block {block}")


# --------------------------------------------------------------------------- #
# the full bundle
# --------------------------------------------------------------------------- #


def test_full_bundle_has_every_block(full_bundle):
    assert full_bundle["schema_version"] == INTERPRET_SCHEMA_VERSION
    assert full_bundle["kind"] == BUNDLE_KIND
    assert full_bundle["class_names"] == CLASS_NAMES
    assert full_bundle["num_classes"] == BLOCKS
    assert full_bundle["num_concepts"] == BLOCKS
    for block in ("graph", "grounding", "confound_probe", "clinical"):
        assert full_bundle[block]["available"] is True, block
        assert full_bundle[block]["reason_code"] == "ok"
        assert full_bundle[block]["reason"] is None
    assert isinstance(full_bundle["confound_warnings"], list)
    # provenance says which providers ran, with what, and skipped nothing.
    prov = full_bundle["provenance"]
    assert prov["dataset"] == "planted"
    assert prov["skipped"] == []
    assert {p["block"] for p in prov["providers"]} == {
        "graph",
        "grounding",
        "confound_probe",
        "clinical",
    }
    assert all(p["ran"] for p in prov["providers"])
    graph_params = next(p for p in prov["providers"] if p["block"] == "graph")["params"]
    assert graph_params["n_concepts"] == BLOCKS and graph_params["seed"] == 0


def test_graph_block_is_the_hebbian_asset(full_bundle, stats):
    asset = full_bundle["graph"]["asset"]
    assert asset["provider"] == "hebbian"
    assert asset["layer"] == LAYER
    assert asset["num_units"] == U
    assert asset["num_concepts"] == BLOCKS
    assert asset["class_names"] == CLASS_NAMES
    assert len(asset["edges"]) == U * 4          # k=4 partners per unit
    assert len(asset["membership"]) == U
    assert full_bundle["graph"]["num_units"] == U
    assert full_bundle["graph"]["num_dead_units"] == 0
    assert full_bundle["graph"]["concept_ids"] == [f"c:{i}" for i in range(BLOCKS)]
    # the caller's provenance travels inside the asset too, so an extracted
    # graph.json is still self-describing.
    assert asset["provenance"]["dataset"] == "planted"


def test_grounding_block_reports_its_measurement(full_bundle):
    g = full_bundle["grounding"]
    assert g["n_images"] == 240
    assert g["n_concepts"] == BLOCKS
    assert g["n_named"] + g["n_unnamed"] == BLOCKS
    assert g["n_named"] >= 2  # pigment network + globules were planted
    assert g["vocabulary"]["names"] == list(DERMOSCOPIC_ATTRIBUTES)
    assert g["params"]["min_support"] == 5
    assert "unit_activations" in g["activation_source"]
    named = _record_for_block(full_bundle, DERM_BLOCK)
    assert named["named"] is True
    assert named["label"] == "pigment network"
    assert named["attributes"][0]["attribute"] == "pigment_network"
    assert named["attributes"][0]["support"] >= 5
    assert named["display"] == "pigment network"
    assert _record_for_block(full_bundle, GLOBULES_BLOCK)["label"] == "globules"


def test_clinical_block_is_the_honest_report(full_bundle, split):
    y, _ = split
    c = full_bundle["clinical"]
    assert c["n"] == int(y.size)
    assert c["report"]["clinical_schema_version"] == 1
    assert c["summary"] == c["report"]["summary"]
    assert "sensitivity" in c["summary"] and "majority-class baseline" in c["summary"]
    assert c["threshold"] == 0.2
    assert c["target_sensitivity"] == 0.95
    assert c["headline"]["roc_auc"] > 0.5
    assert c["headline"]["focus_class"]["class"] == "Melanoma"
    assert c["headline"]["sensitivity_at_threshold"]["n"] > 0
    assert c["report"]["binary"]["at_threshold"]["threshold"] == 0.2
    assert c["notes"] == []


def test_malignant_group_comes_from_the_taxonomy(full_bundle):
    c = full_bundle["clinical"]
    assert c["positive_classes"] == ["Melanoma", "Basal cell carcinoma"]
    assert c["positive_class_indices"] == [0, 2]
    assert c["positive_class_source"] == "taxonomy.malignant"
    assert c["report"]["binary"]["label"] == "malignancy"
    assert c["taxonomy"]["Melanoma"] is True


def test_full_bundle_json_round_trips(full_bundle):
    text = json.dumps(full_bundle, allow_nan=False)  # strict: no NaN/Infinity
    assert json.loads(text) == full_bundle


# --------------------------------------------------------------------------- #
# degradation — a caller with only Hebbian stats
# --------------------------------------------------------------------------- #


def test_graph_only_bundle_degrades_with_reasons(graph_only_bundle):
    b = graph_only_bundle
    assert b["graph"]["available"] is True
    assert b["graph"]["asset"]["num_units"] == U
    for block, code in (
        ("grounding", "missing_concept_activations"),
        ("confound_probe", "missing_concept_activations"),
        ("clinical", "missing_labels"),
    ):
        assert b[block]["available"] is False, block
        assert b[block]["reason_code"] == code
        assert isinstance(b[block]["reason"], str) and b[block]["reason"]
    assert b["confound_warnings"] == []
    assert b["clinical"]["report"] is None and b["clinical"]["summary"] is None
    assert [s["block"] for s in b["provenance"]["skipped"]] == [
        "grounding",
        "confound_probe",
        "clinical",
    ]
    assert json.loads(json.dumps(b, allow_nan=False)) == b


def test_absent_block_is_not_measured_zero(graph_only_bundle, full_bundle):
    """"nothing was looked for" must be distinguishable from "nothing found"."""
    absent = graph_only_bundle["confound_probe"]
    assert absent["available"] is False
    assert absent["n_flagged"] is None and absent["n_concepts_probed"] is None
    measured = full_bundle["confound_probe"]
    assert measured["available"] is True
    assert measured["n_flagged"] == 1 and measured["n_concepts_probed"] == BLOCKS


def test_clinical_only_bundle(stats, split):
    y, p = split
    b = build_interpretability_bundle(
        stats, class_names=CLASS_NAMES, y_true=y, y_prob=p, n_concepts=BLOCKS
    )
    assert b["clinical"]["available"] is True
    assert b["grounding"]["available"] is False
    assert b["confound_probe"]["available"] is False
    assert b["confound_warnings"] == []


def test_labels_without_probabilities_says_so(stats, split):
    y, _ = split
    b = build_interpretability_bundle(
        stats, class_names=CLASS_NAMES, y_true=y, n_concepts=BLOCKS
    )
    assert b["clinical"]["reason_code"] == "missing_probabilities"
    assert "probabilities" in b["clinical"]["reason"]


def test_grounding_without_metadata_still_names_concepts(stats, probe):
    derm, _meta, units = probe
    b = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        unit_activations=units,
        dermoscopic=(derm, list(DERMOSCOPIC_ATTRIBUTES)),
        n_concepts=BLOCKS,
    )
    assert b["grounding"]["available"] is True
    assert b["grounding"]["n_named"] >= 2
    assert b["confound_probe"]["available"] is False
    assert b["confound_probe"]["reason_code"] == "missing_metadata_attributes"
    assert "not the same as none being present" in b["confound_probe"]["reason"]


def test_confound_probe_without_dermoscopic_declares_its_weakness(stats, probe):
    _derm, meta, units = probe
    b = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        unit_activations=units,
        metadata=(meta, list(METADATA_ATTRIBUTES)),
        n_concepts=BLOCKS,
    )
    assert b["grounding"]["available"] is False
    assert b["grounding"]["reason_code"] == "missing_dermoscopic_attributes"
    probe_block = b["confound_probe"]
    assert probe_block["available"] is True
    assert probe_block["has_dermoscopic_counterweight"] is False
    assert "counterweight" in probe_block["caveat"]


# --------------------------------------------------------------------------- #
# the join: grounding names a graph concept without touching the graph
# --------------------------------------------------------------------------- #


def test_grounding_joins_onto_graph_concept_ids(full_bundle):
    graph_ids = {f"c:{c['id']}" for c in full_bundle["graph"]["asset"]["concepts"]}
    assert graph_ids == set(full_bundle["graph"]["concept_ids"])
    grounded_ids = {r["concept_id"] for r in full_bundle["grounding"]["concepts"]}
    assert grounded_ids == graph_ids
    probed_ids = {r["concept_id"] for r in full_bundle["confound_probe"]["concepts"]}
    assert probed_ids == graph_ids
    # the join is a real join: the record's units are the graph concept's units.
    for record in full_bundle["grounding"]["concepts"]:
        concept = next(
            c
            for c in full_bundle["graph"]["asset"]["concepts"]
            if f"c:{c['id']}" == record["concept_id"]
        )
        assert record["units"] == concept["units"]
        assert record["concept"] == concept["id"]


def test_graph_concepts_stay_unnamed(full_bundle):
    """No label may be written back into the graph, named concepts included."""
    graph_block = full_bundle["graph"]
    for concept in graph_block["asset"]["concepts"]:
        assert "label" not in concept and "name" not in concept
    for node in graph_block["asset"]["nodes"]:
        assert "label" not in node

    def _walk(obj, path="graph"):
        if isinstance(obj, dict):
            for key, value in obj.items():
                assert key != "label", f"a label leaked into the graph at {path}"
                _walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                _walk(value, f"{path}[{i}]")

    _walk(graph_block)
    assert "unnamed by construction" in graph_block["labels"]


def test_unnamed_concept_stays_unnamed_end_to_end(full_bundle):
    """The planted noise concept must carry no name anywhere in the bundle."""
    noise = _record_for_block(full_bundle, NOISE_BLOCK)
    assert noise["named"] is False
    assert noise["label"] is None
    assert noise["display"] == noise["concept_id"]      # falls back to the id
    assert noise["reason"].startswith("unnamed:")       # ... and says why
    # the evidence is still reported, just too weak to be a name.
    assert noise["attributes"] and all(not a["qualifies"] for a in noise["attributes"])

    concept_id = noise["concept_id"]
    labels = [
        r["label"]
        for r in full_bundle["grounding"]["concepts"]
        if r["concept_id"] == concept_id
    ]
    assert labels == [None]
    # nothing else in the bundle attaches a name to this concept id.
    graph_concept = next(
        c
        for c in full_bundle["graph"]["asset"]["concepts"]
        if f"c:{c['id']}" == concept_id
    )
    assert set(graph_concept) == {
        "id",
        "kind",
        "layer",
        "units",
        "channels",
        "coherence",
        "importance",
        "class_scores",
        "class_affinity",
    }


# --------------------------------------------------------------------------- #
# the shortcut detector, surfaced at the top level
# --------------------------------------------------------------------------- #


def test_metadata_tracking_concept_surfaces_in_top_level_warnings(full_bundle):
    warnings = full_bundle["confound_warnings"]
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning["metadata_attribute"] == "location: foot"
    assert warning["metadata_effect"] > warning["dermoscopic_effect"]
    assert warning["confound_score"] > 0.8
    assert "body-site" in warning["message"] and "caution" in warning["message"]
    assert warning["severity"] == "caution"

    # ... and it is the concept built from the planted confound block, which is
    # still unnamed: a flagged shortcut never earns a label.
    shortcut = _record_for_block(full_bundle, CONFOUND_BLOCK)
    assert warning["concept_id"] == shortcut["concept_id"]
    assert shortcut["label"] is None and shortcut["named"] is False

    # the honest dermoscopic concept is not flagged.
    honest = _record_for_block(full_bundle, DERM_BLOCK)
    flagged_ids = {w["concept_id"] for w in warnings}
    assert honest["concept_id"] not in flagged_ids
    probe_record = next(
        r
        for r in full_bundle["confound_probe"]["concepts"]
        if r["concept_id"] == honest["concept_id"]
    )
    assert probe_record["flagged"] is False
    assert probe_record["dermoscopic_attribute"] == "pigment_network"


def test_confound_warnings_are_top_level_and_never_null(graph_only_bundle, full_bundle):
    for bundle in (graph_only_bundle, full_bundle):
        assert "confound_warnings" in bundle
        assert isinstance(bundle["confound_warnings"], list)
    # every warning is joinable and self-explanatory without opening a block.
    for warning in full_bundle["confound_warnings"]:
        assert {
            "concept_id",
            "metadata_attribute",
            "metadata_effect",
            "dermoscopic_effect",
            "margin",
            "confound_score",
            "message",
        } <= set(warning)


# --------------------------------------------------------------------------- #
# accuracy can never be rendered without its floor
# --------------------------------------------------------------------------- #


def _majority_baseline_in(obj) -> bool:
    return isinstance(obj, dict) and (
        "majority_class_rate" in obj
        or (
            isinstance(obj.get("baselines"), dict)
            and "majority_class_rate" in obj["baselines"]
        )
    )


def test_accuracy_never_appears_without_the_majority_baseline(full_bundle):
    """Every 'accuracy' in the clinical block has its floor in reach."""
    seen = 0

    def _walk(obj, ancestors, path):
        nonlocal seen
        if isinstance(obj, dict):
            if "accuracy" in obj:
                seen += 1
                assert any(
                    _majority_baseline_in(a) for a in [obj, *ancestors]
                ), f"accuracy at {path} has no majority-class baseline in reach"
            for key, value in obj.items():
                _walk(value, [obj, *ancestors], f"{path}.{key}")
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                _walk(value, ancestors, f"{path}[{i}]")

    _walk(full_bundle["clinical"], [], "clinical")
    assert seen >= 2  # multiclass accuracy + the binary operating point

    # The headline pairing is stronger still: same object, not a sibling a
    # renderer could drop.
    headline = full_bundle["clinical"]["headline"]["accuracy"]
    assert headline["value"] is not None
    assert headline["majority_class_rate"] == pytest.approx(
        full_bundle["clinical"]["baselines"]["majority_class_rate"]
    )
    assert headline["majority_class_name"] in CLASS_NAMES
    assert headline["uniform_chance"] == pytest.approx(1 / BLOCKS)
    assert headline["meaningful_floor"] in ("majority_class_rate", "uniform_chance")
    assert headline["margin_over_majority_class"] == pytest.approx(
        headline["value"] - headline["majority_class_rate"]
    )
    assert "majority_class_rate" in headline["note"]


def test_summary_travels_with_the_clinical_block(full_bundle):
    summary = full_bundle["clinical"]["summary"]
    assert summary.index("accuracy ") < summary.index("majority-class baseline")
    assert full_bundle["clinical"]["headline"]["summary"] == summary


# --------------------------------------------------------------------------- #
# serialization + determinism
# --------------------------------------------------------------------------- #


def _assert_json_native(obj, path="root"):
    if obj is None or isinstance(obj, (bool, str)):
        assert not isinstance(obj, np.generic), f"numpy scalar leaked at {path}"
        return
    if isinstance(obj, (int, float)):
        assert not isinstance(obj, np.generic), f"numpy scalar leaked at {path}"
        if isinstance(obj, float):
            assert np.isfinite(obj), f"non-finite float at {path}"
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert isinstance(key, str), f"non-str key at {path}"
            _assert_json_native(value, f"{path}.{key}")
        return
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            _assert_json_native(value, f"{path}[{i}]")
        return
    raise AssertionError(f"non-JSON type {type(obj)!r} at {path}")


def test_no_numpy_scalars_and_no_nan_escape(full_bundle, graph_only_bundle):
    for bundle in (full_bundle, graph_only_bundle):
        _assert_json_native(bundle)
        json.dumps(bundle, allow_nan=False)
        assert bundle["provenance"]["non_finite_coerced"] == []


def test_non_finite_inputs_become_null_and_are_recorded(stats, probe):
    """A NaN that sneaks in is reported as a named hole, not as invalid JSON."""
    derm, _meta, units = probe
    broken = np.array(units, dtype=np.float64)
    broken[0, :] = np.nan
    bundle = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        unit_activations=broken,
        dermoscopic=(derm, list(DERMOSCOPIC_ATTRIBUTES)),
        n_concepts=BLOCKS,
    )
    json.dumps(bundle, allow_nan=False)  # still strict JSON
    _assert_json_native(bundle)
    assert bundle["provenance"]["non_finite_coerced"], "the hole must be named"
    assert all(
        p.startswith("bundle.") for p in bundle["provenance"]["non_finite_coerced"]
    )


def test_bundle_is_deterministic(stats, probe, split):
    derm, meta, units = probe
    y, p = split

    def build():
        return build_interpretability_bundle(
            stats,
            class_names=CLASS_NAMES,
            y_true=y,
            y_prob=p,
            unit_activations=units,
            dermoscopic=(derm, list(DERMOSCOPIC_ATTRIBUTES)),
            metadata=(meta, list(METADATA_ATTRIBUTES)),
            taxonomy=TAXONOMY,
            n_concepts=BLOCKS,
            seed=0,
        )

    first, second = build(), build()
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    # no wall-clock anywhere: a rebuild is diffable against what shipped.
    assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", json.dumps(first))
    assert "no timestamp is recorded" in first["provenance"]["determinism"]


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #


def test_concept_activations_from_units_is_the_member_mean(stats, probe):
    _derm, _meta, units = probe
    bundle = build_interpretability_bundle(
        stats, class_names=CLASS_NAMES, n_concepts=BLOCKS
    )
    concepts = bundle["graph"]["asset"]["concepts"]
    acts = concept_activations_from_units(units, concepts)
    assert acts.shape == (units.shape[0], len(concepts))
    expected = units[:, concepts[0]["units"]].mean(axis=1)
    np.testing.assert_allclose(acts[:, 0], expected)
    with pytest.raises(ValueError):
        concept_activations_from_units(units[:, :4], concepts)
    with pytest.raises(ValueError):
        concept_activations_from_units(units[0], concepts)


def test_precomputed_concept_activations_match_unit_projection(stats, probe):
    derm, _meta, units = probe
    base = build_interpretability_bundle(
        stats, class_names=CLASS_NAMES, n_concepts=BLOCKS
    )
    acts = concept_activations_from_units(units, base["graph"]["asset"]["concepts"])
    via_units = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        unit_activations=units,
        dermoscopic=(derm, list(DERMOSCOPIC_ATTRIBUTES)),
        n_concepts=BLOCKS,
    )
    via_concepts = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        concept_activations=acts,
        dermoscopic=(derm, list(DERMOSCOPIC_ATTRIBUTES)),
        n_concepts=BLOCKS,
    )
    assert via_units["grounding"]["concepts"] == via_concepts["grounding"]["concepts"]
    assert via_concepts["grounding"]["activation_source"] == "concept_activations"


def test_attribute_table_input_is_accepted(stats, probe):
    derm, _meta, units = probe
    table = AttributeTable(
        image_ids=[f"ISIC_{i:07d}" for i in range(derm.shape[0])],
        attribute_names=list(DERMOSCOPIC_ATTRIBUTES),
        matrix=derm,
        display_names={"pigment_network": "pigment network"},
        provenance={"dataset": "ISIC 2018 Task 2"},
    )
    bundle = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        unit_activations=units,
        dermoscopic=table,
        n_concepts=BLOCKS,
    )
    assert bundle["grounding"]["available"] is True
    assert bundle["grounding"]["vocabulary"]["provenance"]["dataset"] == (
        "ISIC 2018 Task 2"
    )


def test_bare_matrix_with_the_canonical_column_count_is_accepted(stats, probe):
    derm, _meta, units = probe
    bundle = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        unit_activations=units,
        dermoscopic=derm,  # [N, 5] — the ISIC 2018 Task 2 vocabulary
        n_concepts=BLOCKS,
    )
    assert bundle["grounding"]["vocabulary"]["names"] == list(DERMOSCOPIC_ATTRIBUTES)


def test_shape_mismatches_raise_rather_than_degrade(stats, probe, split):
    """A wrong shape is a bug; degrading would hide it behind an honest null."""
    derm, _meta, units = probe
    y, p = split
    with pytest.raises(ValueError, match="attribute row"):
        build_interpretability_bundle(
            stats,
            class_names=CLASS_NAMES,
            unit_activations=units,
            dermoscopic=(derm[:-5], list(DERMOSCOPIC_ATTRIBUTES)),
            n_concepts=BLOCKS,
        )
    with pytest.raises(ValueError, match="clustering produced"):
        build_interpretability_bundle(
            stats,
            class_names=CLASS_NAMES,
            concept_activations=np.zeros((240, BLOCKS + 3)),
            dermoscopic=(derm, list(DERMOSCOPIC_ATTRIBUTES)),
            n_concepts=BLOCKS,
        )
    with pytest.raises(ValueError, match="class_names"):
        build_interpretability_bundle(stats, class_names=["a", "b"])
    with pytest.raises(ValueError, match="columns"):
        build_interpretability_bundle(
            stats, class_names=CLASS_NAMES, y_true=y, y_prob=p[:, :2]
        )
    with pytest.raises(ValueError, match="bare"):
        build_interpretability_bundle(
            stats,
            class_names=CLASS_NAMES,
            unit_activations=units,
            dermoscopic=np.zeros((240, 4)),  # not the 5-column vocabulary
            n_concepts=BLOCKS,
        )


def test_stats_dict_form_is_accepted(stats):
    bundle = build_interpretability_bundle(
        stats.to_dict(), class_names=CLASS_NAMES, n_concepts=BLOCKS
    )
    assert bundle["graph"]["asset"]["num_units"] == U
    with pytest.raises(TypeError):
        build_interpretability_bundle("not stats", class_names=CLASS_NAMES)


def test_class_names_default_to_positional_placeholders(stats):
    bundle = build_interpretability_bundle(stats, class_names=None, n_concepts=BLOCKS)
    assert bundle["class_names"] == [f"class_{i}" for i in range(BLOCKS)]


def test_missing_focus_class_is_recorded_not_fatal(stats, split):
    y, p = split
    bundle = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        y_true=y,
        y_prob=p,
        focus_class="Dermatofibroma",  # not in this dataset
        n_concepts=BLOCKS,
    )
    notes = " ".join(bundle["clinical"]["notes"])
    assert "Dermatofibroma" in notes
    # falls back to melanoma, which this dataset does have.
    assert bundle["clinical"]["headline"]["focus_class"]["class"] == "Melanoma"


def test_without_a_taxonomy_the_binary_group_falls_back_to_melanoma(stats, split):
    y, p = split
    bundle = build_interpretability_bundle(
        stats, class_names=CLASS_NAMES, y_true=y, y_prob=p, n_concepts=BLOCKS
    )
    c = bundle["clinical"]
    assert c["positive_classes"] == ["Melanoma"]
    assert c["positive_class_source"].startswith("melanoma class")
    assert any("no taxonomy" in note for note in c["notes"])


def test_explicit_positive_classes_win_over_the_taxonomy(stats, split):
    y, p = split
    bundle = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        y_true=y,
        y_prob=p,
        taxonomy=TAXONOMY,
        positive_classes=["Melanoma"],
        n_concepts=BLOCKS,
    )
    assert bundle["clinical"]["positive_classes"] == ["Melanoma"]
    assert bundle["clinical"]["positive_class_source"] == "explicit positive_classes"


def test_dataclass_taxonomy_is_accepted(stats, split):
    from vitreous.data import Taxonomy

    y, p = split
    taxonomy = Taxonomy(
        malignant=dict(TAXONOMY),
        category_level={c: (1 if TAXONOMY[c] else 0) for c in CLASS_NAMES},
        category_labels=["benign", "malignant"],
    )
    bundle = build_interpretability_bundle(
        stats,
        class_names=CLASS_NAMES,
        y_true=y,
        y_prob=p,
        taxonomy=taxonomy,
        n_concepts=BLOCKS,
    )
    assert bundle["clinical"]["positive_class_indices"] == [0, 2]
    assert bundle["clinical"]["taxonomy"]["malignant"]["Melanoma"] is True
    json.dumps(bundle, allow_nan=False)


def test_ham10000_taxonomy_from_the_registry_drives_the_group():
    """The real HAM10000 taxonomy picks out {mel, bcc, akiec}."""
    from vitreous.data import get_dataset

    spec = get_dataset("ham10000").spec
    names = list(spec.class_names)
    stats = HebbianStats(
        coact=np.eye(12) + 0.05,
        mean_act=np.ones(12),
        class_act=np.ones((len(names), 12)),
        unit_index=np.arange(12),
        layer="neurons",
    )
    rng = np.random.default_rng(0)
    y = rng.integers(0, len(names), 300)
    p = rng.random((300, len(names)))
    p = p / p.sum(axis=1, keepdims=True)
    bundle = build_interpretability_bundle(
        stats, class_names=names, y_true=y, y_prob=p, taxonomy=spec.taxonomy,
    )
    assert bundle["clinical"]["positive_class_source"] == "taxonomy.malignant"
    assert len(bundle["clinical"]["positive_classes"]) == 3
    assert "Melanoma" in bundle["clinical"]["positive_classes"]
    assert bundle["clinical"]["report"]["binary"]["label"] == "malignancy"


# --------------------------------------------------------------------------- #
# import purity (the M0 rule)
# --------------------------------------------------------------------------- #


def test_import_does_not_pull_in_torch():
    """`import vitreous.interpret` must not import torch, even if installed.

    Run in a subprocess so a torch already imported by another test cannot mask
    a regression — the same pattern test_clinical.py / test_hebbian.py use.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import vitreous.interpret as interpret
        assert "torch" not in sys.modules, "torch was imported at import time"
        assert "sklearn" not in sys.modules, "sklearn was imported at import time"
        assert "PIL" not in sys.modules, "PIL was imported at import time"
        assert "networkx" not in sys.modules, "networkx was imported at import time"
        assert callable(interpret.build_interpretability_bundle)
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
