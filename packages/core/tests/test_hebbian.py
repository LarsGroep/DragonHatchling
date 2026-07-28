"""HebbianGraphProvider + build_hebbian_graph_asset — offline, numpy-only.

Synthetic co-activation with a *planted* block structure; no torch, no model —
mirroring the M0 discipline of test_graph.py / test_som.py. The synthetic stats
are produced by replicating in numpy exactly what ``HebbianRecorder`` measures
(pool → rectify → L2-normalize → outer product), so what the provider sees here
is the same object shape it sees in production.

Covers planted-structure recovery (communities + concepts), determinism, exact
top-k edge semantics incl. tie-breaking, dead-unit exclusion, the HebbianStats
round-trips, the asset's structural contract (mirroring test_graph.py's
build_graph_asset assertions), and the M0 import-purity rule.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vitreous.graph import Community, GraphEdge, GraphNode, GraphProvider
from vitreous.hebbian import (
    DEFAULT_ACTIVITY_THRESHOLD,
    HebbianConcept,
    HebbianGraphProvider,
    HebbianStats,
    build_hebbian_graph_asset,
    class_node_id,
    cluster_concepts,
    concept_node_id,
    correlation_matrix,
    unit_node_id,
    ward_labels,
)

BLOCKS = 4          # planted blocks == planted classes
BLOCK = 12          # units per block
U_LIVE = BLOCKS * BLOCK   # 48 live units
N_DEAD = 6          # near-dead units appended after the live ones
K = 8
LAYER = "neurons"
CLASS_NAMES = ["akiec", "bcc", "mel", "nv"]


# --------------------------------------------------------------------------- #
# synthetic stats — a numpy replica of what HebbianRecorder measures
# --------------------------------------------------------------------------- #


def _planted_stats(seed: int = 0, n_dead: int = 0, n_samples: int = 480):
    """Stats whose units fire in ``BLOCKS`` disjoint blocks, one block per class.

    Each sample activates exactly one block (plus optional near-dead units that
    twitch on every sample at 1e-3 amplitude). Activations are rectified and
    L2-normalized per sample and the co-activation matrix is their mean outer
    product — the recorder's measurement, computed here in closed form.
    """
    rng = np.random.default_rng(seed)
    u = U_LIVE + n_dead
    labels = np.repeat(np.arange(BLOCKS), n_samples // BLOCKS)
    acts = np.zeros((n_samples, u), dtype=np.float64)
    for s in range(n_samples):
        b = int(labels[s])
        acts[s, b * BLOCK : (b + 1) * BLOCK] = 1.0 + 0.05 * rng.standard_normal(BLOCK)
        if n_dead:
            acts[s, U_LIVE:] = 1e-3 * rng.random(n_dead)
    acts = np.maximum(acts, 0.0)  # rectified, as the hook does
    a_hat = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)

    coact = a_hat.T @ a_hat / n_samples
    mean_act = a_hat.mean(axis=0)
    class_act = np.stack([a_hat[labels == c].mean(axis=0) for c in range(BLOCKS)])
    class_count = np.array([(labels == c).sum() for c in range(BLOCKS)], dtype=float)
    return HebbianStats(
        coact=coact,
        mean_act=mean_act,
        class_act=class_act,
        class_count=class_count,
        unit_index=np.arange(u) * 2 + 1,  # non-identity channel ids
        layer=LAYER,
        num_updates=n_samples // 16,
    )


def _block_of(unit: int) -> int:
    return unit // BLOCK


@pytest.fixture(scope="module")
def stats():
    """Live units only (no dead units) — the clean planted structure."""
    return _planted_stats(seed=0, n_dead=0)


@pytest.fixture(scope="module")
def stats_with_dead():
    return _planted_stats(seed=0, n_dead=N_DEAD)


@pytest.fixture(scope="module")
def provider(stats):
    return HebbianGraphProvider(stats, k=K, seed=0, class_names=CLASS_NAMES)


# --------------------------------------------------------------------------- #
# HebbianStats
# --------------------------------------------------------------------------- #


def test_stats_shapes(stats):
    assert stats.num_units == U_LIVE
    assert stats.num_classes == BLOCKS
    assert stats.coact.shape == (U_LIVE, U_LIVE)
    assert stats.class_act.shape == (BLOCKS, U_LIVE)
    assert stats.unit_index.shape == (U_LIVE,)


def test_stats_shape_validation():
    with pytest.raises(ValueError):
        HebbianStats(
            coact=np.zeros((4, 5)),
            mean_act=np.zeros(4),
            class_act=np.zeros((2, 4)),
            unit_index=np.arange(4),
        )
    with pytest.raises(ValueError):
        HebbianStats(
            coact=np.zeros((4, 4)),
            mean_act=np.zeros(3),
            class_act=np.zeros((2, 4)),
            unit_index=np.arange(4),
        )
    with pytest.raises(ValueError):
        HebbianStats(
            coact=np.zeros((4, 4)),
            mean_act=np.zeros(4),
            class_act=np.zeros((2, 3)),
            unit_index=np.arange(4),
        )


def test_stats_dict_roundtrip_is_lossless(stats):
    payload = stats.to_dict()
    # JSON-safe: plain lists, no ndarrays anywhere.
    assert isinstance(payload["coact"], list)
    assert isinstance(payload["coact"][0], list)
    assert not any(isinstance(v, np.ndarray) for v in payload.values())

    restored = HebbianStats.from_dict(payload)
    assert np.array_equal(restored.coact, stats.coact)
    assert np.array_equal(restored.mean_act, stats.mean_act)
    assert np.array_equal(restored.class_act, stats.class_act)
    assert np.array_equal(restored.class_count, stats.class_count)
    assert np.array_equal(restored.unit_index, stats.unit_index)
    assert restored.layer == stats.layer
    assert restored.num_updates == stats.num_updates

    # ... and survives an actual JSON encode/decode unchanged.
    via_json = HebbianStats.from_dict(json.loads(json.dumps(payload)))
    assert np.array_equal(via_json.coact, stats.coact)
    assert via_json.to_dict() == payload


def test_stats_npz_roundtrip(stats, tmp_path):
    path = tmp_path / "hebbian_stats.npz"
    stats.save_npz(path)
    restored = HebbianStats.load_npz(path)
    assert np.array_equal(restored.coact, stats.coact)
    assert np.array_equal(restored.class_act, stats.class_act)
    assert restored.layer == stats.layer
    assert restored.num_updates == stats.num_updates


def test_correlation_is_normalized(stats):
    corr = stats.correlation()
    assert corr.shape == (U_LIVE, U_LIVE)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert corr.min() >= -1e-12 and corr.max() <= 1.0 + 1e-9
    # Same-block pairs co-activate; cross-block pairs never fire together.
    assert corr[0, 1] > 0.5
    assert corr[0, BLOCK] < 1e-9


def test_correlation_handles_never_firing_units():
    coact = np.zeros((3, 3))
    coact[0, 0] = coact[1, 1] = 1.0
    coact[0, 1] = coact[1, 0] = 0.5
    corr = correlation_matrix(coact)
    assert np.isfinite(corr).all()
    assert corr[2, 2] == 0.0  # a dead unit's row is exactly zero, not NaN
    with pytest.raises(ValueError):
        correlation_matrix(np.zeros((2, 3)))


# --------------------------------------------------------------------------- #
# provider surface
# --------------------------------------------------------------------------- #


def test_provider_satisfies_protocol(provider):
    assert isinstance(provider, GraphProvider)


def test_provider_accepts_stats_per_call(stats):
    bound = HebbianGraphProvider(stats, k=K, seed=0, class_names=CLASS_NAMES)
    unbound = HebbianGraphProvider(k=K, seed=0, class_names=CLASS_NAMES)
    assert [e.weight for e in unbound.edges(stats)] == [
        e.weight for e in bound.edges()
    ]
    with pytest.raises(ValueError):
        HebbianGraphProvider(k=K).edges()


def test_node_counts_and_kinds(provider, stats):
    nodes = provider.nodes()
    units = [n for n in nodes if n.kind == "unit"]
    concepts = [n for n in nodes if n.kind == "concept"]
    classes = [n for n in nodes if n.kind == "class"]
    assert len(units) == U_LIVE
    assert len(classes) == BLOCKS
    assert len(concepts) == len(provider.concepts())
    assert len(nodes) == len(units) + len(concepts) + len(classes)
    assert len({n.id for n in nodes}) == len(nodes)  # ids unique
    assert isinstance(nodes[0], GraphNode)
    # unit nodes carry their original channel id, not just the tracked index.
    assert units[3].ref["channel"] == int(stats.unit_index[3])
    assert units[3].id == unit_node_id(LAYER, 3)
    assert classes[0].id == class_node_id(CLASS_NAMES[0])
    assert concepts[0].id == concept_node_id(0)


# --------------------------------------------------------------------------- #
# edges: exact top-k semantics
# --------------------------------------------------------------------------- #


def test_edge_count_and_per_unit_topk(provider, stats):
    edges = provider.edges()
    assert isinstance(edges[0], GraphEdge)
    assert len(edges) == U_LIVE * K
    by_dst: dict = {}
    for e in edges:
        by_dst.setdefault(e.target, []).append(e.source)
    assert len(by_dst) == U_LIVE
    assert all(len(v) == K for v in by_dst.values())
    # no self-edges (a unit's correlation with itself is 1 by construction).
    assert all(e.source != e.target for e in edges)
    assert all(0.0 <= e.weight <= 1.0 for e in edges)


def test_edges_are_the_actual_topk(provider, stats):
    corr = stats.correlation()
    edges = provider.edges()
    for dst in (0, 17, U_LIVE - 1):
        kept = sorted(
            int(e.source.rsplit(":", 1)[1])
            for e in edges
            if e.target == unit_node_id(LAYER, dst)
        )
        row = corr[dst].copy()
        row[dst] = -np.inf
        expected = sorted(int(j) for j in np.argsort(-row, kind="stable")[:K])
        assert kept == expected
        # every kept partner is in the same planted block
        assert all(_block_of(j) == _block_of(dst) for j in kept)


def test_edge_weights_match_correlation(provider, stats):
    corr = stats.correlation()
    for e in provider.edges()[:100]:
        src = int(e.source.rsplit(":", 1)[1])
        dst = int(e.target.rsplit(":", 1)[1])
        assert abs(e.weight - float(corr[dst, src])) < 1e-9


def test_edge_tie_breaking_is_deterministic():
    """With exact ties the lowest unit indices win, in ascending order."""
    u = 6
    coact = np.eye(u)
    coact[0, 1:] = 0.5
    coact[1:, 0] = 0.5
    st = HebbianStats(
        coact=coact,
        mean_act=np.ones(u),
        class_act=np.ones((2, u)),
        unit_index=np.arange(u),
        layer=LAYER,
    )
    p = HebbianGraphProvider(st, k=3, seed=0, class_names=["a", "b"])
    kept = [
        int(e.source.rsplit(":", 1)[1])
        for e in p.edges()
        if e.target == unit_node_id(LAYER, 0)
    ]
    assert kept == [1, 2, 3]  # ties broken towards the smallest index, in order


def test_k_is_capped_below_unit_count():
    u = 4
    st = HebbianStats(
        coact=np.eye(u) + 0.1,
        mean_act=np.ones(u),
        class_act=np.ones((2, u)),
        unit_index=np.arange(u),
    )
    p = HebbianGraphProvider(st, k=32)
    assert len(p.edges()) == u * (u - 1)


# --------------------------------------------------------------------------- #
# communities: planted-structure recovery
# --------------------------------------------------------------------------- #


def test_communities_recover_planted_blocks(provider):
    comms = provider.communities()
    assert isinstance(comms[0], Community)
    members = [m for c in comms for m in c.members]
    assert len(members) == U_LIVE          # covers every unit
    assert len(set(members)) == U_LIVE     # disjoint partition
    assert len(comms) == BLOCKS
    for c in comms:
        blocks = {_block_of(int(m.rsplit(":", 1)[1])) for m in c.members}
        assert len(blocks) == 1            # a community never spans two blocks
        assert len(c.members) == BLOCK     # ... and holds a whole block
    # communities are ordered by their smallest member
    firsts = [min(int(m.rsplit(":", 1)[1]) for m in c.members) for c in comms]
    assert firsts == sorted(firsts)


def test_communities_deterministic(stats):
    a = HebbianGraphProvider(stats, k=K, seed=42).communities()
    b = HebbianGraphProvider(stats, k=K, seed=42).communities()
    assert [c.members for c in a] == [c.members for c in b]


# --------------------------------------------------------------------------- #
# concepts
# --------------------------------------------------------------------------- #


def test_concepts_recover_planted_blocks(stats):
    concepts = cluster_concepts(stats, CLASS_NAMES, n_concepts=BLOCKS)
    assert len(concepts) == BLOCKS
    assert isinstance(concepts[0], HebbianConcept)
    covered = sorted(u for c in concepts for u in c.units)
    assert covered == list(range(U_LIVE))
    for c in concepts:
        assert len({_block_of(u) for u in c.units}) == 1
        assert len(c.units) == BLOCK
        # channels are the original channel ids, not the tracked indices
        assert c.channels == [int(stats.unit_index[u]) for u in c.units]
        # a block fires for exactly one class, which reads as affinity 1.0
        block = _block_of(c.units[0])
        assert list(c.class_affinity) == [CLASS_NAMES[block]]
        assert c.class_affinity[CLASS_NAMES[block]] == pytest.approx(1.0)
        # measured, un-normalized per-class scores are kept alongside
        assert len(c.class_scores) == BLOCKS
        assert np.argmax(c.class_scores) == block
        assert c.coherence > 0.5
        assert c.importance > 0.0


def test_concepts_are_unnamed_and_ordered(stats):
    concepts = cluster_concepts(stats, CLASS_NAMES, n_concepts=BLOCKS)
    # honesty rule: no human-language label is invented anywhere.
    assert not hasattr(concepts[0], "label")
    assert "label" not in concepts[0].to_dict()
    # ids are 0..P-1 in descending-importance order.
    assert [c.concept_id for c in concepts] == list(range(len(concepts)))
    importances = [c.importance for c in concepts]
    assert importances == sorted(importances, reverse=True)


def test_concepts_subdivide_but_never_span_blocks(stats):
    """The default n_concepts=16 splits the 4 blocks further; no concept may
    mix units from two different planted blocks."""
    concepts = cluster_concepts(stats, CLASS_NAMES)
    assert len(concepts) > BLOCKS
    for c in concepts:
        assert len({_block_of(u) for u in c.units}) == 1


def test_class_names_length_is_validated(stats):
    with pytest.raises(ValueError):
        cluster_concepts(stats, ["only", "two"])


def test_class_names_optional(stats):
    concepts = cluster_concepts(stats, None, n_concepts=BLOCKS)
    assert all(k.startswith("class_") for c in concepts for k in c.class_affinity)


# --------------------------------------------------------------------------- #
# dead-unit exclusion
# --------------------------------------------------------------------------- #


def test_dead_units_are_detected(stats_with_dead):
    dead = stats_with_dead.dead_units(DEFAULT_ACTIVITY_THRESHOLD)
    assert sorted(dead.tolist()) == list(range(U_LIVE, U_LIVE + N_DEAD))


def test_dead_units_excluded_from_concepts(stats_with_dead):
    concepts = cluster_concepts(stats_with_dead, CLASS_NAMES, n_concepts=BLOCKS)
    clustered = {u for c in concepts for u in c.units}
    assert clustered == set(range(U_LIVE))
    assert not clustered & set(range(U_LIVE, U_LIVE + N_DEAD))
    # exclusion happens before clustering, so the live blocks stay clean.
    for c in concepts:
        assert len({_block_of(u) for u in c.units}) == 1


def test_dead_units_are_flagged_not_hidden(stats_with_dead):
    asset = build_hebbian_graph_asset(
        stats_with_dead, CLASS_NAMES, n_concepts=BLOCKS, seed=0
    )
    assert asset["num_units"] == U_LIVE + N_DEAD
    assert len(asset["nodes"]) == U_LIVE + N_DEAD  # still emitted
    assert asset["dead_units"] == list(range(U_LIVE, U_LIVE + N_DEAD))
    for n in asset["nodes"]:
        assert n["dead"] is (n["idx"] >= U_LIVE)
        assert (n["concept"] == -1) is (n["idx"] >= U_LIVE)


def test_degenerate_all_dead_memory_keeps_every_unit():
    """If nothing meaningfully fired, clustering falls back to all units rather
    than emitting an empty graph."""
    u = 8
    rng = np.random.default_rng(1)
    acts = rng.random((40, u)) * 1e-6
    a_hat = acts / np.linalg.norm(acts, axis=1, keepdims=True)
    st = HebbianStats(
        coact=a_hat.T @ a_hat / 40,
        mean_act=a_hat.mean(axis=0),
        class_act=np.zeros((2, u)),
        unit_index=np.arange(u),
    )
    concepts = cluster_concepts(st, ["a", "b"], n_concepts=2)
    assert sum(len(c.units) for c in concepts) == u


# --------------------------------------------------------------------------- #
# ward clustering primitive
# --------------------------------------------------------------------------- #


def test_ward_labels_recovers_separated_groups():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.repeat(centers, 8, axis=0) + 0.05 * rng.standard_normal((24, 2))
    labels = ward_labels(X, 3)
    assert set(labels.tolist()) == {0, 1, 2}
    for g in range(3):
        assert len(set(labels[g * 8 : (g + 1) * 8].tolist())) == 1
    # canonical: first point is always in community 0, and the run is seedless
    assert labels[0] == 0
    assert np.array_equal(labels, ward_labels(X, 3))


def test_ward_labels_degenerate_cases():
    X = np.arange(12, dtype=float).reshape(4, 3)
    assert np.array_equal(ward_labels(X, 1), np.zeros(4, dtype=np.int64))
    assert np.array_equal(ward_labels(X, 9), np.zeros(4, dtype=np.int64))


# --------------------------------------------------------------------------- #
# hebbian_graph.json asset
# --------------------------------------------------------------------------- #


def test_build_asset_structure(stats):
    asset = build_hebbian_graph_asset(stats, CLASS_NAMES, k=K, n_concepts=BLOCKS, seed=0)
    assert asset["provider"] == "hebbian"
    assert asset["layer"] == LAYER
    assert asset["num_units"] == U_LIVE
    assert asset["num_classes"] == BLOCKS
    assert asset["num_concepts"] == BLOCKS
    assert asset["k"] == K
    assert asset["seed"] == 0
    assert asset["num_updates"] == stats.num_updates
    assert asset["class_names"] == CLASS_NAMES

    assert len(asset["nodes"]) == U_LIVE
    assert len(asset["edges"]) == U_LIVE * K
    # compact node fields
    node = asset["nodes"][0]
    assert set(node.keys()) == {
        "idx",
        "kind",
        "channel",
        "community",
        "concept",
        "activation",
        "dead",
    }
    assert node["kind"] == "unit"
    # every node lands in a real community and a real concept here
    assert all(n["community"] >= 0 for n in asset["nodes"])
    assert all(n["concept"] >= 0 for n in asset["nodes"])
    # compact edge triples [src, dst, weight-3dp]
    src, dst, w = asset["edges"][0]
    assert isinstance(src, int) and isinstance(dst, int)
    assert round(w, 3) == w
    assert 0.0 <= w <= 1.0

    # concept / class / linkage payloads
    assert len(asset["classes"]) == BLOCKS
    assert [c["name"] for c in asset["classes"]] == CLASS_NAMES
    assert len(asset["membership"]) == U_LIVE  # every live unit joins one concept
    assert all(len(m) == 2 for m in asset["membership"])
    assert len(asset["affinity"]) == BLOCKS    # one class edge per planted concept
    for cid, class_idx, weight in asset["affinity"]:
        assert 0 <= class_idx < BLOCKS
        assert 0 < weight <= 1.0
        assert round(weight, 3) == weight
    assert sum(len(c) for c in asset["communities"]) == U_LIVE
    assert asset["id_convention"]["unit"] == "u:{layer}:{idx}"
    assert "co-activation" in asset["edge_semantics"]
    assert asset["community_method"]["method"] == "louvain"


def test_build_asset_concepts_carry_only_measurements(stats):
    asset = build_hebbian_graph_asset(stats, CLASS_NAMES, n_concepts=BLOCKS)
    for c in asset["concepts"]:
        assert set(c.keys()) == {
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
        assert "label" not in c and "name" not in c
        assert len(c["class_scores"]) == BLOCKS


def test_build_asset_deterministic(stats):
    a = build_hebbian_graph_asset(stats, CLASS_NAMES, k=K, n_concepts=BLOCKS, seed=0)
    b = build_hebbian_graph_asset(stats, CLASS_NAMES, k=K, n_concepts=BLOCKS, seed=0)
    assert a == b
    # byte-identical once serialized, which is what actually ships.
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_build_asset_json_serializable(stats):
    asset = build_hebbian_graph_asset(stats, CLASS_NAMES, n_concepts=BLOCKS)
    reparsed = json.loads(json.dumps(asset))  # no numpy scalars may leak
    assert reparsed["num_units"] == U_LIVE


def test_build_asset_records_provenance(stats):
    asset = build_hebbian_graph_asset(
        stats, CLASS_NAMES, provenance={"dataset": "ham10000", "epoch": 30}
    )
    assert asset["provenance"]["dataset"] == "ham10000"


def test_asset_agrees_with_provider(stats):
    p = HebbianGraphProvider(stats, k=K, seed=0, class_names=CLASS_NAMES, n_concepts=BLOCKS)
    asset = build_hebbian_graph_asset(stats, CLASS_NAMES, k=K, n_concepts=BLOCKS, seed=0)
    assert len(asset["edges"]) == len(p.edges())
    assert len(asset["concepts"]) == len(p.concepts())
    assert len(asset["communities"]) == len(p.communities())
    # membership + affinity edges of the provider match the asset's link lists
    concept_edges = p.concept_edges()
    assert len(concept_edges) == len(asset["membership"]) + len(asset["affinity"])


# --------------------------------------------------------------------------- #
# import purity (the M0 rule) + the lazy torch recorder
# --------------------------------------------------------------------------- #


def test_import_does_not_pull_in_torch():
    """`import vitreous.hebbian` must not import torch, even if it is installed.

    Run in a subprocess so a torch already imported by another test cannot mask
    a regression — the same pattern test_imports.py uses.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import vitreous.hebbian as hebbian
        assert "torch" not in sys.modules, "torch was imported at import time"
        assert "networkx" not in sys.modules, "networkx was imported at import time"
        # the numpy-only surface is fully usable without torch
        assert callable(hebbian.build_hebbian_graph_asset)
        assert hebbian.HebbianStats is not None
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_recorder_is_lazy():
    """The recorder is exposed through module __getattr__ (torch on first use)."""
    import vitreous.hebbian as hebbian

    assert "HebbianRecorder" not in hebbian.__all__  # documented as lazy
    with pytest.raises(AttributeError):
        hebbian.NoSuchSymbol  # noqa: B018

    pytest.importorskip("torch")
    assert hebbian.HebbianRecorder.__name__ == "HebbianRecorder"


def test_recorder_observes_without_changing_the_model():
    """Attaching the recorder must leave forward outputs bit-identical."""
    torch = pytest.importorskip("torch")
    from vitreous.hebbian import HebbianRecorder

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 12), torch.nn.ReLU())
    x = torch.randn(8, 16)
    baseline = model(x).clone()

    rec = HebbianRecorder(
        num_classes=3, layers={"fc": model[0]}, max_units=8, momentum=0.5, seed=0
    )
    try:
        rec.observe_labels(torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
        observed = model(x)
        assert torch.equal(observed, baseline)  # pure observation
        assert observed.grad_fn is not None     # graph untouched

        st = rec.stats()
        assert st.num_units == 8                # max_units subsampling applied
        assert st.num_classes == 3
        assert st.coact.shape == (8, 8)
        assert st.num_updates == 1
        assert st.layer == "fc"
        assert st.class_count.sum() == 8
        # round-trips like any other stats object
        assert HebbianStats.from_dict(st.to_dict()).num_units == 8
    finally:
        rec.detach()
