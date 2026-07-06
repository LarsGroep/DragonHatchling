"""ViTTokenGraphProvider + build_graph_asset (§8) — offline, synthetic.

Synthetic row-stochastic attention; no model. Covers node counts/kinds, exact
top-k edge counts and weight fidelity, Louvain determinism + full partition, and
the graph.json structure incl. the implicit residual convention.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vitreous.graph import (
    GraphProvider,
    ViTTokenGraphProvider,
    build_graph_asset,
    node_id,
)

L, H, T = 12, 6, 197
K = 8


def _synthetic_trace(seed: int = 0):
    rng = np.random.default_rng(seed)
    raw = rng.random((L, H, T, T)).astype(np.float32)
    attn = raw / raw.sum(axis=-1, keepdims=True)
    return SimpleNamespace(attention=attn)


@pytest.fixture(scope="module")
def trace():
    return _synthetic_trace(0)


@pytest.fixture(scope="module")
def provider():
    return ViTTokenGraphProvider(k=K, seed=0)


def test_provider_satisfies_protocol(provider):
    assert isinstance(provider, GraphProvider)


def test_node_count_and_kinds(provider, trace):
    nodes = provider.nodes(trace)
    assert len(nodes) == T * L                      # 197 * 12 = 2364
    # 197 nodes per layer, one CLS per layer.
    assert sum(1 for n in nodes if n.kind == "cls_token") == L
    assert sum(1 for n in nodes if n.kind == "patch_token") == (T - 1) * L
    # CLS is always token 0.
    for n in nodes:
        assert n.kind == ("cls_token" if n.ref["token"] == 0 else "patch_token")
    # ids are unique and layer-scoped.
    assert len({n.id for n in nodes}) == T * L


def test_topk_edge_count_per_layer(provider, trace):
    for layer in range(L):
        edges = provider.edges(trace, layer)
        assert len(edges) == T * K                  # 197 * 8 = 1576
        # exactly K edges per destination token.
        by_dst: dict = {}
        for e in edges:
            by_dst[e.target] = by_dst.get(e.target, 0) + 1
        assert len(by_dst) == T
        assert all(c == K for c in by_dst.values())


def test_edge_weights_match_trace_attention(provider, trace):
    layer = 3
    abar = trace.attention[layer].mean(axis=0)      # [T,T], row = destination/query
    for e in provider.edges(trace, layer)[:200]:
        src = int(e.source.split("T")[1])
        dst = int(e.target.split("T")[1])
        assert abs(e.weight - float(abar[dst, src])) < 1e-6


def test_topk_are_the_actual_topk(provider, trace):
    layer = 0
    abar = trace.attention[layer].mean(axis=0)
    edges = provider.edges(trace, layer)
    # For destination 0, the kept sources must equal the true top-k keys.
    kept = sorted(int(e.source.split("T")[1]) for e in edges if e.target == node_id(0, 0))
    true_top = sorted(int(j) for j in np.argsort(-abar[0], kind="stable")[:K])
    assert kept == true_top


def test_communities_partition_all_nodes(provider, trace):
    for layer in (0, 5, 11):
        comms = provider.communities(trace, layer)
        members = [m for c in comms for m in c.members]
        assert len(members) == T                    # covers every token
        assert len(set(members)) == T               # disjoint partition
        assert all(c.layer == layer for c in comms)


def test_louvain_determinism(trace):
    p1 = ViTTokenGraphProvider(k=K, seed=42)
    p2 = ViTTokenGraphProvider(k=K, seed=42)
    a = [c.members for c in p1.communities(trace, 4)]
    b = [c.members for c in p2.communities(trace, 4)]
    assert a == b


def test_build_graph_asset_structure(trace):
    ga = build_graph_asset(trace, k=K, seed=0)
    assert ga["num_layers"] == L
    assert ga["num_tokens"] == T
    assert ga["k"] == K
    assert ga["cls_index"] == 0
    assert len(ga["layers"]) == L
    layer0 = ga["layers"][0]
    assert len(layer0["nodes"]) == T
    assert len(layer0["edges"]) == T * K
    # compact node fields
    node = layer0["nodes"][0]
    assert set(node.keys()) == {"idx", "kind", "community"}
    assert node["kind"] == "cls_token"
    # every node has a real community (no unassigned -1)
    assert all(n["community"] >= 0 for n in layer0["nodes"])
    # compact edge triples [src, dst, weight-3dp]
    src, dst, w = layer0["edges"][0]
    assert isinstance(src, int) and isinstance(dst, int)
    assert round(w, 3) == w


def test_build_graph_asset_residual_is_implicit(trace):
    ga = build_graph_asset(trace, k=K, seed=0)
    res = ga["residual"]
    assert res["materialized"] is False
    # implicit residual edges = num_tokens * (num_layers - 1)
    assert res["count"] == T * (L - 1)
    assert "description" in res and "residual" in res["description"].lower()


def test_build_graph_asset_deterministic(trace):
    a = build_graph_asset(trace, k=K, seed=0)
    b = build_graph_asset(trace, k=K, seed=0)
    assert a == b


def test_configurable_k(trace):
    p = ViTTokenGraphProvider(k=4, seed=0)
    assert len(p.edges(trace, 0)) == T * 4
