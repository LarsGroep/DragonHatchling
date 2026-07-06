"""Interaction Graph provider abstraction + ViT token provider (§8).

The :class:`GraphProvider` Protocol is what keeps the frontend model-agnostic:
v1 ships :class:`ViTTokenGraphProvider` (default); the interface admits CNNs,
sparse-unit models, or diffusion models later without touching the frontend.

Default validated semantics (not speculative): nodes = tokens (CLS + 196
patches) per attention layer; edges = **top-k (k=8) attention weights** per
destination token from that layer's head-averaged attention; Louvain
communities over the layer's undirected weighted attention graph. Everything is
precomputed into ``graph.json`` by :func:`build_graph_asset`.

``graph.json`` structure
------------------------
::

    {
      "num_layers": 12, "num_tokens": 197, "k": 8, "grid": 14, "cls_index": 0,
      "residual": { ... },          # convention for unrolled residual edges
      "layers": [
        {
          "layer": 0,
          "nodes": [ {"idx": 0, "kind": "cls_token", "community": 3}, ... ],   # 197
          "edges": [ [src_idx, dst_idx, weight], ... ]                          # 197*k
        },
        ...  # one entry per attention layer
      ]
    }

Edges are compact ``[src_idx, dst_idx, weight]`` triples where ``src``/``dst``
are token indices *within the layer* (0..196) and ``weight`` is the
head-averaged attention value quantized to 3 decimals. Each edge points from the
attended **key** token (src) to the **destination/query** token (dst): for each
destination token the top-k keys by attention are kept (self-attention edges are
allowed), so there are exactly ``num_tokens * k`` edges per layer.

Unrolled residual edges are **not materialized**. Instead the ``residual`` flag
describes the convention so the frontend can synthesize them: for each token
index ``i`` and each consecutive layer pair ``(t, t+1)``, an identity edge links
``(layer t, token i)`` to ``(layer t+1, token i)`` — that is
``num_tokens * (num_layers - 1)`` edges, avoided on disk.

Determinism: Louvain is seeded (``seed``), so identical input yields identical
communities. NetworkX's built-in ``louvain_communities`` is used (no extra
dependency beyond networkx).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Protocol, Tuple, runtime_checkable

import numpy as np

NodeKind = Literal[
    "patch_token",
    "cls_token",
    "attention_head",
    "concept",
    "community",
    "unit",
]

DEFAULT_K = 8
DEFAULT_SEED = 0
CLS_INDEX = 0
GRID = 14


@dataclass
class GraphNode:
    """A node in the interaction graph (§8).

    ``ref`` points into pack assets (token idx, Gaussian idx, concept id) so the
    frontend resolver can link a node to every other view in O(1).
    """

    id: str
    kind: NodeKind
    layer: int
    ref: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A directed, weighted edge between two graph nodes (§8)."""

    source: str
    target: str
    weight: float
    layer: int


@dataclass
class Community:
    """A detected community (Louvain over a layer's attention graph)."""

    id: str
    layer: int
    members: List[str] = field(default_factory=list)


@runtime_checkable
class GraphProvider(Protocol):
    """Backend abstraction that turns a trace into graph structure (§8)."""

    def nodes(self, trace: Any) -> List[GraphNode]:
        ...

    def edges(self, trace: Any, layer: int) -> List[GraphEdge]:
        ...

    def communities(self, trace: Any, layer: int) -> List[Community]:
        ...


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _to_numpy(x: Any) -> np.ndarray:
    """Detach + numpy-ify a torch tensor or pass numpy through (no torch import)."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    return np.asarray(x)


def _head_avg(attn: np.ndarray, layer: int) -> np.ndarray:
    """Head-averaged attention ``[T, T]`` for ``layer`` (rows = query/destination)."""
    if attn.ndim != 4:
        raise ValueError(f"expected attention [L,H,T,T], got {attn.shape}")
    return attn[layer].mean(axis=0).astype(np.float64)


def node_id(layer: int, idx: int) -> str:
    """Canonical per-layer node id, e.g. ``L3_T17``."""
    return f"L{layer}_T{idx}"


def _kind(idx: int) -> NodeKind:
    return "cls_token" if idx == CLS_INDEX else "patch_token"


# --------------------------------------------------------------------------- #
# ViTTokenGraphProvider
# --------------------------------------------------------------------------- #


class ViTTokenGraphProvider:
    """Default GraphProvider: tokens as nodes, top-k attention as edges (§8)."""

    def __init__(self, k: int = DEFAULT_K, seed: int = DEFAULT_SEED) -> None:
        self.k = int(k)
        self.seed = int(seed)

    # -- shapes -------------------------------------------------------------- #

    @staticmethod
    def _dims(trace: Any) -> Tuple[int, int]:
        attn = _to_numpy(trace.attention)
        L, _H, T, _ = attn.shape
        return int(L), int(T)

    # -- nodes --------------------------------------------------------------- #

    def nodes(self, trace: Any) -> List[GraphNode]:
        """All ``num_tokens * num_layers`` token nodes (one copy per layer)."""
        L, T = self._dims(trace)
        out: List[GraphNode] = []
        for layer in range(L):
            for i in range(T):
                out.append(
                    GraphNode(
                        id=node_id(layer, i),
                        kind=_kind(i),
                        layer=layer,
                        ref={"token": i, "layer": layer},
                    )
                )
        return out

    # -- edges --------------------------------------------------------------- #

    def edges(self, trace: Any, layer: int) -> List[GraphEdge]:
        """Top-k (per destination token) attention edges at ``layer``.

        For each destination (query) token ``i`` the ``k`` highest-attention keys
        ``j`` are kept as edges ``key -> query`` with weight ``a_ij``. Exactly
        ``num_tokens * k`` edges.
        """
        attn = _to_numpy(trace.attention)
        a = _head_avg(attn, layer)                    # [T, T], rows = destination
        T = a.shape[0]
        k = min(self.k, T)
        out: List[GraphEdge] = []
        for i in range(T):                            # destination / query
            row = a[i]
            # top-k keys by weight; deterministic tie-break by index.
            top = np.argsort(-row, kind="stable")[:k]
            for j in top:
                out.append(
                    GraphEdge(
                        source=node_id(layer, int(j)),   # attended key
                        target=node_id(layer, i),        # destination
                        weight=float(row[int(j)]),
                        layer=layer,
                    )
                )
        return out

    # -- communities --------------------------------------------------------- #

    def _partition(self, trace: Any, layer: int) -> List[List[int]]:
        """Louvain partition (list of token-index lists) for ``layer``, seeded."""
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        attn = _to_numpy(trace.attention)
        a = _head_avg(attn, layer)
        T = a.shape[0]
        # Undirected weighted graph: symmetrize, drop the diagonal.
        w = (a + a.T) * 0.5
        np.fill_diagonal(w, 0.0)
        g = nx.Graph()
        g.add_nodes_from(range(T))
        iu = np.triu_indices(T, k=1)
        for i, j, weight in zip(iu[0].tolist(), iu[1].tolist(), w[iu].tolist()):
            if weight > 0.0:
                g.add_edge(i, j, weight=float(weight))
        parts = louvain_communities(g, weight="weight", seed=self.seed)
        # Deterministic ordering: sort communities by their smallest member.
        return [sorted(int(x) for x in c) for c in sorted(parts, key=lambda c: min(c))]

    def communities(self, trace: Any, layer: int) -> List[Community]:
        """Louvain communities over the layer's undirected attention graph."""
        parts = self._partition(trace, layer)
        return [
            Community(
                id=f"L{layer}_C{ci}",
                layer=layer,
                members=[node_id(layer, i) for i in members],
            )
            for ci, members in enumerate(parts)
        ]


# --------------------------------------------------------------------------- #
# build_graph_asset
# --------------------------------------------------------------------------- #


def build_graph_asset(
    trace: Any,
    *,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Build the compact ``graph.json`` structure for all attention layers (§8)."""
    provider = ViTTokenGraphProvider(k=k, seed=seed)
    L, T = provider._dims(trace)
    attn = _to_numpy(trace.attention)

    layers: List[Dict[str, Any]] = []
    for layer in range(L):
        a = _head_avg(attn, layer)
        # community index per token
        parts = provider._partition(trace, layer)
        comm_of = {i: ci for ci, members in enumerate(parts) for i in members}

        nodes = [
            {"idx": i, "kind": _kind(i), "community": int(comm_of.get(i, -1))}
            for i in range(T)
        ]

        kk = min(k, T)
        edges: List[List[Any]] = []
        for i in range(T):                        # destination / query
            row = a[i]
            top = np.argsort(-row, kind="stable")[:kk]
            for j in top:
                edges.append([int(j), i, round(float(row[int(j)]), 3)])

        layers.append({"layer": layer, "nodes": nodes, "edges": edges})

    return {
        "num_layers": int(L),
        "num_tokens": int(T),
        "k": int(k),
        "grid": GRID,
        "cls_index": CLS_INDEX,
        "seed": int(seed),
        "edge_semantics": "key->query; per destination token top-k head-averaged attention; weight rounded to 3 decimals",
        "residual": {
            "kind": "identity",
            "materialized": False,
            "weight": 1.0,
            "count": int(T * (L - 1)) if L > 1 else 0,
            "description": (
                "Unrolled residual-stream edges are implicit. For each token "
                "index i in 0..num_tokens-1 and each consecutive layer pair "
                "(t, t+1) with t in 0..num_layers-2, synthesize edge "
                "(layer t, token i) -> (layer t+1, token i). Not stored on disk."
            ),
        },
        "layers": layers,
    }


__all__ = [
    "NodeKind",
    "GraphNode",
    "GraphEdge",
    "Community",
    "GraphProvider",
    "ViTTokenGraphProvider",
    "build_graph_asset",
    "node_id",
]
