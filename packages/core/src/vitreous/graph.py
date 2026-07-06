"""Interaction Graph provider abstraction (§8).

The :class:`GraphProvider` Protocol is what keeps the frontend model-agnostic:
v1 ships ``ViTTokenGraphProvider`` (default) and ``ConceptGraphProvider`` at M3,
but the interface admits CNNs, sparse-unit models, or diffusion models later
without touching the frontend.

Default validated semantics: nodes = tokens (+ CLS, + optional per-layer head
rings); edges = top-k (k=8) attention weights at layer *t*; Louvain communities
per layer. M0 ships the dataclasses and the Protocol; providers land at M3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Protocol, runtime_checkable

NodeKind = Literal[
    "patch_token",
    "cls_token",
    "attention_head",
    "concept",
    "community",
    "unit",
]


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


__all__ = [
    "NodeKind",
    "GraphNode",
    "GraphEdge",
    "Community",
    "GraphProvider",
]
