"""Hierarchical concept tree over the Hebbian co-activation structure.

:mod:`hatchvision.explain.concepts` produces a *flat* partition of the
tracked units into concepts.  This module instead reads the whole Ward
dendrogram and keeps it as a **tree**: a root spanning every active unit,
splitting into progressively more specific sub-concepts.  The tree is the
scaffold for a decision-tree-style classifier (see
:mod:`hatchvision.hebbian.heads`): routing an image from the root to a leaf
is classification, and every node — coarse or fine — is a nameable concept
that can later be shown as image pixels rather than text.

The construction mirrors :func:`cluster_concepts` exactly where it matters
(dead-unit filtering, correlation-row fingerprints, Ward linkage, per-node
max-normalized class affinity) so flat and hierarchical views stay
consistent; the only difference is that we descend the dendrogram instead of
cutting it once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hatchvision.hebbian.memory import HebbianFeatureMemory


@dataclass
class ConceptNode:
    """A node of the concept hierarchy.

    Parameters mirror :class:`~hatchvision.explain.concepts.Concept` but add
    ``depth`` and ``children`` so the whole tree round-trips through JSON.
    ``units`` are indices into the memory's *tracked* units (``unit_index``),
    the same coordinate system concept clustering and the ONNX bundle use.
    """

    node_id: str
    layer: str
    units: List[int]
    depth: int
    coherence: float                       # mean intra-cluster correlation
    importance: float                      # mean firing rate of member units
    class_affinity: Dict[str, float]       # class name -> max-normalized score
    children: List["ConceptNode"] = field(default_factory=list)
    label: Optional[str] = None            # optional text (top classes)
    patches: List[str] = field(default_factory=list)  # pixel identity data URIs

    # ------------------------------------------------------------ convenience

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def walk(self):
        """Yield this node and every descendant (pre-order)."""
        yield self
        for child in self.children:
            yield from child.walk()

    def leaves(self) -> List["ConceptNode"]:
        return [n for n in self.walk() if n.is_leaf]

    # -------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        d = {
            "node_id": self.node_id,
            "layer": self.layer,
            "units": list(self.units),
            "depth": self.depth,
            "coherence": round(float(self.coherence), 6),
            "importance": round(float(self.importance), 6),
            "class_affinity": {k: round(float(v), 6) for k, v in self.class_affinity.items()},
            "children": [c.to_dict() for c in self.children],
        }
        if self.label is not None:
            d["label"] = self.label
        if self.patches:
            d["patches"] = list(self.patches)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptNode":
        return cls(
            node_id=d["node_id"],
            layer=d["layer"],
            units=list(d["units"]),
            depth=int(d["depth"]),
            coherence=float(d["coherence"]),
            importance=float(d["importance"]),
            class_affinity={k: float(v) for k, v in d.get("class_affinity", {}).items()},
            children=[cls.from_dict(c) for c in d.get("children", [])],
            label=d.get("label"),
            patches=list(d.get("patches", [])),
        )


def _coherence(corr: np.ndarray, units: Sequence[int]) -> float:
    """Mean off-diagonal correlation among ``units`` (1.0 for a singleton)."""
    if len(units) <= 1:
        return 1.0
    sub = corr[np.ix_(units, units)]
    off = sub[~np.eye(len(units), dtype=bool)]
    return float(off.mean()) if off.size else 1.0


def _affinity_dict(
    affinity: np.ndarray,
    units: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, float]:
    """Per-node class affinity, max-normalized so the top class scores 1.0.

    Same normalization idea as :func:`cluster_concepts`; here we keep *every*
    class (pruning only exact zeros) so the tree-routed head has a full class
    distribution at each leaf.
    """
    cls_scores = affinity[:, units].mean(axis=1)
    peak = cls_scores.max()
    norm = cls_scores / peak if peak > 0 else cls_scores
    return {class_names[i]: float(norm[i]) for i in range(len(class_names)) if norm[i] > 0}


def _label(class_affinity: Dict[str, float]) -> str:
    top = sorted(class_affinity.items(), key=lambda kv: -kv[1])[:2]
    return " / ".join(name for name, _ in top) or "concept"


def build_concept_tree(
    memory: HebbianFeatureMemory,
    layer: str,
    class_names: Sequence[str],
    max_depth: int = 3,
    min_units: int = 2,
    activity_threshold: float = 0.02,
    min_coherence_gain: float = 0.0,
) -> ConceptNode:
    """Build a concept hierarchy from the Ward dendrogram of a layer.

    Dead/near-dead units are filtered exactly as in :func:`cluster_concepts`,
    the correlation-row fingerprints are Ward-linked **once**, and the
    resulting dendrogram is walked top-down: the root spans all active units,
    and each internal node is split into its two dendrogram children.  A split
    is *rejected* (the node becomes a leaf) when

    * ``depth == max_depth``,
    * either child would have fewer than ``min_units`` units, or
    * the mean of the two children's coherence does not exceed the parent's
      coherence by at least ``min_coherence_gain``.

    The children therefore always partition the parent's units, coherence
    tends to rise with depth, and single-child chains are collapsed.
    """
    from scipy.cluster.hierarchy import linkage

    corr = memory.correlation(layer).numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    mean_act = memory.stats[layer].mean_act.numpy()
    affinity = memory.class_affinity(layer).numpy()

    active = np.where(mean_act > activity_threshold * max(mean_act.max(), 1e-12))[0]
    if len(active) < max(min_units * 2, 4):     # degenerate memory; keep all
        active = np.arange(corr.shape[0])
    active = np.asarray(active)
    n = len(active)

    counter = {"i": 0}

    def new_id() -> str:
        nid = f"n{counter['i']}"
        counter["i"] += 1
        return nid

    def make_leaf_node(unit_positions: Sequence[int], depth: int) -> ConceptNode:
        units = [int(active[p]) for p in unit_positions]
        aff = _affinity_dict(affinity, units, class_names)
        node = ConceptNode(
            node_id=new_id(),
            layer=layer,
            units=units,
            depth=depth,
            coherence=_coherence(corr, units),
            importance=float(mean_act[units].mean()),
            class_affinity=aff,
            label=_label(aff),
        )
        return node

    if n <= min_units or n < 2:
        # Nothing to split — a single concept over all active units.
        return make_leaf_node(list(range(n)), 0)

    fingerprints = corr[np.ix_(active, active)]
    Z = linkage(fingerprints, method="ward")

    # Memoized leaf-position sets per linkage cluster id (0..n-1 are leaves).
    cache: Dict[int, List[int]] = {}

    def leaves_of(cid: int) -> List[int]:
        if cid < n:
            return [cid]
        if cid in cache:
            return cache[cid]
        left, right = int(Z[cid - n, 0]), int(Z[cid - n, 1])
        res = leaves_of(left) + leaves_of(right)
        cache[cid] = res
        return res

    def make_node(cid: int, depth: int) -> ConceptNode:
        positions = leaves_of(cid)
        node = make_leaf_node(positions, depth)
        if depth >= max_depth or cid < n:
            return node
        left, right = int(Z[cid - n, 0]), int(Z[cid - n, 1])
        lpos, rpos = leaves_of(left), leaves_of(right)
        if len(lpos) < min_units or len(rpos) < min_units:
            return node
        lcoh = _coherence(corr, [int(active[p]) for p in lpos])
        rcoh = _coherence(corr, [int(active[p]) for p in rpos])
        if 0.5 * (lcoh + rcoh) - node.coherence < min_coherence_gain:
            return node
        node.children = [make_node(left, depth + 1), make_node(right, depth + 1)]
        return node

    root = make_node(2 * n - 2, 0)   # last merge = whole tree
    return _collapse_single_children(root)


def _collapse_single_children(node: ConceptNode) -> ConceptNode:
    """Splice out any node that has exactly one child (defensive)."""
    while len(node.children) == 1:
        child = node.children[0]
        child.depth = node.depth
        node = child
    node.children = [_collapse_single_children(c) for c in node.children]
    return node


def node_scores(
    tree: ConceptNode,
    acts: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Per-image activation of every node in the tree.

    ``acts`` is the pooled/rectified activation dict produced by
    :func:`~hatchvision.explain.concepts.probe_activations` (tracked units
    only).  Each image's activation vector is L2-normalized first — the same
    ``a / (||a|| + 1e-8)`` the memory applies — so a node's score (mean
    activation of its member units) is comparable across images regardless of
    overall firing magnitude.  Returns ``{node_id: [n_images]}``.
    """
    a = acts[tree.layer].float()
    a_hat = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    scores: Dict[str, torch.Tensor] = {}
    for node in tree.walk():
        if node.units:
            scores[node.node_id] = a_hat[:, node.units].mean(dim=1)
        else:
            scores[node.node_id] = torch.zeros(a_hat.shape[0])
    return scores
