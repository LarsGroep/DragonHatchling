"""Hebbian co-activation graph — a second :class:`~vitreous.graph.GraphProvider`.

``vitreous.graph``'s module docstring promises that the provider interface
"admits CNNs, sparse-unit models, or diffusion models later without touching the
frontend". This module is the first redemption of that promise: it renders a
model's **Hebbian co-activation structure** (units that fire together) as the
same node/edge/community triple the ViT token provider emits, so the workbench
can draw a real Hebbian brain instead of an attention map.

Ported from the legacy ``hatchvision`` package (``hebbian/memory.py``,
``explain/concepts.py``, ``export/ivgraph.py``), whose exported HAM10000 graph
(``webapp/graph.json``: 322 unit + 16 concept + 7 class nodes) is the shape this
module reproduces.

What is measured
----------------
"Neurons that fire together, wire together." A :class:`HebbianRecorder` attaches
**observation-only** forward hooks to a live model and keeps, per observed layer,
an exponential moving average of the outer product ``a aᵀ`` of the *spatially
pooled, rectified, L2-normalized* activations, plus the mean activation vector
and per-class conditional firing rates. Everything is detached inside
``torch.no_grad()``; hooks return ``None`` and never touch activations or
gradients, so attaching a recorder leaves optimization bit-identical.

The recorded numbers are frozen into a plain numpy :class:`HebbianStats`, and
:class:`HebbianGraphProvider` turns *that* — never a live model — into graph
structure. So a graph can be rebuilt, re-clustered and re-exported from saved
statistics without retraining (the legacy ``hebbian_state.pt`` workflow, but
JSON/npz instead of a torch pickle).

Derived quantities, and where each comes from
---------------------------------------------
``correlation``
    ``coact_ij / sqrt(coact_ii · coact_jj)`` — cosine-normalized co-activation.
    Measured; the normalization is the only transform.
``edges``
    Per unit, its ``k`` strongest co-activation partners (self excluded),
    weight = the correlation above, quantized to 3 decimals exactly like
    :func:`vitreous.graph.build_graph_asset`.
``communities``
    Louvain (``networkx``, seeded) over the symmetrized correlation matrix with
    the diagonal removed — the same recipe as ``graph._partition``.
``concepts``
    Ward agglomeration over each *live* unit's co-activation fingerprint (its
    row of the correlation matrix). Dead / near-dead units — mean activation
    below ``activity_threshold`` × the most active unit — are dropped first:
    with wide sparse neuron spaces they otherwise dominate the geometry and
    everything alive chains into one giant cluster.
``coherence`` / ``importance``
    A concept's mean off-diagonal intra-cluster correlation / the mean measured
    activation of its member units.
``class affinity``
    A concept's mean per-class firing rate (from ``class_act``), normalized so
    its strongest class reads 1.0 — the concept→class edges of the bundle.

Honesty rule
------------
Every emitted number traces to a measurement. Concepts carry ids, measured class
affinities, coherence and importance — **no human-language names**: naming a
concept is a separate grounding step and is deliberately not done here. Dead
units are excluded from clustering but still emitted as nodes (flagged), never
hidden.

Import discipline (the M0 rule)
-------------------------------
``import vitreous.hebbian`` must not import torch. Only numpy (+ the torch-free
``vitreous.graph``) is imported eagerly; ``networkx`` is imported inside the
community routine; the torch-dependent :class:`HebbianRecorder` class is built
lazily through the module-level ``__getattr__`` (the same pattern
``vitreous.concepts`` uses for its SAE). Ward clustering is implemented here in
numpy rather than pulled from scikit-learn, keeping the base dependency set
intact (``vitreous.som`` does the same for k-means).

``hebbian_graph.json`` structure
--------------------------------
::

    {
      "provider": "hebbian", "layer": "neurons",
      "num_units": 322, "num_concepts": 16, "num_classes": 7,
      "k": 8, "seed": 0, "num_updates": 4210,
      "class_names": [...],
      "nodes":      [ {"idx", "kind": "unit", "channel", "community",
                       "concept", "activation", "dead"}, ... ],   # num_units
      "edges":      [ [src_idx, dst_idx, weight], ... ],          # num_units * k
      "concepts":   [ {"id", "units", "channels", "coherence", "importance",
                       "class_scores", "class_affinity"}, ... ],
      "classes":    [ {"idx", "kind": "class", "name"}, ... ],
      "membership": [ [unit_idx, concept_id], ... ],
      "affinity":   [ [concept_id, class_idx, weight], ... ],
      "communities":[ [unit_idx, ...], ... ],
      ...semantics/provenance fields
    }

``edges`` are compact ``[src, dst, weight]`` triples in **unit-index space**
(0..num_units-1) pointing partner → unit, mirroring the ViT provider's compact
triples. ``membership``/``affinity`` are kept as separate lists so every list has
homogeneous endpoints; ``id_convention`` records the string ids
(``u:<layer>:<idx>``, ``c:<id>``, ``k:<name>``) the legacy browser bundle uses,
so a frontend can rebuild them without inventing a scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .graph import DEFAULT_K, DEFAULT_SEED, Community, GraphEdge, GraphNode

# --------------------------------------------------------------------------- #
# defaults (recorder + provider)
# --------------------------------------------------------------------------- #

DEFAULT_MAX_UNITS = 256          # wide layers are subsampled to this many units
DEFAULT_MOMENTUM = 0.05          # EMA momentum of the co-activation memory
DEFAULT_N_CONCEPTS = 16          # matches the exported HAM10000 bundle
DEFAULT_MIN_UNITS = 2            # a concept needs at least this many units
DEFAULT_ACTIVITY_THRESHOLD = 0.02  # × max mean activation → dead-unit cutoff
DEFAULT_TOP_CLASSES = 3          # class-affinity edges emitted per concept
WEIGHT_DECIMALS = 3              # edge-weight quantization (as in graph.py)
SCALAR_DECIMALS = 6              # node/concept scalar quantization

#: Node kinds this provider emits. ``"unit"`` and ``"concept"`` are already
#: declared by :data:`vitreous.graph.NodeKind`; ``"class"`` is local to this
#: module (the ViT provider has no class nodes). It is a plain string, which the
#: runtime-checkable ``GraphProvider`` Protocol permits — promoting it into
#: ``graph.NodeKind`` is a one-line change owned by ``graph.py``.
UNIT_KIND = "unit"
CONCEPT_KIND = "concept"
CLASS_KIND = "class"

#: String-id convention of the legacy browser bundle (``webapp/graph.json``).
ID_CONVENTION = {
    "unit": "u:{layer}:{idx}",
    "concept": "c:{id}",
    "class": "k:{name}",
}


def unit_node_id(layer: str, idx: int) -> str:
    """Canonical unit node id, e.g. ``u:neurons:17`` (``idx`` is the tracked
    unit index, not the original channel — ``channel`` carries that)."""
    return f"u:{layer}:{int(idx)}"


def concept_node_id(concept_id: int) -> str:
    """Canonical concept node id, e.g. ``c:3``."""
    return f"c:{int(concept_id)}"


def class_node_id(name: str) -> str:
    """Canonical class node id, e.g. ``k:Melanoma``."""
    return f"k:{name}"


def community_id(layer: str, index: int) -> str:
    """Canonical community id, e.g. ``neurons_C2`` (mirrors ``L3_C2``)."""
    return f"{layer}_C{int(index)}"


# --------------------------------------------------------------------------- #
# numpy helpers (torch-free)
# --------------------------------------------------------------------------- #


def _to_numpy(x: Any) -> np.ndarray:
    """Detach + numpy-ify a torch tensor, or pass numpy/array-like through."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def correlation_matrix(coact: Any) -> np.ndarray:
    """Cosine-normalize a co-activation matrix: ``c_ij / sqrt(c_ii · c_jj)``.

    This is the only transform applied to the measured EMA: it removes each
    unit's overall firing magnitude (which drifts during training) so an edge
    weight reads as "how much of the time do these two fire *together*".
    Rows of never-firing units are ``0/0`` and become exactly ``0``.
    """
    c = np.asarray(coact, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"coact must be square [U, U], got {c.shape}")
    d = np.sqrt(np.maximum(np.diag(c), 1e-12))
    out = c / (d[:, None] * d[None, :])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _canonical_relabel(labels: np.ndarray) -> np.ndarray:
    """Relabel so cluster ids appear in ascending first-occurrence order."""
    labels = np.asarray(labels, dtype=np.int64)
    remap: Dict[int, int] = {}
    out = np.empty_like(labels)
    nxt = 0
    for i, lab in enumerate(labels.tolist()):
        if lab not in remap:
            remap[lab] = nxt
            nxt += 1
        out[i] = remap[lab]
    return out


def ward_labels(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Ward agglomerative clustering of ``X`` ``[N, D]`` → labels ``[N]``.

    A local, numpy-only implementation (no scikit-learn / scipy — ``packages``
    ``/core`` keeps its base dependency set; ``vitreous.som._kmeans`` sets the
    same precedent). Greedy merges of the globally closest pair under the
    Lance–Williams recurrence for **Ward linkage on squared Euclidean
    distances**::

        d²(i∪j, k) = [(n_i+n_k)·d²(i,k) + (n_j+n_k)·d²(j,k) − n_k·d²(i,j)]
                     / (n_i+n_j+n_k)

    Ward's criterion is reducible, so greedy global merging gives the same
    dendrogram as the nearest-neighbour-chain algorithms used by scipy/sklearn.
    Ties are broken deterministically towards the lexicographically smallest
    ``(i, j)``, and labels are canonically relabelled by first occurrence, so
    the result depends only on ``X`` — no seed, no run-to-run drift.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be [N, D], got {X.shape}")
    n = X.shape[0]
    n_clusters = int(n_clusters)
    if n_clusters <= 1 or n <= n_clusters:
        return np.zeros(n, dtype=np.int64)

    sq = (X**2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)

    sizes = np.ones(n, dtype=np.float64)
    members: List[List[int]] = [[i] for i in range(n)]
    n_active = n

    while n_active > n_clusters:
        # argmin scans row-major, so among equal minima the smallest (i, j)
        # wins — the deterministic tie-break.
        i, j = divmod(int(np.argmin(d2)), n)
        if i > j:
            i, j = j, i
        ni, nj = sizes[i], sizes[j]
        nk = sizes
        merged = ((ni + nk) * d2[i] + (nj + nk) * d2[j] - nk * d2[i, j]) / (
            ni + nj + nk
        )
        merged[i] = np.inf
        merged[j] = np.inf
        d2[i, :] = merged
        d2[:, i] = merged
        d2[j, :] = np.inf
        d2[:, j] = np.inf
        sizes[i] = ni + nj
        members[i] = members[i] + members[j]
        members[j] = []
        n_active -= 1

    labels = np.empty(n, dtype=np.int64)
    survivors = sorted((m for m in members if m), key=min)
    for cid, group in enumerate(survivors):
        for idx in group:
            labels[idx] = cid
    return _canonical_relabel(labels)


def _resolve_class_names(
    class_names: Optional[Sequence[str]], num_classes: int
) -> List[str]:
    """Validate ``class_names`` against ``num_classes``.

    ``None`` yields positional placeholders ``class_0..class_{C-1}``. Those are
    *indices rendered as strings*, not semantics — this module never invents a
    label that carries meaning.
    """
    if class_names is None:
        return [f"class_{i}" for i in range(int(num_classes))]
    names = [str(n) for n in class_names]
    if len(names) != int(num_classes):
        raise ValueError(
            f"class_names has {len(names)} entries but stats declare "
            f"{int(num_classes)} classes"
        )
    return names


# --------------------------------------------------------------------------- #
# HebbianStats — the serializable measurement
# --------------------------------------------------------------------------- #


@dataclass
class HebbianStats:
    """Frozen Hebbian measurements for one observed layer (numpy, no torch).

    Every field is something the recorder *measured*; nothing here is inferred.

    Attributes
    ----------
    coact:
        ``[U, U]`` EMA of ``â âᵀ`` where ``â`` is the spatially pooled,
        rectified, L2-normalized activation vector of one sample. Symmetric,
        non-negative; the diagonal is each unit's mean squared normalized
        firing.
    mean_act:
        ``[U]`` EMA of ``â`` — each tracked unit's mean normalized firing rate.
        Drives the dead-unit cutoff and concept ``importance``.
    class_act:
        ``[num_classes, U]`` **mean** normalized firing per class (the running
        per-class sum divided by ``class_count``); rows for classes never seen
        are zero.
    class_count:
        ``[num_classes]`` how many samples of each class were observed — the
        denominator behind ``class_act``, kept so the measurement is auditable.
    unit_index:
        ``[U]`` original channel ids of the tracked units. Wide layers are
        subsampled (``max_units``), so tracked index ``i`` is channel
        ``unit_index[i]``.
    layer:
        Name of the observed layer (used in node ids and provenance).
    num_updates:
        Number of hook fires (batches) folded into the EMA.
    """

    coact: np.ndarray
    mean_act: np.ndarray
    class_act: np.ndarray
    unit_index: np.ndarray
    layer: str = "hebbian"
    num_updates: int = 0
    class_count: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.coact = np.asarray(self.coact, dtype=np.float64)
        self.mean_act = np.asarray(self.mean_act, dtype=np.float64).ravel()
        self.class_act = np.asarray(self.class_act, dtype=np.float64)
        self.unit_index = np.asarray(self.unit_index, dtype=np.int64).ravel()
        self.layer = str(self.layer)
        self.num_updates = int(self.num_updates)
        if self.coact.ndim != 2 or self.coact.shape[0] != self.coact.shape[1]:
            raise ValueError(f"coact must be square [U, U], got {self.coact.shape}")
        u = self.coact.shape[0]
        if self.mean_act.shape[0] != u:
            raise ValueError(
                f"mean_act has {self.mean_act.shape[0]} entries, expected {u}"
            )
        if self.class_act.ndim != 2 or self.class_act.shape[1] != u:
            raise ValueError(
                f"class_act must be [num_classes, {u}], got {self.class_act.shape}"
            )
        if self.unit_index.shape[0] != u:
            raise ValueError(
                f"unit_index has {self.unit_index.shape[0]} entries, expected {u}"
            )
        if self.class_count is None:
            self.class_count = np.zeros(self.class_act.shape[0], dtype=np.float64)
        else:
            self.class_count = np.asarray(self.class_count, dtype=np.float64).ravel()
            if self.class_count.shape[0] != self.class_act.shape[0]:
                raise ValueError(
                    f"class_count has {self.class_count.shape[0]} entries, "
                    f"expected {self.class_act.shape[0]}"
                )

    # -- shape ------------------------------------------------------------- #

    @property
    def num_units(self) -> int:
        """Number of tracked units ``U``."""
        return int(self.coact.shape[0])

    @property
    def num_classes(self) -> int:
        """Number of classes the class-conditional statistics cover."""
        return int(self.class_act.shape[0])

    # -- derived ------------------------------------------------------------ #

    def correlation(self) -> np.ndarray:
        """``[U, U]`` cosine-normalized co-activation (see
        :func:`correlation_matrix`)."""
        return correlation_matrix(self.coact)

    def class_affinity(self) -> np.ndarray:
        """``[num_classes, U]`` mean per-class firing — ``class_act`` verbatim.

        Present as a method for parity with the legacy
        ``HebbianFeatureMemory.class_affinity``; the division by ``class_count``
        already happened when the stats were frozen.
        """
        return self.class_act

    def dead_units(
        self, activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD
    ) -> np.ndarray:
        """Indices of units whose mean activation is below
        ``activity_threshold`` × the most active unit's."""
        cutoff = float(activity_threshold) * max(float(self.mean_act.max()), 1e-12)
        return np.where(~(self.mean_act > cutoff))[0]

    # -- serialization ------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict (plain lists, no ndarrays) — the shipping format.

        Round-trips losslessly through :meth:`from_dict` and through
        ``json.dumps``/``json.loads`` (Python's JSON float repr is exact for
        float64), so recorded statistics can be shipped and re-clustered
        without retraining or a torch pickle.
        """
        return {
            "layer": self.layer,
            "num_updates": int(self.num_updates),
            "num_units": self.num_units,
            "num_classes": self.num_classes,
            "unit_index": [int(i) for i in self.unit_index.tolist()],
            "coact": self.coact.tolist(),
            "mean_act": self.mean_act.tolist(),
            "class_act": self.class_act.tolist(),
            "class_count": self.class_count.tolist(),  # type: ignore[union-attr]
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HebbianStats":
        """Inverse of :meth:`to_dict`."""
        return cls(
            coact=np.asarray(payload["coact"], dtype=np.float64),
            mean_act=np.asarray(payload["mean_act"], dtype=np.float64),
            class_act=np.asarray(payload["class_act"], dtype=np.float64),
            unit_index=np.asarray(payload["unit_index"], dtype=np.int64),
            layer=str(payload.get("layer", "hebbian")),
            num_updates=int(payload.get("num_updates", 0)),
            class_count=(
                None
                if payload.get("class_count") is None
                else np.asarray(payload["class_count"], dtype=np.float64)
            ),
        )

    def save_npz(self, path: Any) -> Any:
        """Write the arrays to a compressed ``.npz`` (no pickle, no torch)."""
        np.savez_compressed(
            path,
            coact=self.coact,
            mean_act=self.mean_act,
            class_act=self.class_act,
            class_count=self.class_count,
            unit_index=self.unit_index,
            layer=np.array(self.layer),
            num_updates=np.array(self.num_updates),
        )
        return path

    @classmethod
    def load_npz(cls, path: Any) -> "HebbianStats":
        """Read back a :meth:`save_npz` file."""
        with np.load(path, allow_pickle=False) as z:
            return cls(
                coact=z["coact"],
                mean_act=z["mean_act"],
                class_act=z["class_act"],
                unit_index=z["unit_index"],
                layer=str(z["layer"]),
                num_updates=int(z["num_updates"]),
                class_count=z["class_count"],
            )


# --------------------------------------------------------------------------- #
# concepts
# --------------------------------------------------------------------------- #


@dataclass
class HebbianConcept:
    """A cluster of units that co-activate — measured, deliberately unnamed.

    ``units`` index into the tracked units (``channels`` gives the original
    channel ids). ``coherence`` is the mean off-diagonal correlation inside the
    cluster; ``importance`` the mean measured activation of its members;
    ``class_scores`` the raw measured mean firing rate per class (all classes,
    index-aligned with ``class_names``); ``class_affinity`` the top few of those
    normalized so the strongest class reads ``1.0`` — the concept→class edges.

    There is **no label field**: turning a concept into human language is a
    separate grounding step, and inventing one here would be unmeasured.
    """

    concept_id: int
    layer: str
    units: List[int]
    channels: List[int]
    coherence: float
    importance: float
    class_scores: List[float]
    class_affinity: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict for the graph asset."""
        return {
            "id": int(self.concept_id),
            "kind": CONCEPT_KIND,
            "layer": self.layer,
            "units": [int(u) for u in self.units],
            "channels": [int(c) for c in self.channels],
            "coherence": round(float(self.coherence), SCALAR_DECIMALS),
            "importance": round(float(self.importance), SCALAR_DECIMALS),
            "class_scores": [
                round(float(s), SCALAR_DECIMALS) for s in self.class_scores
            ],
            "class_affinity": {
                name: round(float(score), WEIGHT_DECIMALS)
                for name, score in self.class_affinity.items()
            },
        }


def cluster_concepts(
    stats: HebbianStats,
    class_names: Optional[Sequence[str]] = None,
    *,
    n_concepts: int = DEFAULT_N_CONCEPTS,
    min_units: int = DEFAULT_MIN_UNITS,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    top_classes: int = DEFAULT_TOP_CLASSES,
) -> List[HebbianConcept]:
    """Cluster units into concepts by their Hebbian co-activation fingerprint.

    Port of ``hatchvision.explain.concepts.cluster_concepts``. Dead and
    near-dead units (mean activation below ``activity_threshold`` × the most
    active unit) are excluded **first** — with wide sparse neuron spaces they
    otherwise dominate the geometry and everything alive chains into one giant
    cluster. The survivors are clustered by Ward agglomeration on the rows of
    the correlation matrix (each unit's "who do I fire with" fingerprint), which
    produces balanced clusters where average-linkage on raw correlation distance
    degenerates.

    Clusters smaller than ``min_units`` are dropped. Concepts are ordered by
    descending ``importance`` (ties broken by smallest member index) and given
    ids ``0..P-1`` in that order, so the ordering is fully deterministic.

    Class affinity is normalized per concept relative to its strongest class
    (top class = 1.0), so scores stay meaningful for datasets with many classes;
    the un-normalized measurements are kept alongside in ``class_scores``.
    """
    names = _resolve_class_names(class_names, stats.num_classes)
    corr = stats.correlation()
    mean_act = stats.mean_act
    u = stats.num_units

    cutoff = float(activity_threshold) * max(float(mean_act.max()), 1e-12)
    active = np.where(mean_act > cutoff)[0]
    if len(active) < max(int(min_units) * 2, 4):
        # Degenerate memory (almost nothing fired): keep every unit rather than
        # emit an empty graph, and say so via the returned concepts' contents.
        active = np.arange(u)

    n_clusters = max(
        1, min(int(n_concepts), len(active) // max(int(min_units), 1), len(active))
    )
    fingerprints = corr[np.ix_(active, active)]
    labels = ward_labels(fingerprints, n_clusters)

    affinity = stats.class_affinity()  # [num_classes, U]

    groups: List[Tuple[float, int, np.ndarray]] = []
    for cid in range(int(labels.max()) + 1):
        units = active[labels == cid]
        if len(units) < int(min_units):
            continue
        importance = float(mean_act[units].mean())
        groups.append((importance, int(units.min()), units))
    groups.sort(key=lambda g: (-g[0], g[1]))

    concepts: List[HebbianConcept] = []
    for new_id, (importance, _first, units) in enumerate(groups):
        sub = corr[np.ix_(units, units)]
        off_diag = sub[~np.eye(len(units), dtype=bool)]
        coherence = float(off_diag.mean()) if off_diag.size else 0.0
        cls_scores = affinity[:, units].mean(axis=1) if affinity.size else np.zeros(0)
        peak = float(cls_scores.max()) if cls_scores.size else 0.0
        norm = cls_scores / peak if peak > 0 else cls_scores
        order = np.argsort(-norm, kind="stable") if norm.size else np.zeros(0, int)
        top = {
            names[int(i)]: float(norm[int(i)])
            for i in order[: int(top_classes)]
            if norm[int(i)] > 0
        }
        concepts.append(
            HebbianConcept(
                concept_id=new_id,
                layer=stats.layer,
                units=[int(x) for x in units.tolist()],
                channels=[int(stats.unit_index[int(x)]) for x in units.tolist()],
                coherence=coherence,
                importance=importance,
                class_scores=[float(s) for s in np.asarray(cls_scores).tolist()],
                class_affinity=top,
            )
        )
    return concepts


# --------------------------------------------------------------------------- #
# HebbianGraphProvider
# --------------------------------------------------------------------------- #


class HebbianGraphProvider:
    """GraphProvider over Hebbian co-activation statistics (numpy-only).

    Satisfies the :class:`vitreous.graph.GraphProvider` Protocol: ``nodes`` /
    ``edges`` / ``communities``, returning the very same
    :class:`~vitreous.graph.GraphNode` / :class:`~vitreous.graph.GraphEdge` /
    :class:`~vitreous.graph.Community` dataclasses the ViT provider returns.

    Unlike the ViT provider (whose graph varies per attention *layer*), a
    Hebbian memory summarizes one observed layer into a single static graph, so
    the ``layer`` arguments are accepted for interface parity and ignored — the
    same accommodation :class:`vitreous.som.SomGraphProvider` makes.

    The "trace" this provider consumes is a :class:`HebbianStats` — a frozen
    measurement, never a live model. It may be bound at construction time
    (``HebbianGraphProvider(stats)``) or passed per call
    (``provider.nodes(stats)``); the per-call form is what makes the Protocol
    signatures line up with the ViT provider's.
    """

    def __init__(
        self,
        stats: Optional[HebbianStats] = None,
        *,
        k: int = DEFAULT_K,
        seed: int = DEFAULT_SEED,
        class_names: Optional[Sequence[str]] = None,
        n_concepts: int = DEFAULT_N_CONCEPTS,
        min_units: int = DEFAULT_MIN_UNITS,
        activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
        top_classes: int = DEFAULT_TOP_CLASSES,
    ) -> None:
        self.stats = stats
        self.k = int(k)
        self.seed = int(seed)
        self.class_names = None if class_names is None else [str(c) for c in class_names]
        self.n_concepts = int(n_concepts)
        self.min_units = int(min_units)
        self.activity_threshold = float(activity_threshold)
        self.top_classes = int(top_classes)

    # -- plumbing ------------------------------------------------------------ #

    def _resolve(self, trace: Any) -> HebbianStats:
        """Accept ``HebbianStats`` (or a dict from :meth:`HebbianStats.to_dict`)
        per call, falling back to the instance's bound stats."""
        if trace is None:
            trace = self.stats
        if trace is None:
            raise ValueError(
                "no HebbianStats supplied — pass one to the constructor or the call"
            )
        if isinstance(trace, HebbianStats):
            return trace
        if isinstance(trace, dict):
            return HebbianStats.from_dict(trace)
        raise TypeError(f"expected HebbianStats, got {type(trace).__name__}")

    # -- nodes --------------------------------------------------------------- #

    def nodes(self, trace: Any = None) -> List[GraphNode]:
        """Unit + concept + class nodes.

        ``U`` unit nodes (one per tracked unit, carrying its original channel,
        measured mean activation, community and concept), one node per detected
        concept, and one per class in the class-conditional statistics.
        ``GraphNode.layer`` is ``0`` (a Hebbian memory is a single static graph);
        the observed layer's *name* travels in ``ref["layer"]``.
        """
        stats = self._resolve(trace)
        parts = self._partition(stats)
        comm_of = {i: ci for ci, members in enumerate(parts) for i in members}
        concepts = self.concepts(stats)
        concept_of = {u: c.concept_id for c in concepts for u in c.units}
        dead = set(int(d) for d in stats.dead_units(self.activity_threshold).tolist())
        names = _resolve_class_names(self.class_names, stats.num_classes)

        out: List[GraphNode] = []
        for i in range(stats.num_units):
            out.append(
                GraphNode(
                    id=unit_node_id(stats.layer, i),
                    kind=UNIT_KIND,
                    layer=0,
                    ref={
                        "unit": i,
                        "channel": int(stats.unit_index[i]),
                        "layer": stats.layer,
                        "activation": float(stats.mean_act[i]),
                        "community": int(comm_of.get(i, -1)),
                        "concept": int(concept_of.get(i, -1)),
                        "dead": bool(i in dead),
                    },
                )
            )
        for c in concepts:
            out.append(
                GraphNode(
                    id=concept_node_id(c.concept_id),
                    kind=CONCEPT_KIND,
                    layer=0,
                    ref={
                        "concept": c.concept_id,
                        "layer": stats.layer,
                        "units": list(c.units),
                        "coherence": c.coherence,
                        "importance": c.importance,
                        "classes": dict(c.class_affinity),
                    },
                )
            )
        for ci, name in enumerate(names):
            out.append(
                GraphNode(
                    id=class_node_id(name),
                    kind=CLASS_KIND,
                    layer=0,
                    ref={"class_index": ci, "name": name},
                )
            )
        return out

    # -- edges --------------------------------------------------------------- #

    def _top_partners(self, corr: np.ndarray) -> List[Tuple[int, int, float]]:
        """``(partner, unit, weight)`` for each unit's top-k co-activation
        partners — self excluded, deterministic tie-break by unit index."""
        u = corr.shape[0]
        k = min(self.k, max(u - 1, 0))
        out: List[Tuple[int, int, float]] = []
        for i in range(u):
            row = corr[i].copy()
            row[i] = -np.inf  # a unit's correlation with itself is 1 by
            # construction and carries no information
            top = np.argsort(-row, kind="stable")[:k]
            for j in top:
                w = float(np.clip(corr[i, int(j)], 0.0, 1.0))
                out.append((int(j), i, w))
        return out

    def edges(self, trace: Any = None, layer: int = 0) -> List[GraphEdge]:
        """Top-k co-activation edges per unit — exactly ``num_units * k``.

        For each unit ``i`` the ``k`` partners with the highest correlation are
        kept as edges ``partner -> unit`` (self-edges excluded, so ``k`` is
        capped at ``num_units - 1``). Weights are the cosine-normalized
        co-activation clipped into ``[0, 1]`` — the clip is a no-op for a
        genuine EMA Gram matrix and only guards hand-built inputs.

        Concept membership and class-affinity edges live in
        :meth:`concept_edges`, so this method stays the pure unit↔unit
        co-activation graph.
        """
        stats = self._resolve(trace)
        corr = stats.correlation()
        return [
            GraphEdge(
                source=unit_node_id(stats.layer, src),
                target=unit_node_id(stats.layer, dst),
                weight=w,
                layer=0,
            )
            for src, dst, w in self._top_partners(corr)
        ]

    def concept_edges(self, trace: Any = None) -> List[GraphEdge]:
        """Membership (``unit -> concept``, weight 1.0) + class affinity
        (``concept -> class``, weight = normalized affinity) edges.

        These are the bundle's structural links: which units make up a concept,
        and which classes that concept fires on.
        """
        stats = self._resolve(trace)
        out: List[GraphEdge] = []
        for c in self.concepts(stats):
            for u in c.units:
                out.append(
                    GraphEdge(
                        source=unit_node_id(stats.layer, u),
                        target=concept_node_id(c.concept_id),
                        weight=1.0,
                        layer=0,
                    )
                )
            for name, score in c.class_affinity.items():
                if score <= 0:
                    continue
                out.append(
                    GraphEdge(
                        source=concept_node_id(c.concept_id),
                        target=class_node_id(name),
                        weight=float(score),
                        layer=0,
                    )
                )
        return out

    # -- communities --------------------------------------------------------- #

    def _partition(self, trace: Any = None) -> List[List[int]]:
        """Louvain partition (lists of unit indices), seeded — mirrors
        ``vitreous.graph.ViTTokenGraphProvider._partition``."""
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        stats = self._resolve(trace)
        corr = stats.correlation()
        u = corr.shape[0]
        # Undirected weighted graph: symmetrize, drop the diagonal.
        w = (corr + corr.T) * 0.5
        np.fill_diagonal(w, 0.0)
        g = nx.Graph()
        g.add_nodes_from(range(u))
        iu = np.triu_indices(u, k=1)
        for i, j, weight in zip(iu[0].tolist(), iu[1].tolist(), w[iu].tolist()):
            if weight > 0.0:
                g.add_edge(i, j, weight=float(weight))
        parts = louvain_communities(g, weight="weight", seed=self.seed)
        # Deterministic ordering: sort communities by their smallest member.
        return [sorted(int(x) for x in c) for c in sorted(parts, key=lambda c: min(c))]

    def communities(self, trace: Any = None, layer: int = 0) -> List[Community]:
        """Louvain communities over the symmetrized co-activation graph."""
        stats = self._resolve(trace)
        return [
            Community(
                id=community_id(stats.layer, ci),
                layer=0,
                members=[unit_node_id(stats.layer, i) for i in members],
            )
            for ci, members in enumerate(self._partition(stats))
        ]

    # -- concepts ------------------------------------------------------------ #

    def concepts(self, trace: Any = None) -> List[HebbianConcept]:
        """Concepts clustered from the co-activation fingerprints (see
        :func:`cluster_concepts`)."""
        stats = self._resolve(trace)
        return cluster_concepts(
            stats,
            self.class_names,
            n_concepts=self.n_concepts,
            min_units=self.min_units,
            activity_threshold=self.activity_threshold,
            top_classes=self.top_classes,
        )


# --------------------------------------------------------------------------- #
# build_hebbian_graph_asset
# --------------------------------------------------------------------------- #


def build_hebbian_graph_asset(
    stats: HebbianStats,
    class_names: Optional[Sequence[str]] = None,
    *,
    k: int = DEFAULT_K,
    n_concepts: int = DEFAULT_N_CONCEPTS,
    seed: int = DEFAULT_SEED,
    min_units: int = DEFAULT_MIN_UNITS,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    top_classes: int = DEFAULT_TOP_CLASSES,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the compact ``hebbian_graph.json`` payload (see module docstring).

    Structurally parallel to :func:`vitreous.graph.build_graph_asset`: compact
    ``[src, dst, weight]`` edge triples in unit-index space, weights quantized
    to 3 decimals, a per-node community index, and a seeded/deterministic
    partition — the same input always yields a byte-identical asset.

    ``provenance`` is recorded verbatim (dataset, run, checkpoint, …) and is the
    only free-form field; every other number is measured.
    """
    provider = HebbianGraphProvider(
        stats,
        k=k,
        seed=seed,
        class_names=class_names,
        n_concepts=n_concepts,
        min_units=min_units,
        activity_threshold=activity_threshold,
        top_classes=top_classes,
    )
    names = _resolve_class_names(class_names, stats.num_classes)
    corr = stats.correlation()
    u = stats.num_units

    parts = provider._partition(stats)
    comm_of = {i: ci for ci, members in enumerate(parts) for i in members}
    concepts = provider.concepts(stats)
    concept_of = {unit: c.concept_id for c in concepts for unit in c.units}
    dead = set(int(d) for d in stats.dead_units(activity_threshold).tolist())

    nodes = [
        {
            "idx": i,
            "kind": UNIT_KIND,
            "channel": int(stats.unit_index[i]),
            "community": int(comm_of.get(i, -1)),
            "concept": int(concept_of.get(i, -1)),
            "activation": round(float(stats.mean_act[i]), SCALAR_DECIMALS),
            "dead": bool(i in dead),
        }
        for i in range(u)
    ]

    edges = [
        [int(src), int(dst), round(float(w), WEIGHT_DECIMALS)]
        for src, dst, w in provider._top_partners(corr)
    ]

    membership = [
        [int(unit), int(c.concept_id)] for c in concepts for unit in c.units
    ]
    name_to_index = {name: i for i, name in enumerate(names)}
    affinity = [
        [int(c.concept_id), int(name_to_index[name]), round(float(score), WEIGHT_DECIMALS)]
        for c in concepts
        for name, score in c.class_affinity.items()
        if score > 0
    ]

    return {
        "provider": "hebbian",
        "layer": stats.layer,
        "num_units": int(u),
        "num_concepts": len(concepts),
        "num_classes": int(stats.num_classes),
        "k": int(min(k, max(u - 1, 0))),
        "seed": int(seed),
        "num_updates": int(stats.num_updates),
        "class_names": list(names),
        "activity_threshold": float(activity_threshold),
        "dead_units": sorted(dead),
        "nodes": nodes,
        "edges": edges,
        "concepts": [c.to_dict() for c in concepts],
        "classes": [
            {"idx": i, "kind": CLASS_KIND, "name": name} for i, name in enumerate(names)
        ],
        "membership": membership,
        "affinity": affinity,
        "communities": [list(members) for members in parts],
        "id_convention": dict(ID_CONVENTION),
        "edge_semantics": (
            "partner->unit; per unit the top-k strongest co-activation partners "
            "(self excluded); weight = coact_ij / sqrt(coact_ii * coact_jj) "
            "clipped to [0,1] and rounded to 3 decimals"
        ),
        "node_semantics": (
            "one node per tracked unit; 'channel' is the original layer channel, "
            "'activation' the measured mean normalized firing rate, 'dead' marks "
            "units below activity_threshold * max activation (excluded from "
            "concepts, never hidden)"
        ),
        "concept_method": {
            "method": "ward_on_coactivation_fingerprints",
            "n_concepts": int(n_concepts),
            "min_units": int(min_units),
            "activity_threshold": float(activity_threshold),
            "top_classes": int(top_classes),
            "labels": "none — concepts are unnamed by design; naming is a "
            "separate grounding step",
        },
        "community_method": {
            "method": "louvain",
            "seed": int(seed),
            "graph": "symmetrized cosine-normalized co-activation, diagonal removed",
        },
        "provenance": provenance or {},
    }


# --------------------------------------------------------------------------- #
# lazy HebbianRecorder (torch imported only on first access)
# --------------------------------------------------------------------------- #

_RECORDER_CLASS: Optional[type] = None


def _build_recorder_class() -> type:
    """Build (and cache) the torch ``HebbianRecorder`` class. Imports torch."""
    global _RECORDER_CLASS
    if _RECORDER_CLASS is not None:
        return _RECORDER_CLASS

    from contextlib import contextmanager

    import torch

    class HebbianRecorder:
        """Observe co-activation of hidden units while a model runs (torch).

        Port of ``hatchvision.hebbian.memory.HebbianFeatureMemory``, trimmed to
        the recording surface: analysis lives in :class:`HebbianGraphProvider`,
        which consumes the numpy :class:`HebbianStats` this produces.

        **Pure observation.** Hooks run inside ``torch.no_grad()``, ``detach()``
        every tensor they touch and return ``None``, so they cannot alter
        activations or gradients: training is bit-identical with and without a
        recorder attached. (The legacy repo asserts exactly this; the property
        must survive the port.)

        Parameters
        ----------
        model:
            Any module exposing ``hebbian_layers() -> Dict[str, nn.Module]``, or
            pass ``layers`` explicitly.
        num_classes:
            Enables class-conditional firing statistics when labels are supplied
            through :meth:`observe_labels` before each forward pass.
        layers:
            ``{name: module}`` to observe; overrides ``model.hebbian_layers()``.
        max_units:
            Layers wider than this are subsampled to a fixed (seeded) random set
            of channels so the co-activation matrix stays tractable.
        momentum:
            EMA momentum; higher forgets old batches faster.
        seed:
            Seeds the channel subsampling, so the tracked unit set is
            reproducible.
        """

        def __init__(
            self,
            model: Any = None,
            *,
            num_classes: int,
            layers: Optional[Dict[str, Any]] = None,
            max_units: int = DEFAULT_MAX_UNITS,
            momentum: float = DEFAULT_MOMENTUM,
            seed: int = DEFAULT_SEED,
        ) -> None:
            if layers is None:
                if model is None or not hasattr(model, "hebbian_layers"):
                    raise ValueError(
                        "model has no hebbian_layers(); pass layers= explicitly"
                    )
                layers = model.hebbian_layers()
            if not layers:
                raise ValueError("no layers to observe")
            self.num_classes = int(num_classes)
            self.max_units = int(max_units)
            self.momentum = float(momentum)
            self.seed = int(seed)
            self.enabled = True
            self._labels: Any = None
            self._handles: List[Any] = []
            self._state: Dict[str, Dict[str, Any]] = {}
            self._gen = torch.Generator().manual_seed(int(seed))
            for name, module in layers.items():
                self._handles.append(module.register_forward_hook(self._make_hook(name)))

        # -- hooks (observation only) ------------------------------------- #

        @staticmethod
        def _pool(out: Any) -> Any:
            """Reduce any activation tensor to ``[batch, units]``."""
            if out.dim() == 4:  # conv maps [B, C, H, W]
                return out.mean(dim=(2, 3))
            if out.dim() == 3:  # token seqs [B, T, D]
                return out.mean(dim=1)
            return out  # already [B, D]

        def _make_hook(self, name: str):
            def hook(_module, _inputs, output):
                # Returning None guarantees the hook cannot replace the output.
                if not self.enabled or not isinstance(output, torch.Tensor):
                    return None
                with torch.no_grad():
                    a = self._pool(output.detach().float())
                    a = torch.relu(a)  # firing rates are positive
                    self._update(name, a)
                return None

            return hook

        def _update(self, name: str, a: Any) -> None:
            if name not in self._state:
                dim = int(a.shape[1])
                unit_index = torch.arange(dim)
                if dim > self.max_units:
                    unit_index = (
                        torch.randperm(dim, generator=self._gen)[: self.max_units]
                        .sort()
                        .values
                    )
                    dim = self.max_units
                self._state[name] = {
                    "dim": dim,
                    "coact": torch.zeros(dim, dim),
                    "mean_act": torch.zeros(dim),
                    "class_act": torch.zeros(self.num_classes, dim),
                    "class_count": torch.zeros(self.num_classes),
                    "unit_index": unit_index,
                    "updates": 0,
                }
            st = self._state[name]
            a = a[:, st["unit_index"].to(a.device)]
            a = a.to(st["coact"].device)
            # Normalize each sample so co-activation reflects firing *pattern*,
            # not overall magnitude (which drifts during training).
            a_hat = a / (a.norm(dim=1, keepdim=True) + 1e-8)
            batch_co = a_hat.t() @ a_hat / a.shape[0]
            m = self.momentum
            st["coact"].mul_(1 - m).add_(batch_co, alpha=m)
            st["mean_act"].mul_(1 - m).add_(a_hat.mean(dim=0), alpha=m)
            labels = self._labels
            if labels is not None and labels.shape[0] == a.shape[0]:
                labels = labels.to(dtype=torch.long, device=st["class_act"].device)
                st["class_act"].index_add_(0, labels, a_hat)
                st["class_count"].index_add_(
                    0, labels, torch.ones(labels.shape[0], dtype=torch.float32)
                )
            st["updates"] += 1

        # -- public API ---------------------------------------------------- #

        def observe_labels(self, labels: Any) -> None:
            """Provide labels for the *next* forward pass (class statistics)."""
            self._labels = labels.detach().cpu()

        @contextmanager
        def paused(self):
            """Temporarily stop recording (probing must not contaminate stats)."""
            prev, self.enabled = self.enabled, False
            try:
                yield self
            finally:
                self.enabled = prev

        def detach(self) -> None:
            """Remove all hooks from the model."""
            for h in self._handles:
                h.remove()
            self._handles.clear()

        def __enter__(self) -> "HebbianRecorder":
            return self

        def __exit__(self, *_exc) -> None:
            self.detach()

        @property
        def layer_names(self) -> List[str]:
            """Names of the layers that have actually fired at least once."""
            return list(self._state)

        def stats(self, layer: Optional[str] = None) -> HebbianStats:
            """Freeze one layer's measurements into numpy :class:`HebbianStats`.

            ``layer`` defaults to the only observed layer (an explicit name is
            required when several fired). ``class_act`` is divided by
            ``class_count`` here, so the frozen stats carry mean per-class
            firing rates.
            """
            if not self._state:
                raise RuntimeError("no activations observed yet")
            if layer is None:
                if len(self._state) > 1:
                    raise ValueError(
                        f"several layers observed {list(self._state)}; pass layer="
                    )
                layer = next(iter(self._state))
            if layer not in self._state:
                raise KeyError(f"layer {layer!r} not observed; have {list(self._state)}")
            st = self._state[layer]
            count = st["class_count"].clamp(min=1.0).unsqueeze(1)
            return HebbianStats(
                coact=_to_numpy(st["coact"]),
                mean_act=_to_numpy(st["mean_act"]),
                class_act=_to_numpy(st["class_act"] / count),
                class_count=_to_numpy(st["class_count"]),
                unit_index=np.asarray(
                    st["unit_index"].detach().cpu().numpy(), dtype=np.int64
                ),
                layer=str(layer),
                num_updates=int(st["updates"]),
            )

        def all_stats(self) -> Dict[str, HebbianStats]:
            """:meth:`stats` for every observed layer."""
            return {name: self.stats(name) for name in self._state}

    _RECORDER_CLASS = HebbianRecorder
    return _RECORDER_CLASS


def __getattr__(name: str) -> Any:  # PEP 562 — lazy torch import for the recorder
    if name == "HebbianRecorder":
        return _build_recorder_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_MAX_UNITS",
    "DEFAULT_MOMENTUM",
    "DEFAULT_N_CONCEPTS",
    "DEFAULT_MIN_UNITS",
    "DEFAULT_ACTIVITY_THRESHOLD",
    "DEFAULT_TOP_CLASSES",
    "ID_CONVENTION",
    "HebbianStats",
    "HebbianConcept",
    "HebbianGraphProvider",
    "build_hebbian_graph_asset",
    "cluster_concepts",
    "correlation_matrix",
    "ward_labels",
    "unit_node_id",
    "concept_node_id",
    "class_node_id",
    "community_id",
    # HebbianRecorder is provided lazily via module __getattr__ (imports torch).
]
