"""SGP — SomGraphProvider: render a learned 3-D SOM as a ViTreous graph (§ SGP).

This is the core of the **SGP project** (``docs/SGP-ARCHITECTURE.md``): it turns
a trained UMT-ViT Self-Organizing Map (its neuron weights on a 3-D lattice) plus
a per-image BMU assignment map into the additive pack assets that the ViTreous
workbench renders — a native, topology-preserving graph whose node coordinates
are the *real* neuron lattice, not a force-simulated layout.

**Import discipline (M0 rule):** this module is **numpy-only**. It never imports
torch, and it never imports ``experiments/umtvit`` — the two packages stay
decoupled (both design contracts forbid a cross-import). The integration happens
in the Kaggle notebook, which installs both and passes plain numpy arrays across
the boundary: the caller pulls ``model(x)["volume"]`` and ``som.weights`` out as
numpy and hands them here.

Neuron indexing (frozen, matches ``umtvit.models.som3d.Soft3DSOM``)
------------------------------------------------------------------
Neurons live on a 3-D grid ``(Gz, Gy, Gx)`` and are flattened row-major:

    k = z * (Gy * Gx) + y * Gx + x

so ``divmod`` recovers ``(z, y, x)`` — identical to the SOM's own
``meshgrid(..., indexing="ij")`` weight order, which is what makes a BMU index
computed here line up with the SOM's neurons.

Honesty rules (§ SGP §1)
------------------------
Every quantity is measured: node coordinates are the literal lattice; edges are
literal lattice adjacency; edge weights are measured weight-space similarity
``1/(1+‖w_a − w_b‖)`` (so U-matrix ridges read as dark gaps); ``hits`` are real
BMU win counts; dead neurons (zero hits) are flagged, never hidden. The depth
axis of a BMU map is encoder depth — a *learned hierarchy*, never physical depth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "grid_coords",
    "lattice_edges",
    "som_umatrix",
    "som_communities",
    "bmu_indices",
    "bmu_map",
    "hit_counts",
    "build_som_graph_asset",
    "SomGraphProvider",
    "SomState",
]


# --------------------------------------------------------------------------- #
# lattice geometry
# --------------------------------------------------------------------------- #


def _validate_grid(grid: Sequence[int]) -> Tuple[int, int, int]:
    if len(grid) != 3:
        raise ValueError(f"grid must be a 3-tuple (Gz, Gy, Gx), got {tuple(grid)!r}")
    gz, gy, gx = (int(g) for g in grid)
    if gz <= 0 or gy <= 0 or gx <= 0:
        raise ValueError(f"grid dims must be positive, got {(gz, gy, gx)!r}")
    return gz, gy, gx


def grid_coords(grid: Sequence[int]) -> np.ndarray:
    """Return ``[K, 3]`` integer ``(z, y, x)`` coords in the frozen neuron order.

    ``k = z*Gy*Gx + y*Gx + x`` — the same flattening
    ``Soft3DSOM`` uses, so ``grid_coords(grid)[k]`` is neuron ``k``'s lattice
    position.
    """
    gz, gy, gx = _validate_grid(grid)
    zz, yy, xx = np.meshgrid(
        np.arange(gz), np.arange(gy), np.arange(gx), indexing="ij"
    )
    return np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(np.int64)


def lattice_edges(
    grid: Sequence[int], *, connectivity: str = "faces"
) -> List[Tuple[int, int]]:
    """Unique undirected lattice-adjacency edges ``(a, b)`` with ``a < b``.

    ``connectivity="faces"`` (default) is the 6-connected neighbourhood (±1 on
    exactly one axis). ``connectivity="full"`` is 26-connected (any ±1 offset).
    Every edge is a genuine grid neighbour — the honesty rule for SGP edges.
    """
    gz, gy, gx = _validate_grid(grid)
    coords = grid_coords(grid)
    index = {(int(z), int(y), int(x)): k for k, (z, y, x) in enumerate(coords)}

    if connectivity == "faces":
        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    elif connectivity == "full":
        offsets = [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)
        ]
    else:
        raise ValueError(
            f"unknown connectivity {connectivity!r}; expected 'faces' or 'full'"
        )

    edges: set = set()
    for (z, y, x), k in index.items():
        for dz, dy, dx in offsets:
            nb = (z + dz, y + dy, x + dx)
            j = index.get(nb)
            if j is not None:
                a, b = (k, j) if k < j else (j, k)
                edges.add((a, b))
    return sorted(edges)


# --------------------------------------------------------------------------- #
# U-matrix + communities
# --------------------------------------------------------------------------- #


def som_umatrix(
    weights: np.ndarray, grid: Sequence[int], *, connectivity: str = "faces"
) -> np.ndarray:
    """Per-neuron U-matrix value ``[K]``: mean weight-space distance to its
    lattice neighbours. High values are cluster boundaries (dark ridges)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"weights must be [K, C], got {w.shape}")
    K = w.shape[0]
    if K != int(np.prod(_validate_grid(grid))):
        raise ValueError(
            f"weights has {K} rows but grid {tuple(grid)} implies "
            f"{int(np.prod(_validate_grid(grid)))} neurons"
        )
    acc = np.zeros(K, dtype=np.float64)
    cnt = np.zeros(K, dtype=np.float64)
    for a, b in lattice_edges(grid, connectivity=connectivity):
        d = float(np.linalg.norm(w[a] - w[b]))
        acc[a] += d
        acc[b] += d
        cnt[a] += 1.0
        cnt[b] += 1.0
    return acc / np.where(cnt > 0, cnt, 1.0)


def _kmeans(X: np.ndarray, k: int, seed: int, n_iter: int = 100) -> np.ndarray:
    """Tiny seeded k-means++ (Lloyd) over ``X`` ``[N, D]`` → labels ``[N]``.

    Local so ``packages/core`` keeps its numpy-only dependency set (no sklearn).
    Deterministic for a given ``seed``.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = int(min(k, n))
    if k <= 1:
        return np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)

    # k-means++ seeding.
    first = int(rng.integers(n))
    centers = [X[first]]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = float(d2.sum())
        if total <= 0:
            centers.append(X[int(rng.integers(n))])
        else:
            idx = int(rng.choice(n, p=d2 / total))
            centers.append(X[idx])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(axis=1))
    C = np.stack(centers, axis=0)

    labels = np.zeros(n, dtype=np.int64)
    for it in range(n_iter):
        dists = (
            (X**2).sum(1)[:, None] + (C**2).sum(1)[None, :] - 2.0 * X @ C.T
        )
        new = dists.argmin(axis=1).astype(np.int64)
        if it > 0 and np.array_equal(new, labels):
            labels = new
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    return labels


def _canonical_relabel(labels: np.ndarray) -> np.ndarray:
    """Relabel so community ids appear in ascending first-occurrence order.

    Makes the community assignment independent of k-means' internal centroid
    order: the community containing neuron 0 is always ``0``, etc.
    """
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


def som_communities(
    weights: np.ndarray, k: int = 12, *, seed: int = 0
) -> np.ndarray:
    """Seeded k-means communities over neuron weight vectors → ``[K]`` labels.

    Deterministic (``seed``) and canonically relabelled. This is the v1 grouping
    (the U-matrix-watershed alternative is recorded as an SGP S6 option); the
    method + seed are stamped into ``som.json`` so it is reproducible.
    """
    return _canonical_relabel(_kmeans(np.asarray(weights, dtype=np.float64), k, seed))


# --------------------------------------------------------------------------- #
# BMU assignment (per-image activation)
# --------------------------------------------------------------------------- #


def bmu_indices(voxels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Best-matching-unit index per voxel: ``argmin_k ‖v − w_k‖`` → ``[M]``.

    ``voxels`` is ``[M, C]``, ``weights`` is ``[K, C]``. Uses the
    ``‖v‖²+‖w‖²−2v·w`` expansion so the pairwise matrix is only ``[M, K]``
    (never the ``[M, K, C]`` broadcast).
    """
    v = np.asarray(voxels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.ndim != 2 or w.ndim != 2 or v.shape[1] != w.shape[1]:
        raise ValueError(
            f"voxels [M,C] and weights [K,C] must share C; got {v.shape}, {w.shape}"
        )
    d = (v**2).sum(1)[:, None] + (w**2).sum(1)[None, :] - 2.0 * v @ w.T
    return d.argmin(axis=1).astype(np.int64)


def bmu_map(volume: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-voxel BMU map for one image ``→ [Z, H, W]`` uint16.

    ``volume`` is one image's latent voxel volume ``[H, W, Z, C]`` (i.e.
    ``UMTViT(x)["volume"][0]`` as numpy). The returned map is depth-first
    (``[Z, H, W]``) so scrubbing depth ``z`` indexes ``map[z]`` — the frame the
    SGP activation replay lights up.
    """
    v = np.asarray(volume)
    if v.ndim != 4:
        raise ValueError(f"volume must be [H, W, Z, C], got {v.shape}")
    H, W, Z, C = v.shape
    idx = bmu_indices(v.reshape(-1, C), weights).reshape(H, W, Z)
    return np.ascontiguousarray(np.transpose(idx, (2, 0, 1)).astype(np.uint16))


def hit_counts(indices: np.ndarray, num_neurons: int) -> np.ndarray:
    """BMU win counts per neuron ``→ [K]`` int64 (node sizing; ``0`` = dead)."""
    idx = np.asarray(indices, dtype=np.int64).ravel()
    return np.bincount(idx, minlength=int(num_neurons))[: int(num_neurons)].astype(
        np.int64
    )


# --------------------------------------------------------------------------- #
# som.json builder
# --------------------------------------------------------------------------- #


def build_som_graph_asset(
    weights: np.ndarray,
    grid: Sequence[int],
    *,
    hits: Optional[np.ndarray] = None,
    community_k: int = 12,
    seed: int = 0,
    connectivity: str = "faces",
    depth_steps: Optional[int] = None,
    volume_grid: Optional[Sequence[int]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the additive ``som.json`` asset for a trained SOM (§ SGP §4.1).

    Parameters
    ----------
    weights:
        SOM neuron weights ``[K, C]`` (``Soft3DSOM.weights`` as numpy).
    grid:
        ``(Gz, Gy, Gx)`` neuron lattice shape; ``K == Gz*Gy*Gx``.
    hits:
        Optional ``[K]`` BMU win counts (from a training/gallery pass); drives
        node sizing and the ``dead`` flag. Defaults to all-zero (every node
        flagged dead until a hit pass is supplied).
    community_k, seed:
        k-means community count + seed (stamped into the asset).
    connectivity:
        Lattice edge rule — ``"faces"`` (6-conn, default) or ``"full"`` (26-conn).
    depth_steps:
        Z of the BMU maps (encoder depth). Defaults to ``Gz`` but is really the
        transformer depth ``L``; pass it explicitly.
    volume_grid:
        ``(H', W')`` of the BMU maps, for the frontend patch↔voxel mapping.
    provenance:
        Free-form ``{run, dataset, epoch, ...}`` recorded verbatim.

    Returns
    -------
    dict
        The ``som.json`` payload (JSON-serializable), ready for
        ``PackWriter.add_json("som.json", asset)``.
    """
    w = np.asarray(weights, dtype=np.float64)
    gz, gy, gx = _validate_grid(grid)
    K = gz * gy * gx
    if w.ndim != 2 or w.shape[0] != K:
        raise ValueError(
            f"weights must be [K={K}, C]; got {w.shape} for grid {(gz, gy, gx)}"
        )

    coords = grid_coords(grid)
    umat = som_umatrix(w, grid, connectivity=connectivity)
    comm = som_communities(w, community_k, seed=seed)
    if hits is None:
        hits_arr = np.zeros(K, dtype=np.int64)
    else:
        hits_arr = np.asarray(hits, dtype=np.int64).ravel()
        if hits_arr.shape[0] != K:
            raise ValueError(f"hits must have {K} entries, got {hits_arr.shape[0]}")

    nodes: List[Dict[str, Any]] = []
    for k in range(K):
        z, y, x = (int(v) for v in coords[k])
        nodes.append(
            {
                "idx": k,
                "grid": [z, y, x],
                "hits": int(hits_arr[k]),
                "umatrix": round(float(umat[k]), 6),
                "community": int(comm[k]),
                "dead": bool(hits_arr[k] == 0),
            }
        )

    edges: List[List[Any]] = []
    for a, b in lattice_edges(grid, connectivity=connectivity):
        dist = float(np.linalg.norm(w[a] - w[b]))
        sim = 1.0 / (1.0 + dist)
        edges.append([int(a), int(b), round(sim, 6)])

    conn_label = "6-connected" if connectivity == "faces" else "26-connected"
    asset: Dict[str, Any] = {
        "provider": "som",
        "grid": [gz, gy, gx],
        "num_neurons": K,
        "depth_steps": int(depth_steps if depth_steps is not None else gz),
        "depth_semantics": "learned hierarchy (encoder depth), not physical depth",
        "volume_grid": [int(v) for v in volume_grid] if volume_grid is not None else None,
        "adjacency": conn_label,
        "nodes": nodes,
        "edges": edges,
        "edge_semantics": (
            "w = 1/(1+||w_a - w_b||); lattice adjacency only — every edge is a "
            "real grid neighbour"
        ),
        "communities": {
            "method": "kmeans_weights",
            "k": int(community_k),
            "seed": int(seed),
        },
        "dead_neurons": int((hits_arr == 0).sum()),
        "provenance": provenance or {},
    }
    return asset


# --------------------------------------------------------------------------- #
# SomGraphProvider — the GraphProvider-shaped view over a SOM
# --------------------------------------------------------------------------- #


class SomState:
    """Minimal container for a trained SOM's static geometry (numpy).

    Bundles the weights, grid, and optional hits so :class:`SomGraphProvider`
    can expose the same ``nodes``/``edges``/``communities`` surface the ViT
    ``GraphProvider`` does (``vitreous.graph``) — keeping the promise that the
    frontend abstraction admits new model families.
    """

    def __init__(
        self,
        weights: np.ndarray,
        grid: Sequence[int],
        *,
        hits: Optional[np.ndarray] = None,
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.grid = _validate_grid(grid)
        self.hits = None if hits is None else np.asarray(hits, dtype=np.int64).ravel()


class SomGraphProvider:
    """A SOM-backed provider mirroring ``vitreous.graph.GraphProvider`` (§ SGP §5).

    Unlike the ViT provider (nodes/edges vary per attention *layer*), the SOM
    graph is a single static lattice — ``layer`` arguments are accepted for
    interface parity and ignored.
    """

    def __init__(self, k: int = 12, seed: int = 0, connectivity: str = "faces") -> None:
        self.k = int(k)
        self.seed = int(seed)
        self.connectivity = connectivity

    def nodes(self, state: SomState) -> List[Dict[str, Any]]:
        asset = build_som_graph_asset(
            state.weights,
            state.grid,
            hits=state.hits,
            community_k=self.k,
            seed=self.seed,
            connectivity=self.connectivity,
        )
        return asset["nodes"]

    def edges(self, state: SomState, layer: int = 0) -> List[List[Any]]:
        w = state.weights
        out: List[List[Any]] = []
        for a, b in lattice_edges(state.grid, connectivity=self.connectivity):
            sim = 1.0 / (1.0 + float(np.linalg.norm(w[a] - w[b])))
            out.append([int(a), int(b), round(sim, 6)])
        return out

    def communities(self, state: SomState, layer: int = 0) -> np.ndarray:
        return som_communities(state.weights, self.k, seed=self.seed)
