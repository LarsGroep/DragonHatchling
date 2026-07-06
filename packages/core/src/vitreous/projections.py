"""Dataset-level latent projections (§10).

The Latent Embedding Explorer renders dataset-level 2D projections of CLS
tokens (per-image points) and per-image inference *trajectories* (the current
image's CLS position at each layer, projected into the layer's map). Projections
are **not** per-pack — they are a dataset-level artifact produced once per
(dataset, model, layer) and stored in their own directory (see
:func:`build_projection_artifacts`).

Methods
-------
* **pca**  — always available; ``PCA(2)`` supports ``.transform`` (trajectories,
  live uploads project into the fitted map).
* **umap** — default lens when installed; ``UMAP`` supports ``.transform``.
  Optional dependency (``umap-learn``); skipped honestly if unavailable.
* **tsne** — ``TSNE`` has **no** ``.transform`` (it re-fits every call), so it is
  used for the static landscape only, never for trajectories/uploads.

Reducers that support ``.transform`` (PCA, UMAP) are persisted with joblib so
the live service can place uploads *inside* the dataset landscape. All heavy
imports (scikit-learn, umap, joblib) are lazy so ``import vitreous`` stays free
of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_METHODS: Tuple[str, ...] = ("pca", "umap", "tsne")
# Methods whose fitted reducer supports out-of-sample .transform().
TRANSFORM_METHODS: Tuple[str, ...] = ("pca", "umap")


def umap_available() -> bool:
    """True if ``umap-learn`` (and its numba stack) import cleanly."""
    try:
        import umap  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class ProjectionSet:
    """One method's 2D projection of a set of vectors.

    ``coords`` is ``[N, 2]`` float16; ``reducer`` is the fitted estimator (only
    for transform-capable methods, else ``None``); ``supports_transform`` gates
    trajectory / upload projection.
    """

    method: str
    coords: np.ndarray                     # [N, 2] float16
    seed: int
    n: int
    supports_transform: bool
    reducer: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def _to_f32(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    return np.ascontiguousarray(np.asarray(x), dtype=np.float32)


def _fit_pca(X: np.ndarray, seed: int) -> ProjectionSet:
    from sklearn.decomposition import PCA

    n = X.shape[0]
    reducer = PCA(n_components=2, random_state=seed)
    coords = reducer.fit_transform(X)
    return ProjectionSet(
        method="pca", coords=coords.astype(np.float16), seed=seed, n=n,
        supports_transform=True, reducer=reducer,
        meta={"explained_variance_ratio": [float(v) for v in reducer.explained_variance_ratio_]},
    )


def _fit_umap(X: np.ndarray, seed: int, **kwargs: Any) -> ProjectionSet:
    import umap

    n = X.shape[0]
    n_neighbors = int(kwargs.pop("n_neighbors", min(15, max(2, n - 1))))
    reducer = umap.UMAP(
        n_components=2, random_state=seed, n_neighbors=n_neighbors, **kwargs
    )
    coords = reducer.fit_transform(X)
    return ProjectionSet(
        method="umap", coords=np.asarray(coords).astype(np.float16), seed=seed, n=n,
        supports_transform=True, reducer=reducer, meta={"n_neighbors": n_neighbors},
    )


def _fit_tsne(X: np.ndarray, seed: int, **kwargs: Any) -> ProjectionSet:
    from sklearn.manifold import TSNE

    n = X.shape[0]
    perplexity = float(kwargs.pop("perplexity", min(30.0, max(2.0, (n - 1) / 3.0))))
    reducer = TSNE(
        n_components=2, random_state=seed, perplexity=perplexity, init="pca", **kwargs
    )
    coords = reducer.fit_transform(X)
    return ProjectionSet(
        method="tsne", coords=coords.astype(np.float16), seed=seed, n=n,
        supports_transform=False, reducer=None, meta={"perplexity": perplexity},
    )


def fit_projections(
    cls_vectors: Any,
    methods: Tuple[str, ...] = DEFAULT_METHODS,
    seed: int = 0,
    *,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    tsne_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, ProjectionSet]:
    """Fit 2D projections of ``cls_vectors`` ``[N, D]`` for each method (§10).

    Returns a ``method -> ProjectionSet`` dict. UMAP is silently skipped when
    ``umap-learn`` is not installed (check :func:`umap_available`); PCA and t-SNE
    always run. PCA/UMAP carry a persistable ``reducer``; t-SNE does not.
    """
    X = _to_f32(cls_vectors)
    if X.ndim != 2:
        raise ValueError(f"expected cls_vectors [N, D], got shape {X.shape}")

    out: Dict[str, ProjectionSet] = {}
    for m in methods:
        if m == "pca":
            out["pca"] = _fit_pca(X, seed)
        elif m == "umap":
            if umap_available():
                out["umap"] = _fit_umap(X, seed, **(umap_kwargs or {}))
        elif m == "tsne":
            out["tsne"] = _fit_tsne(X, seed, **(tsne_kwargs or {}))
        else:
            raise ValueError(f"unknown projection method {m!r}")
    return out


def project_trajectory(
    reducers: Dict[str, Any],
    per_layer_cls: Any,
) -> Dict[str, np.ndarray]:
    """Project a per-layer CLS trajectory ``[S, D]`` into each transform-capable map.

    ``reducers`` maps method name -> a fitted reducer *or* a :class:`ProjectionSet`.
    Only methods that support ``.transform`` (PCA, UMAP) yield a trajectory;
    others are skipped. Returns ``method -> [S, 2]`` float16 arrays.
    """
    P = _to_f32(per_layer_cls)
    if P.ndim != 2:
        raise ValueError(f"expected per_layer_cls [S, D], got shape {P.shape}")
    out: Dict[str, np.ndarray] = {}
    for name, red in reducers.items():
        reducer = red.reducer if isinstance(red, ProjectionSet) else red
        if reducer is None or not hasattr(reducer, "transform"):
            continue
        coords = np.asarray(reducer.transform(P)).astype(np.float16)
        out[name] = coords
    return out


# --------------------------------------------------------------------------- #
# serialization (dataset-level artifact dir, NOT per-pack)
# --------------------------------------------------------------------------- #


def save_coords(out_dir: Any, name: str, coords: np.ndarray, sidecar: Dict[str, Any]) -> Tuple[Path, Path]:
    """Write ``name.bin`` (fp16 ``[N,2]`` C-order) + ``name.json`` sidecar."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(coords, dtype=np.float16)
    bin_path = out / f"{name}.bin"
    bin_path.write_bytes(arr.tobytes())
    meta = dict(sidecar)
    meta.update({"file": f"{name}.bin", "dtype": "float16", "shape": list(arr.shape)})
    json_path = out / f"{name}.json"
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return bin_path, json_path


def load_coords(out_dir: Any, name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read back ``name.bin`` + ``name.json`` written by :func:`save_coords`."""
    out = Path(out_dir)
    meta = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    shape = tuple(meta["shape"])
    arr = np.frombuffer((out / f"{name}.bin").read_bytes(), dtype=np.float16).reshape(shape)
    return arr.copy(), meta


def build_projection_artifacts(
    out_dir: Any,
    cls_vectors: Any,
    *,
    layer: int,
    dataset: str = "dataset",
    model: str = "model",
    methods: Tuple[str, ...] = DEFAULT_METHODS,
    seed: int = 0,
    per_layer_cls: Optional[Any] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    tsne_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit + persist the dataset-level projection artifacts for one layer (§10).

    Writes, per method, into ``out_dir``:

    * ``proj_{method}_L{layer}.bin`` / ``.json`` — the ``[N,2]`` fp16 coordinates
      and a sidecar (method, layer, n, seed, reducer filename).
    * ``reducer_{method}_L{layer}.joblib`` — the fitted reducer for
      transform-capable methods (PCA, UMAP), so uploads/trajectories can project.
    * (optional) ``traj_{method}_L{layer}.bin`` / ``.json`` — a per-layer CLS
      trajectory projected with the fitted reducers, when ``per_layer_cls`` given.

    Returns a manifest-like dict describing what was written.
    """
    import joblib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = fit_projections(
        cls_vectors, methods=methods, seed=seed,
        umap_kwargs=umap_kwargs, tsne_kwargs=tsne_kwargs,
    )

    artifacts: Dict[str, Any] = {
        "dataset": dataset, "model": model, "layer": int(layer), "seed": int(seed),
        "methods": {},
    }
    reducers: Dict[str, Any] = {}
    for method, ps in sets.items():
        reducer_file: Optional[str] = None
        if ps.supports_transform and ps.reducer is not None:
            reducer_file = f"reducer_{method}_L{layer}.joblib"
            joblib.dump(ps.reducer, out / reducer_file)
            reducers[method] = ps.reducer
        name = f"proj_{method}_L{layer}"
        save_coords(
            out, name, ps.coords,
            {
                "method": method, "layer": int(layer), "n": int(ps.n), "seed": int(seed),
                "dataset": dataset, "model": model,
                "supports_transform": bool(ps.supports_transform),
                "reducer_file": reducer_file, "extra": ps.meta,
            },
        )
        artifacts["methods"][method] = {
            "coords_file": f"{name}.bin", "sidecar": f"{name}.json",
            "reducer_file": reducer_file, "n": int(ps.n),
            "supports_transform": bool(ps.supports_transform),
        }

    if per_layer_cls is not None and reducers:
        traj = project_trajectory(reducers, per_layer_cls)
        for method, coords in traj.items():
            name = f"traj_{method}_L{layer}"
            save_coords(
                out, name, coords,
                {"method": method, "layer": int(layer), "kind": "trajectory",
                 "steps": int(coords.shape[0]), "seed": int(seed)},
            )
            artifacts["methods"].setdefault(method, {})["trajectory_file"] = f"{name}.bin"

    (out / "projections.json").write_text(
        json.dumps(artifacts, indent=2, sort_keys=True), encoding="utf-8"
    )
    return artifacts


__all__ = [
    "ProjectionSet",
    "DEFAULT_METHODS",
    "TRANSFORM_METHODS",
    "umap_available",
    "fit_projections",
    "project_trajectory",
    "save_coords",
    "load_coords",
    "build_projection_artifacts",
]
