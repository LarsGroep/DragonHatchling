"""Dataset-level projections (§10) — offline, synthetic, no model.

PCA + t-SNE always run (scikit-learn); UMAP is skipif-guarded. Covers coord
shapes/dtype/determinism, transform-capability gating, trajectory projection,
and the reducer + coordinate persist/reload round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from vitreous.projections import (  # noqa: E402
    build_projection_artifacts,
    fit_projections,
    load_coords,
    project_trajectory,
    save_coords,
    umap_available,
)

N, D = 60, 48


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(0).standard_normal((N, D)).astype(np.float32)


@pytest.fixture(scope="module")
def per_layer():
    return np.random.default_rng(1).standard_normal((13, D)).astype(np.float32)


def test_pca_shape_dtype_and_transform(X):
    sets = fit_projections(X, methods=("pca",), seed=0)
    ps = sets["pca"]
    assert ps.coords.shape == (N, 2)
    assert ps.coords.dtype == np.float16
    assert ps.supports_transform is True
    assert ps.reducer is not None and hasattr(ps.reducer, "transform")


def test_pca_determinism(X):
    a = fit_projections(X, methods=("pca",), seed=0)["pca"].coords
    b = fit_projections(X, methods=("pca",), seed=0)["pca"].coords
    assert np.array_equal(a, b)


def test_tsne_shape_and_no_transform(X):
    ps = fit_projections(X, methods=("tsne",), seed=0)["tsne"]
    assert ps.coords.shape == (N, 2)
    assert ps.coords.dtype == np.float16
    assert ps.supports_transform is False
    assert ps.reducer is None


@pytest.mark.skipif(not umap_available(), reason="umap-learn not installed")
def test_umap_shape_and_transform(X):
    ps = fit_projections(X, methods=("umap",), seed=0)["umap"]
    assert ps.coords.shape == (N, 2)
    assert ps.coords.dtype == np.float16
    assert ps.supports_transform is True
    assert hasattr(ps.reducer, "transform")


def test_trajectory_only_for_transform_methods(X, per_layer):
    sets = fit_projections(X, methods=("pca", "tsne"), seed=0)
    traj = project_trajectory(sets, per_layer)
    assert "pca" in traj
    assert traj["pca"].shape == (13, 2)
    assert traj["pca"].dtype == np.float16
    # t-SNE has no transform -> excluded from trajectories.
    assert "tsne" not in traj


def test_coords_persist_reload_roundtrip(tmp_path, X):
    ps = fit_projections(X, methods=("pca",), seed=0)["pca"]
    save_coords(tmp_path, "proj_pca_L0", ps.coords, {"method": "pca", "layer": 0})
    coords, meta = load_coords(tmp_path, "proj_pca_L0")
    assert np.array_equal(coords, ps.coords)
    assert meta["method"] == "pca" and meta["layer"] == 0
    assert meta["dtype"] == "float16"


def test_reducer_persist_reload_roundtrip(tmp_path, X, per_layer):
    import joblib

    methods = ("pca", "umap", "tsne") if umap_available() else ("pca", "tsne")
    art = build_projection_artifacts(
        tmp_path, X, layer=6, dataset="toy", model="m", methods=methods,
        seed=0, per_layer_cls=per_layer,
    )
    assert (tmp_path / "projections.json").exists()
    # PCA reducer reloads and transforms consistently with the live reducer.
    live = fit_projections(X, methods=("pca",), seed=0)["pca"].reducer
    reloaded = joblib.load(tmp_path / "reducer_pca_L6.joblib")
    out_live = live.transform(per_layer)
    out_reload = reloaded.transform(per_layer)
    assert np.allclose(out_live, out_reload)
    # t-SNE persisted coords but no reducer file.
    assert art["methods"]["tsne"]["reducer_file"] is None
    assert not (tmp_path / "reducer_tsne_L6.joblib").exists()


def test_umap_skipped_cleanly_when_absent(X, monkeypatch):
    # Force the availability probe to report False and confirm graceful skip.
    import vitreous.projections as proj

    monkeypatch.setattr(proj, "umap_available", lambda: False)
    sets = proj.fit_projections(X, methods=("pca", "umap", "tsne"), seed=0)
    assert "umap" not in sets
    assert "pca" in sets and "tsne" in sets
