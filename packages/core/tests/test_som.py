"""SGP core (vitreous.som) — offline, numpy-only, synthetic SOM.

Covers the frozen neuron indexing, lattice edge counts, U-matrix, deterministic
communities, BMU assignment/hits, the som.json builder, the GraphProvider-shaped
surface, and a PackWriter/PackReader round-trip of both SGP assets. No torch, no
umtvit import — mirrors the M0 discipline of test_graph.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from vitreous.som import (
    SomGraphProvider,
    SomState,
    bmu_indices,
    bmu_map,
    build_som_graph_asset,
    grid_coords,
    hit_counts,
    lattice_edges,
    som_communities,
    som_umatrix,
)

GRID = (4, 4, 4)  # 64 neurons
K = 64
C = 8


def _synthetic_weights(seed: int = 0) -> np.ndarray:
    """K neurons whose weights vary smoothly with lattice position (so the
    U-matrix and communities are non-degenerate)."""
    coords = grid_coords(GRID).astype(np.float64)
    rng = np.random.default_rng(seed)
    base = coords @ rng.standard_normal((3, C))  # smooth in grid space
    return base + 0.01 * rng.standard_normal((K, C))


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def test_grid_coords_matches_frozen_order():
    coords = grid_coords(GRID)
    assert coords.shape == (K, 3)
    gz, gy, gx = GRID
    for k in range(K):
        z, y, x = coords[k]
        assert k == z * (gy * gx) + y * gx + x  # frozen flattening


def test_lattice_edges_face_count():
    # 6-connected interior edges of a Gz×Gy×Gx grid: axis-wise (g-1)*rest.
    gz, gy, gx = GRID
    expected = (
        (gz - 1) * gy * gx + gz * (gy - 1) * gx + gz * gy * (gx - 1)
    )
    edges = lattice_edges(GRID, connectivity="faces")
    assert len(edges) == expected
    # all a < b, unique, valid.
    assert all(0 <= a < b < K for a, b in edges)
    assert len(set(edges)) == len(edges)


def test_lattice_edges_full_superset():
    faces = set(lattice_edges(GRID, connectivity="faces"))
    full = set(lattice_edges(GRID, connectivity="full"))
    assert faces.issubset(full)
    assert len(full) > len(faces)


def test_lattice_edges_bad_connectivity():
    with pytest.raises(ValueError):
        lattice_edges(GRID, connectivity="hexagonal")


# --------------------------------------------------------------------------- #
# U-matrix + communities
# --------------------------------------------------------------------------- #


def test_umatrix_shape_and_nonnegative():
    w = _synthetic_weights()
    umat = som_umatrix(w, GRID)
    assert umat.shape == (K,)
    assert np.all(umat >= 0)
    assert np.any(umat > 0)


def test_umatrix_grid_mismatch_raises():
    with pytest.raises(ValueError):
        som_umatrix(np.zeros((K + 1, C)), GRID)


def test_communities_deterministic_and_canonical():
    w = _synthetic_weights()
    a = som_communities(w, k=6, seed=0)
    b = som_communities(w, k=6, seed=0)
    assert np.array_equal(a, b)  # deterministic
    assert a.shape == (K,)
    # canonical: neuron 0 in community 0; ids are 0..n-1 dense in first-seen order.
    assert a[0] == 0
    assert set(a.tolist()) == set(range(a.max() + 1))


def test_communities_respect_k():
    w = _synthetic_weights()
    labels = som_communities(w, k=5, seed=1)
    assert labels.max() + 1 <= 5


# --------------------------------------------------------------------------- #
# BMU assignment
# --------------------------------------------------------------------------- #


def test_bmu_indices_recovers_exact_neuron():
    w = _synthetic_weights()
    # A voxel sitting exactly on neuron 17 must map to 17.
    vox = w[17][None, :]
    assert int(bmu_indices(vox, w)[0]) == 17


def test_bmu_map_shape_and_orientation():
    w = _synthetic_weights()
    H, W, Z = 5, 6, 4
    vol = np.random.default_rng(3).standard_normal((H, W, Z, C))
    bm = bmu_map(vol, w)
    assert bm.shape == (Z, H, W)
    assert bm.dtype == np.uint16
    # spot-check one voxel against a direct argmin.
    z, y, x = 2, 1, 3
    direct = int(bmu_indices(vol[y, x, z][None, :], w)[0])
    assert int(bm[z, y, x]) == direct


def test_hit_counts_sum_and_dead():
    idx = np.array([0, 0, 0, 3, 3, 10])
    hc = hit_counts(idx, K)
    assert hc.shape == (K,)
    assert hc.sum() == len(idx)
    assert hc[0] == 3 and hc[3] == 2 and hc[10] == 1
    assert hc[1] == 0  # dead


# --------------------------------------------------------------------------- #
# som.json builder
# --------------------------------------------------------------------------- #


def test_build_asset_structure():
    w = _synthetic_weights()
    hits = np.arange(K, dtype=np.int64)  # neuron 0 dead, rest alive
    asset = build_som_graph_asset(
        w, GRID, hits=hits, community_k=6, seed=0,
        depth_steps=8, volume_grid=(16, 16),
        provenance={"dataset": "ham10000", "epoch": 30},
    )
    assert asset["provider"] == "som"
    assert asset["grid"] == [4, 4, 4]
    assert asset["num_neurons"] == K
    assert asset["depth_steps"] == 8
    assert asset["volume_grid"] == [16, 16]
    assert len(asset["nodes"]) == K
    assert len(asset["edges"]) == len(lattice_edges(GRID))
    # node 0 has hits 0 -> dead; node 1 alive.
    assert asset["nodes"][0]["dead"] is True
    assert asset["nodes"][1]["dead"] is False
    assert asset["dead_neurons"] == 1
    # every edge weight is a similarity in (0, 1].
    assert all(0.0 < e[2] <= 1.0 for e in asset["edges"])
    # provenance carried verbatim.
    assert asset["provenance"]["dataset"] == "ham10000"


def test_build_asset_json_serializable():
    import json

    w = _synthetic_weights()
    asset = build_som_graph_asset(w, GRID, hits=hit_counts(np.arange(K), K))
    # Must round-trip through JSON with no numpy scalars leaking.
    reparsed = json.loads(json.dumps(asset))
    assert reparsed["num_neurons"] == K


def test_build_asset_hits_length_validated():
    w = _synthetic_weights()
    with pytest.raises(ValueError):
        build_som_graph_asset(w, GRID, hits=np.zeros(K + 2))


# --------------------------------------------------------------------------- #
# provider surface
# --------------------------------------------------------------------------- #


def test_provider_surface_matches_builder():
    w = _synthetic_weights()
    hits = hit_counts(np.arange(K), K)
    state = SomState(w, GRID, hits=hits)
    provider = SomGraphProvider(k=6, seed=0)
    nodes = provider.nodes(state)
    edges = provider.edges(state)
    assert len(nodes) == K
    assert len(edges) == len(lattice_edges(GRID))
    comm = provider.communities(state)
    assert comm.shape == (K,)
    assert np.array_equal(comm, som_communities(w, 6, seed=0))


# --------------------------------------------------------------------------- #
# PackWriter / PackReader round-trip (both SGP assets)
# --------------------------------------------------------------------------- #


def test_pack_roundtrip_som_assets(tmp_path):
    from vitreous.packs.writer import PackWriter

    w = _synthetic_weights()
    # one image's bmu map
    vol = np.random.default_rng(5).standard_normal((16, 16, 8, C))
    bm = bmu_map(vol, w)
    hits = hit_counts(bm, K)
    asset = build_som_graph_asset(
        w, GRID, hits=hits, depth_steps=8, volume_grid=(16, 16)
    )

    writer = PackWriter(tmp_path)
    writer.add_json("som.json", asset)
    entry = writer.add_array("som_bmu.bin", bm, encoding="raw", dtype="uint16")
    assert entry.dtype == "uint16"
    assert entry.shape == [8, 16, 16]

    # read the raw bmu map straight back (no manifest needed for add_array bytes).
    blob = (tmp_path / "som_bmu.bin").read_bytes()
    restored = np.frombuffer(blob, dtype=np.uint16).reshape(8, 16, 16)
    assert np.array_equal(restored, bm)

    import json

    reloaded = json.loads((tmp_path / "som.json").read_text())
    assert reloaded["num_neurons"] == K
    assert reloaded["depth_steps"] == 8
