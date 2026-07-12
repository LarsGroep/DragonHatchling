#!/usr/bin/env python3
"""Generate apps/web/public/sgp/demo.json — the /sgp demo fixture.

Run (needs packages/core installed, numpy only — no torch):

    python apps/web/scripts/gen-sgp-demo.py

The fixture is produced by the SAME tested code path a real Kaggle run uses
(`vitreous.som` → `build_som_graph_asset` / `bmu_map` / `hit_counts`), so the
demo bundle and a real `sgp_ham10000.json` are structurally identical — only
the SOM weights and volumes here are synthetic (a smooth lattice-correlated
field with planted clusters, plus a deliberate dead region so the dead-neuron
rendering is exercised). Thumbnails are tiny gradient PNGs written with the
stdlib (zlib/struct) — no PIL.
"""

from __future__ import annotations

import base64
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "core" / "src"))

from vitreous.som import (  # noqa: E402
    bmu_map,
    build_som_graph_asset,
    grid_coords,
    hit_counts,
    bmu_indices,
)

SEED = 7
GRID = (8, 8, 8)          # (Gz, Gy, Gx) — the HAM10000 preset lattice
K = 512
C = 24                     # feature width (small keeps the fixture light)
Z = 8                      # encoder depth
HC = WC = 16               # volume grid (H', W')
N_PROBES = 6
N_CLUSTERS = 7             # planted weight-space clusters


def _png_b64(rgb_fn, size: int = 96) -> str:
    """Encode a size×size RGB image (rgb_fn(x01, y01) -> (r,g,b)) as base64 PNG."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    rows = []
    for y in range(size):
        row = bytearray(b"\x00")
        for x in range(size):
            r, g, b = rgb_fn(x / (size - 1), y / (size - 1))
            row += bytes((int(r) & 255, int(g) & 255, int(b) & 255))
        rows.append(bytes(row))
    idat = zlib.compress(b"".join(rows), 9)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def main() -> None:
    rng = np.random.default_rng(SEED)
    coords = grid_coords(GRID).astype(np.float64)  # [K, 3]

    # ── synthetic SOM weights: smooth over the lattice + planted clusters ──
    # Smooth base field (topology-preserving look) …
    base = coords @ rng.standard_normal((3, C)) * 0.55
    # … plus a cluster identity per lattice region so communities/U-matrix pop.
    centers = coords[rng.choice(K, size=N_CLUSTERS, replace=False)]  # cluster seeds
    d = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(-1)    # [K, n]
    cluster = d.argmin(1)
    offsets = rng.standard_normal((N_CLUSTERS, C)) * 2.4
    weights = (base + offsets[cluster] + rng.standard_normal((K, C)) * 0.12).astype(
        np.float32
    )

    # ── synthetic probe volumes: voxels sampled near neurons along a path ──
    # Each probe walks a straight line through the LATTICE across depth; a
    # voxel at (y, x) draws its feature from a neuron near that depth's focus,
    # offset by its own image position (+ jitter). So the BMU replay visibly
    # MIGRATES across the map as depth advances, neighbouring voxels land on
    # neighbouring neurons (a topology-preserving look), and coverage is broad
    # — while the clusters never visited stay genuinely dead (exercising the
    # dead-neuron rendering with real zeros, not fabricated flags).
    gz, gy, gx = GRID
    gyx = gy * gx
    yy, xx = np.mgrid[0:HC, 0:WC]
    all_bmu = []
    for i in range(N_PROBES):
        start = rng.uniform([0, 0, 0], [gz - 1, gy - 1, gx - 1])
        end = rng.uniform([0, 0, 0], [gz - 1, gy - 1, gx - 1])
        vol = np.zeros((HC, WC, Z, C), dtype=np.float64)
        for z in range(Z):
            focus = start + (end - start) * (z / max(1, Z - 1))
            # image position nudges the lattice target (spatial coherence).
            ty = focus[1] + (yy / (HC - 1) - 0.5) * (gy * 0.8)
            tx = focus[2] + (xx / (WC - 1) - 0.5) * (gx * 0.8)
            tz = np.full_like(ty, focus[0])
            jit = rng.standard_normal((3, HC, WC)) * 0.5
            kz = np.clip(np.rint(tz + jit[0]), 0, gz - 1).astype(int)
            ky = np.clip(np.rint(ty + jit[1]), 0, gy - 1).astype(int)
            kx = np.clip(np.rint(tx + jit[2]), 0, gx - 1).astype(int)
            kk = kz * gyx + ky * gx + kx                     # [HC, WC] neuron ids
            vol[:, :, z, :] = weights[kk] + rng.standard_normal((HC, WC, C)) * 0.10
        all_bmu.append(bmu_map(vol.astype(np.float32), weights))

    hits = hit_counts(np.concatenate([b.ravel() for b in all_bmu]), K)

    som_asset = build_som_graph_asset(
        weights,
        GRID,
        hits=hits,
        community_k=N_CLUSTERS,
        seed=SEED,
        depth_steps=Z,
        volume_grid=(HC, WC),
        provenance={
            "dataset": "sgp-demo",
            "note": "synthetic fixture — generated by gen-sgp-demo.py through the "
            "same vitreous.som code path as a real Kaggle run",
        },
    )

    # ── thumbnails: distinct gradient placeholders per probe ──
    hues = [(66, 133, 244), (13, 148, 136), (245, 158, 11), (139, 92, 246), (34, 197, 94), (225, 29, 72)]

    def thumb(i):
        r0, g0, b0 = hues[i % len(hues)]
        return _png_b64(
            lambda x, y: (
                r0 * (0.35 + 0.65 * x),
                g0 * (0.35 + 0.65 * (1 - y)),
                b0 * (0.4 + 0.6 * y),
            )
        )

    bundle = {
        "sgp_schema_version": 1,
        "dataset": "sgp-demo",
        "som": som_asset,
        "probes": [
            {
                "index": i,
                "thumb_png_b64": thumb(i),
                "bmu": all_bmu[i].astype(int).tolist(),
            }
            for i in range(N_PROBES)
        ],
        "provenance": {"generator": "apps/web/scripts/gen-sgp-demo.py", "seed": SEED},
    }

    out = Path(__file__).resolve().parents[1] / "public" / "sgp" / "demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, separators=(",", ":")))
    dead = som_asset["dead_neurons"]
    print(
        f"wrote {out} ({out.stat().st_size/1e6:.2f} MB) — {K} neurons, "
        f"{len(som_asset['edges'])} edges, {dead} dead ({dead/K:.0%}), {N_PROBES} probes"
    )


if __name__ == "__main__":
    main()
