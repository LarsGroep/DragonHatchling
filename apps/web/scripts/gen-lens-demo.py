#!/usr/bin/env python3
"""Generate apps/web/public/lens/demo.json — the /lens demo fixture.

Run (needs packages/core installed, numpy only — no torch):

    python apps/web/scripts/gen-lens-demo.py

Produced through the SAME tested code path a real pack uses
(`vitreous.malignancy` + the HAM10000 `Taxonomy` on the DatasetSpec), so the
demo bundle and a real lens export are structurally identical — only the
softmax/feature values here are synthetic. It plants class-specific feature
centroids so the benign↔malignant axis separates cleanly, gives each example a
peaked-on-truth softmax, and includes one deliberately OFF-manifold example (a
"phone photo" stand-in) so the OOD refusal renders. Thumbnails are tiny gradient
PNGs written with the stdlib (zlib/struct) — no PIL.
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

from vitreous.data import get_dataset  # noqa: E402
from vitreous.malignancy import (  # noqa: E402
    build_malignancy_axis,
    malignant_indices,
)

SEED = 7
DIM = 32  # small stand-in for the real 384-d CLS (parser is dim-agnostic)


def _png_b64(rgb, size: int = 96) -> str:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    r0, g0, b0 = rgb
    rows = []
    for y in range(size):
        row = bytearray(b"\x00")
        for x in range(size):
            fx, fy = x / (size - 1), y / (size - 1)
            row += bytes((
                int(r0 * (0.45 + 0.55 * fx)) & 255,
                int(g0 * (0.45 + 0.55 * (1 - fy))) & 255,
                int(b0 * (0.5 + 0.5 * fy)) & 255,
            ))
        rows.append(bytes(row))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(
        b"IDAT", zlib.compress(b"".join(rows), 9)
    ) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def main() -> None:
    spec = get_dataset("ham10000").spec
    classes = list(spec.class_names)
    tax = spec.taxonomy
    K = len(classes)
    rng = np.random.default_rng(SEED)
    mal_idx = malignant_indices(classes, tax)
    mal_set = set(mal_idx)

    # Class-specific feature centroids arranged so malignant classes sit toward
    # +axis and benign toward -axis (a clean, separable manifold to render).
    axis_dir = rng.standard_normal(DIM)
    axis_dir /= np.linalg.norm(axis_dir)
    centroids = np.zeros((K, DIM))
    for c in range(K):
        base = rng.standard_normal(DIM) * 0.6
        sign = 1.0 if c in mal_set else -1.0
        centroids[c] = base + sign * 2.4 * axis_dir

    # A labelled training pool → the axis (features + malignant flags).
    pool_f, pool_m = [], []
    for c in range(K):
        for _ in range(40):
            pool_f.append(centroids[c] + rng.standard_normal(DIM) * 0.35)
            pool_m.append(c in mal_set)
    axis = build_malignancy_axis(
        np.asarray(pool_f), pool_m, space="cls_final",
        provenance={"dataset": "ham10000-demo", "note": "synthetic axis"},
    )

    # Example lesions: one clear case per class (peaked softmax on its dx), plus
    # a couple of ambiguous ones, plus a deliberate OFF-manifold "phone" example.
    hues = [
        (225, 29, 72), (147, 51, 234), (234, 179, 8), (59, 130, 246),
        (14, 165, 233), (34, 197, 94), (244, 114, 182),
    ]
    lesions = []

    def _softmax_peaked(true_c, sharp=6.0):
        logits = rng.standard_normal(K) * 0.4
        logits[true_c] += sharp
        e = np.exp(logits - logits.max())
        return (e / e.sum()).tolist()

    for c in range(K):
        feat = centroids[c] + rng.standard_normal(DIM) * 0.3
        lesions.append({
            "id": f"case_{classes[c].split()[0].lower()}_{c}",
            "thumb_png_b64": _png_b64(hues[c % len(hues)]),
            "true_label": classes[c],
            "probabilities": _softmax_peaked(c),
            "feature": [float(x) for x in feat],
        })

    # An ambiguous nevus-vs-melanoma case (mass split) → mid category coordinate.
    mel = classes.index("Melanoma")
    nv = classes.index("Melanocytic nevi")
    amb = np.zeros(K)
    amb[mel] = 0.46
    amb[nv] = 0.44
    amb[classes.index("Benign keratosis")] = 0.10
    lesions.append({
        "id": "case_ambiguous",
        "thumb_png_b64": _png_b64((120, 80, 70)),
        "true_label": None,
        "probabilities": amb.tolist(),
        "feature": [float(x) for x in 0.5 * (centroids[mel] + centroids[nv])],
    })

    # An OFF-manifold "phone photo" stand-in: feature flung orthogonal to the
    # axis → the OOD refusal should fire (honest gate for out-of-distribution).
    off = centroids[nv].copy()
    perp = rng.standard_normal(DIM)
    perp -= perp.dot(np.asarray(axis["u"])) * np.asarray(axis["u"])
    off += 12.0 * perp / np.linalg.norm(perp)
    lesions.append({
        "id": "case_offmanifold_phone",
        "thumb_png_b64": _png_b64((90, 90, 90)),
        "true_label": None,
        "probabilities": _softmax_peaked(nv, sharp=3.0),
        "feature": [float(x) for x in off],
    })

    bundle = {
        "lens_schema_version": 1,
        "dataset": "ham10000",
        "class_names": classes,
        "taxonomy": tax.to_json(),
        "axis": axis,
        "lesions": lesions,
        "provenance": {"generator": "apps/web/scripts/gen-lens-demo.py", "seed": SEED},
    }

    out = Path(__file__).resolve().parents[1] / "public" / "lens" / "demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, separators=(",", ":")))
    print(
        f"wrote {out} ({out.stat().st_size/1e6:.2f} MB) — {K} classes, "
        f"{len(lesions)} lesions, axis dim {axis['dim']}, "
        f"malignant={sorted(n for n,m in tax.malignant.items() if m)}"
    )


if __name__ == "__main__":
    main()
