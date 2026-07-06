"""Gaussian Feature Field builder (§7) — offline, synthetic, torch-free.

Uses a synthetic row-stochastic attention tensor + random tokens + a random
uint8 image (no model, no download). Covers the exact packed shape, value
ranges, CLS handling, the bounded-eccentricity guarantee, the t=0 neutral-
attention convention, fp16 round-trip, and determinism.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vitreous.gaussians import (
    CHANNELS,
    CLS_INDEX,
    CLS_POSITION,
    ECC_MAX,
    GRID,
    build_gaussian_field,
)

R0 = 0.5 / GRID
CH = {name: i for i, name in enumerate(CHANNELS)}


def _synthetic_trace(seed: int = 0):
    rng = np.random.default_rng(seed)
    raw = rng.random((12, 6, 197, 197)).astype(np.float32)
    attn = raw / raw.sum(axis=-1, keepdims=True)      # row-stochastic softmax rows
    tokens = rng.standard_normal((13, 197, 384)).astype(np.float32)
    return SimpleNamespace(attention=attn, tokens=tokens)


def _image(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((224, 224, 3)) * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def field():
    return build_gaussian_field(_synthetic_trace(0), _image())


def test_exact_shape_and_dtype(field):
    assert field.data.shape == (13, 197, 12)
    assert field.data.dtype == np.float16
    assert field.channels == CHANNELS
    assert len(CHANNELS) == 12


def test_channel_order_is_exact(field):
    assert CHANNELS == (
        "x", "y", "rx", "ry", "theta", "r", "g", "b",
        "opacity", "glow", "halo", "activation_raw",
    )
    # And the manifest meta advertises the same order.
    assert field.to_meta()["channels"] == list(CHANNELS)


def test_value_ranges(field):
    f = field.data.astype(np.float32)
    # [0,1] channels
    for name in ("x", "y", "rx", "ry", "r", "g", "b", "opacity", "glow", "halo"):
        c = f[:, :, CH[name]]
        assert c.min() >= -1e-3, name
        assert c.max() <= 1.0 + 1e-3, name
    # theta in [-pi, pi]
    theta = f[:, :, CH["theta"]]
    assert theta.min() >= -np.pi - 1e-2
    assert theta.max() <= np.pi + 1e-2
    # activation_raw non-negative and fp16-safe
    act = f[:, :, CH["activation_raw"]]
    assert act.min() >= 0.0
    assert act.max() < 65504.0


def test_cls_handling(field):
    f = field.data.astype(np.float32)
    # CLS is at the reserved off-grid anchor for every timeline step.
    assert CLS_POSITION == (0.0, 0.0)
    assert np.allclose(f[:, CLS_INDEX, CH["x"]], CLS_POSITION[0])
    assert np.allclose(f[:, CLS_INDEX, CH["y"]], CLS_POSITION[1])
    # CLS stays isotropic (no spatial neighbors).
    assert np.allclose(f[:, CLS_INDEX, CH["rx"]], R0, atol=1e-3)
    assert np.allclose(f[:, CLS_INDEX, CH["ry"]], R0, atol=1e-3)
    # Patch centers are all on-grid in [0.0357, 0.9643], distinct from CLS.
    px = f[3, 1:, CH["x"]]
    assert px.min() >= 0.5 / GRID - 1e-3


def test_patch_center_mapping(field):
    # Token i (1..196) center = ((col+0.5)/14, (row+0.5)/14), row,col=divmod(i-1,14).
    f = field.data.astype(np.float32)
    for i in (1, 15, 100, 196):
        row, col = divmod(i - 1, GRID)
        assert abs(f[5, i, CH["x"]] - (col + 0.5) / GRID) < 2e-3
        assert abs(f[5, i, CH["y"]] - (row + 0.5) / GRID) < 2e-3


def test_eccentricity_bounded(field):
    f = field.data.astype(np.float32)
    rx = f[:, :, CH["rx"]]
    ry = f[:, :, CH["ry"]]
    ratio = rx / np.clip(ry, 1e-6, None)
    # Max axis ratio must not exceed ECC_MAX (== 2.5), with a small fp16 margin.
    assert ratio.max() <= ECC_MAX + 1e-2
    assert ratio.min() >= 1.0 - 1e-2


def test_t0_neutral_attention(field):
    f = field.data.astype(np.float32)
    # Step 0 has no attention: geometry isotropic, glow/halo zero.
    assert np.allclose(f[0, :, CH["rx"]], R0, atol=1e-3)
    assert np.allclose(f[0, :, CH["ry"]], R0, atol=1e-3)
    assert np.allclose(f[0, :, CH["theta"]], 0.0, atol=1e-3)
    assert np.allclose(f[0, :, CH["glow"]], 0.0, atol=1e-3)
    assert np.allclose(f[0, :, CH["halo"]], 0.0, atol=1e-3)
    # But activation-derived channels are defined at t=0 (block-0 tokens exist).
    assert f[0, :, CH["activation_raw"]].max() > 0.0


def test_static_channels_constant_over_time(field):
    f = field.data.astype(np.float32)
    for name in ("x", "y", "r", "g", "b"):
        c = f[:, :, CH[name]]
        assert np.allclose(c, c[0][None, :], atol=1e-3), name


def test_fp16_roundtrip(field):
    # Data is already fp16; bytes -> array must reproduce it exactly.
    buf = np.ascontiguousarray(field.data, dtype=np.float16).tobytes()
    back = np.frombuffer(buf, dtype=np.float16).reshape(13, 197, 12)
    assert np.array_equal(back, field.data)


def test_determinism():
    a = build_gaussian_field(_synthetic_trace(0), _image())
    b = build_gaussian_field(_synthetic_trace(0), _image())
    assert np.array_equal(a.data, b.data)


def test_chefer_layers_drive_glow():
    trace = _synthetic_trace(2)
    # A per-layer attribution [12,197] with a single hot token per layer.
    attr = np.zeros((12, 197), dtype=np.float32)
    attr[:, 42] = 1.0
    f = build_gaussian_field(trace, _image(), chefer_layers=attr)
    assert f.attribution == "chefer"
    g = f.data.astype(np.float32)[:, :, CH["glow"]]
    # Token 42 is the max-glow token at every t>=1 (normalized to 1.0).
    for t in range(1, 13):
        assert abs(g[t, 42] - 1.0) < 1e-2
    # Without chefer_layers the builder falls back to rollout.
    f2 = build_gaussian_field(trace, _image())
    assert f2.attribution == "rollout"
