"""U4 loss-suite tests (ARCHITECTURE §3.6-§3.8).

CPU-only and fast: tiny synthetic volumes/embeddings. Covers each objective term
plus the weighted composer:

- NT-Xent is minimised when paired views coincide;
- smoothness is axes-configurable — adding the ``z`` axis changes the value;
- ordering is non-negative; the monotone-centroid term is ``≥ 0`` and exactly
  ``0`` on a constructed depth-ordered (coarser-with-depth) volume;
- geodesic returns a scalar and back-propagates through the path edge lengths;
- ``total_loss`` weights, sums, and reports detached floats (including 0-weight
  drop-through).
"""

from __future__ import annotations

import math

import pytest
import torch

from umtvit.losses import (
    geodesic_loss,
    monotone_centroid_loss,
    nt_xent,
    ordering_loss,
    smoothness_loss,
    total_loss,
)


# --------------------------------------------------------------------------- #
# NT-Xent
# --------------------------------------------------------------------------- #
def test_ntxent_scalar_and_nonnegative():
    torch.manual_seed(0)
    za = torch.randn(8, 16)
    zb = torch.randn(8, 16)
    loss = nt_xent(za, zb, tau=0.2)
    assert loss.ndim == 0
    assert float(loss) >= 0.0


def test_ntxent_lower_for_aligned_views():
    torch.manual_seed(0)
    za = torch.randn(8, 16)
    aligned = nt_xent(za, za.clone(), tau=0.2)
    scrambled = nt_xent(za, torch.randn(8, 16), tau=0.2)
    assert float(aligned) < float(scrambled)


# --------------------------------------------------------------------------- #
# Smoothness — axes configurable
# --------------------------------------------------------------------------- #
def test_smoothness_axes_change_value():
    torch.manual_seed(0)
    volume = torch.randn(2, 4, 4, 3, 8)
    hw = float(smoothness_loss(volume, ("h", "w")))
    hwz = float(smoothness_loss(volume, ("h", "w", "z")))
    assert hw != hwz, "adding the z axis must change the smoothness value"


def test_smoothness_single_axis_matches_manual():
    torch.manual_seed(1)
    volume = torch.randn(2, 4, 4, 3, 8)
    manual = volume.diff(dim=1).pow(2).mean()  # h axis is tensor dim 1
    assert float(smoothness_loss(volume, ("h",))) == pytest.approx(float(manual))


def test_smoothness_rejects_empty_and_unknown_axes():
    volume = torch.randn(1, 4, 4, 2, 4)
    with pytest.raises(ValueError):
        smoothness_loss(volume, ())
    with pytest.raises(ValueError):
        smoothness_loss(volume, ("h", "t"))


# --------------------------------------------------------------------------- #
# Ordering + monotone centroid
# --------------------------------------------------------------------------- #
def test_ordering_loss_nonnegative_scalar():
    torch.manual_seed(0)
    volume = torch.randn(2, 8, 8, 4, 6)
    loss = ordering_loss(volume, fmax=0.5)
    assert loss.ndim == 0
    assert float(loss) >= 0.0


def test_monotone_centroid_nonnegative():
    torch.manual_seed(0)
    volume = torch.randn(2, 8, 8, 4, 3)
    assert float(monotone_centroid_loss(volume)) >= 0.0


def _depth_ordered_volume() -> torch.Tensor:
    """A volume whose slices get strictly coarser (lower frequency) with depth.

    Each Z-slice is a single spatial sinusoid whose frequency *decreases* with
    depth, so the per-slice spectral centroid is monotone non-increasing and the
    monotone-centroid penalty must be exactly 0.
    """
    height = width = 8
    frequencies = [3, 2, 1, 0]  # shallow → deep: high → low spatial frequency
    xs = torch.arange(width).float()
    slices = []
    for freq in frequencies:
        plane = torch.cos(2 * math.pi * freq * xs / width)[None, :].repeat(height, 1)
        slices.append(plane[None, :, :, None])  # [1, H, W, 1]
    return torch.stack(slices, dim=3)  # [1, H, W, L, 1]


def test_monotone_centroid_zero_on_depth_ordered_volume():
    volume = _depth_ordered_volume()
    assert float(monotone_centroid_loss(volume)) == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Geodesic
# --------------------------------------------------------------------------- #
def test_geodesic_scalar_and_backprops():
    torch.manual_seed(0)
    za = torch.randn(8, 16, requires_grad=True)
    zb = torch.randn(8, 16, requires_grad=True)
    v_sub = torch.randn(300, 16)
    loss = geodesic_loss(v_sub, za, zb)
    assert loss.ndim == 0
    loss.backward()
    assert za.grad is not None and float(za.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# Composer
# --------------------------------------------------------------------------- #
def test_total_loss_weights_and_reports():
    terms = {
        "ntxent": torch.tensor(2.0),
        "som": torch.tensor(4.0),
        "geodesic": torch.tensor(9.0),
    }
    weights = {"ntxent": 1.0, "som": 0.5, "geodesic": 0.0}
    total, detached = total_loss(terms, weights)
    assert float(total) == pytest.approx(1.0 * 2.0 + 0.5 * 4.0 + 0.0 * 9.0)
    assert detached["ntxent"] == pytest.approx(2.0)
    assert detached["geodesic"] == pytest.approx(9.0)  # raw value, unweighted
    assert detached["total"] == pytest.approx(4.0)


def test_total_loss_gradient_path():
    x = torch.tensor(3.0, requires_grad=True)
    total, _ = total_loss({"a": x * x}, {"a": 2.0})
    total.backward()
    assert float(x.grad) == pytest.approx(2.0 * 2.0 * 3.0)  # d/dx 2x^2 = 4x


def test_total_loss_empty_rejected():
    with pytest.raises(ValueError):
        total_loss({}, {})
