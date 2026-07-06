"""Gaussian Feature Field (§7) — the flagship lens.

**Honesty rule:** every Gaussian parameter is a deterministic function of a
measured quantity (geometry from patch positions, appearance from image
statistics, dynamics from model internals). No fitted structure the model does
not use.

Per token *i*, per timeline step *t*, a :class:`Gaussian` bundles center,
covariance, base RGB, opacity, emissive glow, and halo radius. M0 ships the
dataclass and the builder signature; the builder lands at M3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Gaussian:
    """One anisotropic Gaussian for token *i* at timeline step *t* (§7)."""

    x: float
    y: float
    rx: float
    ry: float
    theta: float
    r: float
    g: float
    b: float
    a: float          # opacity: normalized activation magnitude
    activation: float
    attention_in: float
    attribution: float


def build_gaussian_field(trace: Any, image: Any) -> Any:
    """Derive the ``[L+1][N]`` Gaussian field from a trace + image.

    Returns a packed array matching ``gaussians.bin`` (§5). Not implemented at
    M0 — lands at M3.
    """

    raise NotImplementedError("build_gaussian_field lands at M3")


__all__ = ["Gaussian", "build_gaussian_field"]
