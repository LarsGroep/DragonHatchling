"""Gaussian Feature Field (§7) — the flagship lens.

**Honesty rule:** every Gaussian parameter is a deterministic function of a
measured quantity — geometry from patch positions, appearance from image
statistics, dynamics from model internals. No fitted structure the model does
not use; the frontend labels the view as a *lens*, not an attribution method.

Layout produced by :func:`build_gaussian_field`
-----------------------------------------------
A packed ``[S=13][N=197][C=12]`` ``float16`` array, C-order, serialized to
``gaussians.bin``. The timeline has ``S = L+1 = 13`` snapshots (``t = 0..12``)
matching the 13 token snapshots; ``N = 197`` tokens (CLS + 196 patches); the 12
channels, in exact order, are::

    0  x               patch-center x, normalized [0,1]
    1  y               patch-center y, normalized [0,1]
    2  rx              semi-axis along the ellipse major direction, [0,1]
    3  ry              semi-axis along the ellipse minor direction, [0,1]
    4  theta           ellipse orientation (rad), atan2 -> [-pi, pi]
    5  r               base red,   mean patch color, [0,1]
    6  g               base green, mean patch color, [0,1]
    7  b               base blue,  mean patch color, [0,1]
    8  opacity         globally-normalized activation magnitude ||token_t||, [0,1]
    9  glow            emissive glow = normalized attribution (chefer|rollout), [0,1]
    10 halo            attention received (column-sum), globally normalized, [0,1]
    11 activation_raw  raw ||token_t|| L2 magnitude (fp16-clamped, >= 0)

All values are fp16-safe and normalized to the documented ranges ([0,1] except
``theta``). The channel order is also recorded on the pack manifest asset entry
(``assets["gaussians.bin"].meta["channels"]``) so the frontend never hard-codes it.

Derivation (per token *i*, per step *t*)
----------------------------------------
* **center (x, y)** — fixed patch center. Patches ``1..196`` map to the canonical
  14x14 grid via ``divmod(i-1, 14)`` -> ``((col+0.5)/14, (row+0.5)/14)``. The CLS
  token (index 0) is not spatial; it is assigned a **reserved off-grid anchor** at
  normalized ``(0.0, 0.0)`` (top-left corner), distinct from every patch center
  (which lie in ``[0.0357, 0.9643]``). Documented and constant across *t*.
* **covariance (rx, ry, theta)** — base is an isotropic disc of half-patch radius
  ``r0 = 0.5/14``. It is anisotropically stretched toward the token's dominant
  attention neighbors at layer *t*: with head-averaged attention row ``a_i`` over
  the 196 patch tokens, the mean displacement ``v_i = sum_j a_ij (c_j - c_i)``
  gives orientation ``theta = atan2(v_y, v_x)`` and a concentration
  ``rho = clip(|v_i| / V_SCALE, 0, 1)``. The axis ratio is
  ``ratio = 1 + (ECC_MAX-1)*rho`` in ``[1, ECC_MAX]`` with ``ECC_MAX = 2.5``
  (**bounded eccentricity**), realized area-preservingly as
  ``rx = r0*sqrt(ratio)``, ``ry = r0/sqrt(ratio)`` so ``rx/ry = ratio <= 2.5``.
  CLS is kept isotropic (no spatial neighbors).
* **base RGB** — mean color of the patch's pixels from the **unnormalized** image,
  scaled to ``[0,1]``. CLS gets the whole-image mean. Constant across *t*.
* **opacity** — ``||token_t[i]||`` L2 magnitude, normalized by the global maximum
  over all ``(t, i)`` so temporal growth ("diffusion of importance") stays visible.
* **glow** — attribution: per-layer Chefer relevance when ``chefer_layers`` is
  supplied, else attention rollout computed from the trace. Normalized by its
  global max over all layers/tokens.
* **halo** — attention received: column-sum ``sum_j a_ji`` of the layer's
  head-averaged attention, globally normalized.

Attention indexing: for ``t in 1..12`` the attention/attribution of **layer
t-1** is used (there are 12 attention layers but 13 token snapshots). Step
``t = 0`` has *no* attention yet, so all attention-derived channels
(rx/ry stretch, theta, glow, halo) take their **neutral** values (isotropic
``r0``, ``theta = 0``, ``glow = 0``, ``halo = 0``); opacity/activation_raw are
still defined at ``t = 0`` because the block-0 input tokens exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Canonical geometry (see DECISION-LOG "Canonical grid mapping").
GRID = 14
PATCH = 16
N_TOKENS = 197
N_STEPS = 13
CLS_INDEX = 0
CLS_POSITION: Tuple[float, float] = (0.0, 0.0)  # reserved off-grid anchor
R0 = 0.5 / GRID                                  # base (half-patch) radius
ECC_MAX = 2.5                                    # bounded eccentricity (max rx/ry)
V_SCALE = 0.35                                   # displacement -> concentration scale
FP16_MAX = 65504.0

CHANNELS: Tuple[str, ...] = (
    "x",
    "y",
    "rx",
    "ry",
    "theta",
    "r",
    "g",
    "b",
    "opacity",
    "glow",
    "halo",
    "activation_raw",
)
N_CHANNELS = len(CHANNELS)
_CH = {name: i for i, name in enumerate(CHANNELS)}


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


@dataclass
class GaussianField:
    """Packed Gaussian Feature Field: ``[S, N, C]`` fp16 plus provenance.

    ``data`` is the exact array written to ``gaussians.bin``. ``channels`` is the
    per-channel name tuple (== :data:`CHANNELS`); ``norm`` records the global
    normalization constants so the frontend can recover raw magnitudes.
    """

    data: np.ndarray                       # [13, 197, 12] float16, C-order
    channels: Tuple[str, ...] = CHANNELS
    n_steps: int = N_STEPS
    n_tokens: int = N_TOKENS
    grid: int = GRID
    cls_index: int = CLS_INDEX
    cls_position: Tuple[float, float] = CLS_POSITION
    ecc_max: float = ECC_MAX
    attribution: str = "rollout"
    norm: Dict[str, float] = field(default_factory=dict)

    def channel(self, name: str) -> np.ndarray:
        """Return the ``[S, N]`` slice for channel ``name`` as float32."""
        return self.data[:, :, _CH[name]].astype(np.float32)

    def to_meta(self) -> Dict[str, Any]:
        """Manifest ``asset.meta`` payload documenting the binary's semantics."""
        return {
            "channels": list(self.channels),
            "layout": "S=13,N=197,C=12 float16 C-order",
            "n_steps": self.n_steps,
            "n_tokens": self.n_tokens,
            "grid": self.grid,
            "cls_index": self.cls_index,
            "cls_position": list(self.cls_position),
            "ecc_max": self.ecc_max,
            "theta_range": "radians [-pi, pi]",
            "attribution": self.attribution,
            "norm": self.norm,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _to_numpy(x: Any) -> np.ndarray:
    """Detach + numpy-ify a torch tensor or pass numpy through (no torch import)."""
    if x is None:
        raise ValueError("expected an array, got None")
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    return np.asarray(x)


def _image_rgb01(image: Any) -> np.ndarray:
    """Normalize an unnormalized image to ``[H, W, 3]`` float32 in ``[0,1]``.

    Accepts ``[H,W,3]`` (numpy/torch, uint8 or float) or ``[3,H,W]`` torch/numpy.
    uint8 is divided by 255; float inputs already in a small range are used as-is
    (values > 1.5 are treated as 0..255 and rescaled).
    """
    a = _to_numpy(image)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
    a = a.astype(np.float32)
    if a.dtype == np.uint8 or a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def _patch_centers() -> np.ndarray:
    """``[197, 2]`` normalized patch centers; row 0 = CLS reserved anchor."""
    centers = np.zeros((N_TOKENS, 2), dtype=np.float64)
    centers[CLS_INDEX] = CLS_POSITION
    for i in range(1, N_TOKENS):
        row, col = divmod(i - 1, GRID)
        centers[i, 0] = (col + 0.5) / GRID
        centers[i, 1] = (row + 0.5) / GRID
    return centers


def _patch_mean_rgb(img01: np.ndarray) -> np.ndarray:
    """``[197, 3]`` mean patch color in ``[0,1]``; row 0 = whole-image mean."""
    H, W = img01.shape[:2]
    ph, pw = H // GRID, W // GRID
    rgb = np.zeros((N_TOKENS, 3), dtype=np.float64)
    rgb[CLS_INDEX] = img01.reshape(-1, img01.shape[2])[:, :3].mean(axis=0)
    for i in range(1, N_TOKENS):
        row, col = divmod(i - 1, GRID)
        block = img01[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw, :3]
        rgb[i] = block.reshape(-1, 3).mean(axis=0)
    return np.clip(rgb, 0.0, 1.0)


def _rollout_from_trace(attn: np.ndarray, residual_ratio: float = 0.5) -> np.ndarray:
    """Per-layer CLS-row attention rollout ``[L, T]`` (numpy mirror of xai.rollout)."""
    L, H, T, _ = attn.shape
    eye = np.eye(T, dtype=np.float64)
    cumulative: Optional[np.ndarray] = None
    rows = np.zeros((L, T), dtype=np.float64)
    for layer in range(L):
        a = attn[layer].mean(axis=0)
        a = residual_ratio * a + (1.0 - residual_ratio) * eye
        a = a / a.sum(axis=-1, keepdims=True)
        cumulative = a if cumulative is None else a @ cumulative
        rows[layer] = cumulative[0]
    return rows


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #


def build_gaussian_field(
    trace: Any,
    image: Any,
    chefer_layers: Any = None,
) -> GaussianField:
    """Derive the ``[13, 197, 12]`` Gaussian Feature Field from a trace + image (§7).

    Parameters
    ----------
    trace:
        A :class:`~vitreous.instrument.Trace` with ``attention`` ``[L,H,T,T]`` and
        ``tokens`` ``[L+1,T,D]`` (torch or numpy).
    image:
        The **unnormalized** display image (``[H,W,3]`` uint8/float or ``[3,H,W]``)
        — used only for base RGB.
    chefer_layers:
        Optional per-layer attribution ``[L, T]`` (e.g. ``chefer_relevance(...)
        .token_scores``) driving the emissive glow. When ``None`` the glow uses
        attention rollout derived from the trace.

    Returns
    -------
    GaussianField
        Wrapping the packed fp16 array (``.data``) plus channel names and the
        global normalization constants used.
    """
    attn = _to_numpy(trace.attention).astype(np.float64)   # [L,H,T,T]
    tokens = _to_numpy(trace.tokens).astype(np.float64)    # [L+1,T,D]
    if attn.ndim != 4:
        raise ValueError(f"expected attention [L,H,T,T], got {attn.shape}")
    if tokens.ndim != 3:
        raise ValueError(f"expected tokens [L+1,T,D], got {tokens.shape}")

    L, Hh, T, _ = attn.shape
    S = tokens.shape[0]           # L+1
    N = tokens.shape[1]           # T
    if N != N_TOKENS or T != N_TOKENS or S != N_STEPS:
        # Stay general but keep the documented ViT-S/16 defaults as the contract.
        pass

    centers = _patch_centers()[:N]
    img01 = _image_rgb01(image)
    rgb = _patch_mean_rgb(img01)[:N]

    # Head-averaged attention per layer: [L, T, T].
    abar = attn.mean(axis=1)

    # Attribution for glow: per-layer [L, T].
    if chefer_layers is not None:
        attr = _to_numpy(chefer_layers).astype(np.float64)
        if attr.ndim == 1:
            attr = np.broadcast_to(attr, (L, attr.shape[0])).copy()
        attribution_name = "chefer"
    else:
        attr = _rollout_from_trace(attn)
        attribution_name = "rollout"

    # Activation magnitudes ||token_t[i]|| : [S, N].
    mags = np.linalg.norm(tokens, axis=-1)          # [S, N]
    mag_max = float(mags.max()) if mags.size else 1.0
    mag_max = mag_max if mag_max > 0 else 1.0

    # Global attribution / halo maxima (over the 12 attention layers).
    attr_max = float(np.abs(attr).max()) if attr.size else 1.0
    attr_max = attr_max if attr_max > 0 else 1.0
    colsums = abar.sum(axis=1)                       # [L, T] : attention received
    halo_max = float(colsums.max()) if colsums.size else 1.0
    halo_max = halo_max if halo_max > 0 else 1.0

    out = np.zeros((S, N, N_CHANNELS), dtype=np.float64)

    # Static channels (constant across t): center + RGB.
    out[:, :, _CH["x"]] = centers[None, :, 0]
    out[:, :, _CH["y"]] = centers[None, :, 1]
    out[:, :, _CH["r"]] = rgb[None, :, 0]
    out[:, :, _CH["g"]] = rgb[None, :, 1]
    out[:, :, _CH["b"]] = rgb[None, :, 2]

    # Activation-derived channels (defined for every t, incl. t=0).
    out[:, :, _CH["opacity"]] = np.clip(mags / mag_max, 0.0, 1.0)
    out[:, :, _CH["activation_raw"]] = np.clip(mags, 0.0, FP16_MAX)

    # Neutral defaults for the geometry (t=0 and CLS keep these).
    out[:, :, _CH["rx"]] = R0
    out[:, :, _CH["ry"]] = R0
    out[:, :, _CH["theta"]] = 0.0

    patch_slice = slice(1, N)     # patch tokens only (exclude CLS)
    disp = centers[patch_slice][None, :, :] - centers[patch_slice][:, None, :]
    # disp[j_from_i]: displacement c_j - c_i for patch tokens; shape [P, P, 2].

    for t in range(1, S):
        layer = t - 1             # step t uses layer t-1's attention
        a_layer = abar[layer]     # [T, T]

        # --- covariance stretch toward dominant attention neighbors --------- #
        w = a_layer[patch_slice, patch_slice]        # [P, P] patch->patch weights
        # mean displacement v_i = sum_j w_ij (c_j - c_i)
        v = np.einsum("ij,ijd->id", w, disp)         # [P, 2]
        vmag = np.linalg.norm(v, axis=1)             # [P]
        rho = np.clip(vmag / V_SCALE, 0.0, 1.0)
        ratio = 1.0 + (ECC_MAX - 1.0) * rho          # [1, ECC_MAX]
        sqrt_ratio = np.sqrt(ratio)
        theta = np.arctan2(v[:, 1], v[:, 0])
        # only orient where there is a meaningful direction
        theta = np.where(vmag > 1e-9, theta, 0.0)

        out[t, patch_slice, _CH["rx"]] = R0 * sqrt_ratio
        out[t, patch_slice, _CH["ry"]] = R0 / sqrt_ratio
        out[t, patch_slice, _CH["theta"]] = theta

        # --- glow (attribution) --------------------------------------------- #
        out[t, :, _CH["glow"]] = np.clip(attr[layer] / attr_max, 0.0, 1.0)

        # --- halo (attention received) -------------------------------------- #
        out[t, :, _CH["halo"]] = np.clip(colsums[layer] / halo_max, 0.0, 1.0)

    data = out.astype(np.float16)
    return GaussianField(
        data=data,
        channels=CHANNELS,
        n_steps=S,
        n_tokens=N,
        grid=GRID,
        cls_index=CLS_INDEX,
        cls_position=CLS_POSITION,
        ecc_max=ECC_MAX,
        attribution=attribution_name,
        norm={
            "activation_max": mag_max,
            "attribution_max": attr_max,
            "halo_max": halo_max,
            "v_scale": V_SCALE,
        },
    )


__all__ = [
    "Gaussian",
    "GaussianField",
    "build_gaussian_field",
    "CHANNELS",
    "GRID",
    "CLS_INDEX",
    "CLS_POSITION",
    "ECC_MAX",
]
