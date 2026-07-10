"""Augmentation policy registry + two-view transform (ARCHITECTURE §4).

Named, composable policies decide which ops apply, so a medical config can
exclude hue-destroying transforms while a satellite config keeps 90° rotations.
:func:`augment` applies a policy to a PIL image and returns a ``[C, H, W]``
float tensor in ``[0, 1]``: **geometric** ops run first on the PIL image
(crop, rotation, 90° rotation, flips), then the image is resized to the target
size and converted to a tensor for the **photometric** ops (brightness/
contrast, per-channel jitter, Gaussian noise). This ordering matches the
executable notebook reference.

Policies (ARCHITECTURE §4 — dermoscopy excludes hue/channel jitter):

- ``none`` — resize only (identity augmentation).
- ``natural_default`` — general photos: crop, hflip, mild rotation, colour.
- ``dermoscopy_default`` — dermoscopy: crop, both flips, full rotation, gentle
  brightness/contrast, **no** hue/channel jitter (colour is diagnostic).
- ``satellite_default`` — overhead imagery: crop, both flips, 90° rotations.

Torch and the op maths are imported lazily inside :func:`augment`, so
``import umtvit.data.augment`` (and reading the registry) stays torch-free.
"""

from __future__ import annotations

from typing import Dict

from ..config import ConfigError

__all__ = ["AUGMENTATION_POLICIES", "get_policy", "augment"]

# Each policy is a dict of enabled ops. Absent key ⇒ op disabled. Op semantics:
#   crop=(lo, hi)     random square area-fraction crop in [lo, hi]
#   rot=deg           random rotation in [-deg, deg]
#   rot90=True        random 0/90/180/270 rotation (75% of the time)
#   hflip/vflip=True  50% horizontal / vertical flip
#   bc=amt            brightness+contrast jitter magnitude
#   channel_jitter=a  per-channel gain jitter magnitude (hue-affecting)
#   noise=std         additive Gaussian noise std (post-normalisation)
AUGMENTATION_POLICIES: Dict[str, dict] = {
    "none": dict(),
    "natural_default": dict(
        crop=(0.6, 1.0), hflip=True, rot=15, bc=0.3, channel_jitter=0.10, noise=0.02
    ),
    "dermoscopy_default": dict(
        crop=(0.7, 1.0), hflip=True, vflip=True, rot=180, bc=0.2,
        channel_jitter=0.0, noise=0.01,  # no hue/channel ops: colour is diagnostic
    ),
    "satellite_default": dict(
        crop=(0.6, 1.0), hflip=True, vflip=True, rot90=True, bc=0.2,
        channel_jitter=0.05, noise=0.02,
    ),
}


def get_policy(name: str) -> dict:
    """Return the op dict for ``name`` or raise a field-named ConfigError.

    The policy name comes from ``cfg.dataset.augmentation``; an unknown name is
    a configuration error, reported against that field so it is diagnosable.
    """
    try:
        return AUGMENTATION_POLICIES[name]
    except KeyError:
        raise ConfigError(
            f"dataset.augmentation: unknown policy {name!r}; must be one of "
            f"{sorted(AUGMENTATION_POLICIES)}"
        )


def augment(img, size, policy, rng):
    """Augment a PIL ``img`` under ``policy`` → ``[C, H, W]`` float in ``[0, 1]``.

    Args:
        img: source PIL image (RGB).
        size: output side length in pixels.
        policy: an op dict from :data:`AUGMENTATION_POLICIES`.
        rng: a :class:`numpy.random.Generator` supplying all geometric/photometric
            randomness (fresh per view for two-view contrastive sampling).
    """
    import math

    import numpy as np
    import torch
    from PIL import Image

    # ---- geometric ops on the PIL image ---------------------------------- #
    if policy.get("crop"):
        area = rng.uniform(*policy["crop"])
        side = max(8, int(img.width * math.sqrt(area)))
        side = min(side, img.width, img.height)
        x0 = int(rng.integers(0, img.width - side + 1))
        y0 = int(rng.integers(0, img.height - side + 1))
        img = img.crop((x0, y0, x0 + side, y0 + side))
    if policy.get("rot"):
        img = img.rotate(float(rng.uniform(-policy["rot"], policy["rot"])), resample=Image.BILINEAR)
    if policy.get("rot90") and rng.random() < 0.75:
        img = img.rotate(90 * int(rng.integers(1, 4)))
    if policy.get("hflip") and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if policy.get("vflip") and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    img = img.resize((size, size), Image.BILINEAR)

    # ---- photometric ops on the tensor ----------------------------------- #
    x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
    if policy.get("bc"):
        gain = 1.0 + rng.uniform(-policy["bc"], policy["bc"])
        bias = rng.uniform(-policy["bc"], policy["bc"]) * 0.5
        x = x * gain + bias
    if policy.get("channel_jitter"):
        jitter = rng.uniform(-policy["channel_jitter"], policy["channel_jitter"], 3).astype("float32")
        x = x * (1.0 + torch.from_numpy(jitter).view(3, 1, 1))
    if policy.get("noise"):
        x = x + torch.from_numpy(
            rng.normal(0.0, policy["noise"], size=x.shape).astype("float32")
        )
    return x.clamp(0.0, 1.0)
