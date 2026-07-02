"""Minimal, dependency-free Grad-CAM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Grad-CAM over any model that exposes a spatial ``cam_layer``.

    Usage::

        cam = GradCAM(model)                    # uses model.cam_layer()
        heatmaps = cam(images)                  # [B, H, W] in [0, 1]
        cam.close()
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        if target_layer is None:
            if not hasattr(model, "cam_layer") or model.cam_layer() is None:
                raise ValueError("model has no cam_layer(); pass target_layer=")
            target_layer = model.cam_layer()
        self.model = model
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_acts),
            target_layer.register_full_backward_hook(self._save_grads),
        ]

    def _save_acts(self, _m, _i, out):
        self._acts = out

    def _save_grads(self, _m, _gin, gout):
        self._grads = gout[0]

    def __call__(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        upsample_to: Optional[int] = None,
    ) -> torch.Tensor:
        """Return per-image heatmaps in [0, 1], shape [B, H, W]."""
        was_training = self.model.training
        self.model.eval()
        images = images.requires_grad_(True)
        logits = self.model(images)
        if targets is None:
            targets = logits.argmax(dim=1)
        score = logits.gather(1, targets[:, None]).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()

        acts, grads = self._acts, self._grads
        weights = grads.mean(dim=(2, 3), keepdim=True)          # [B, C, 1, 1]
        cam = torch.relu((weights * acts).sum(dim=1))           # [B, h, w]
        size = upsample_to or images.shape[-1]
        cam = F.interpolate(
            cam[:, None], size=(size, size), mode="bilinear", align_corners=False
        )[:, 0]
        flat = cam.flatten(1)
        lo = flat.min(dim=1).values[:, None, None]
        hi = flat.max(dim=1).values[:, None, None]
        cam = (cam - lo) / (hi - lo + 1e-8)
        if was_training:
            self.model.train()
        return cam.detach()

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def denormalize(images: torch.Tensor, mean, std) -> torch.Tensor:
    """Invert dataset normalization for display, clamped to [0, 1]."""
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return (images * std + mean).clamp(0, 1)
