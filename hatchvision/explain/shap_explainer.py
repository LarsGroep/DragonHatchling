"""SHAP feature-importance wrapper (optional dependency).

Uses ``shap.GradientExplainer`` (expected-gradients) over raw pixels, which
works with any of the framework's models.  ``shap`` is an optional extra:
``pip install shap``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


class ShapExplainer:
    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        """``background`` is a small batch of reference images (e.g. 32-64)."""
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP explanations require the optional 'shap' package: "
                "pip install shap"
            ) from e
        self.device = device or next(model.parameters()).device
        self.model = model.eval()
        self._explainer = shap.GradientExplainer(
            model, background.to(self.device)
        )

    def explain(
        self, images: torch.Tensor, nsamples: int = 50
    ) -> np.ndarray:
        """Per-pixel SHAP values for each image's *predicted* class.

        Returns an array of shape ``[B, H, W]`` (summed over channels):
        positive values push toward the predicted class, negative away.
        """
        images = images.to(self.device)
        with torch.no_grad():
            preds = self.model(images).argmax(dim=1).cpu().numpy()
        shap_values = self._explainer.shap_values(images, nsamples=nsamples)
        # shap returns [B, C, H, W, num_classes] (newer) or a list per class.
        if isinstance(shap_values, list):
            per_class = np.stack(shap_values, axis=-1)
        else:
            per_class = np.asarray(shap_values)
        picked = np.stack(
            [per_class[i, ..., preds[i]] for i in range(len(preds))]
        )                                              # [B, C, H, W]
        return picked.sum(axis=1)                      # [B, H, W]
