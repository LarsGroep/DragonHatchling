"""XAI attribution suite (§6).

Each method is a pure function ``(Trace | model, image) -> Attribution``.
v1 ships: raw attention maps, attention rollout, Chefer relevance (default
lens), Grad-CAM, and integrated gradients. Faithfulness metrics live in
:mod:`vitreous.xai.eval`.

M0 ships the :class:`Attribution` container and method signatures as stubs;
implementations land at M2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

Method = Literal["attention", "rollout", "chefer", "gradcam", "ig"]


@dataclass
class Attribution:
    """A single attribution result over tokens and/or pixels.

    Attributes
    ----------
    method:
        Which method produced this attribution.
    token_scores:
        Per-token relevance, shape ``[N]`` (or ``[L, N]`` for per-layer).
    pixel_map:
        Optional dense pixel-level heatmap.
    meta:
        Method parameters and provenance.
    """

    method: Method
    token_scores: Optional[Any] = None
    pixel_map: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def attention_maps(trace: Any) -> Attribution:
    """Raw per-layer/head attention (from a :class:`~vitreous.instrument.Trace`)."""
    raise NotImplementedError("attention_maps lands at M2")


def attention_rollout(trace: Any) -> Attribution:
    """Cumulative attention rollout (per-layer prefix products)."""
    raise NotImplementedError("attention_rollout lands at M2")


def chefer_relevance(model: Any, image: Any, *, target: Optional[int] = None) -> Attribution:
    """Class-specific gradient×attention relevance (the default lens)."""
    raise NotImplementedError("chefer_relevance lands at M2")


def grad_cam(model: Any, image: Any, *, target: Optional[int] = None) -> Attribution:
    """Grad-CAM over the last block's token grid."""
    raise NotImplementedError("grad_cam lands at M2")


def integrated_gradients(
    model: Any, image: Any, *, target: Optional[int] = None, steps: int = 20
) -> Attribution:
    """Integrated gradients, pixel- and token-level."""
    raise NotImplementedError("integrated_gradients lands at M2")


__all__ = [
    "Method",
    "Attribution",
    "attention_maps",
    "attention_rollout",
    "chefer_relevance",
    "grad_cam",
    "integrated_gradients",
]
