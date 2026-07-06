"""Observation-only instrumentation (§6).

``Instrumenter(model).capture(image)`` registers forward hooks on attention
softmaxes and block outputs, runs inference, detaches, and returns a
:class:`Trace`. The hard guarantee (enforced by a regression test at M1) is that
logits are **bit-identical** with hooks on or off — instrumentation observes,
never perturbs.

M0 ships the dataclasses and the interface; ``capture`` is a stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Trace:
    """Captured forward-pass internals for one image.

    Arrays are stored as plain objects (numpy/torch at runtime) keyed by a
    stable name so downstream XAI methods are decoupled from the capture
    mechanism.

    Attributes
    ----------
    attentions:
        Per-layer attention probability tensors, shape ``[H, N, N]`` each.
    tokens:
        Per-layer token embeddings (block inputs + final), ``L+1`` entries.
    logits:
        Final classifier logits.
    meta:
        Free-form capture metadata (model spec, timings, etc.).
    """

    attentions: List[Any] = field(default_factory=list)
    tokens: List[Any] = field(default_factory=list)
    logits: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class Instrumenter:
    """Attaches observation-only hooks to a model and captures a :class:`Trace`.

    Parameters
    ----------
    model:
        The model to instrument (a ``LoadedModel`` or raw ``nn.Module``).
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def capture(self, image: Any) -> Trace:
        """Run instrumented inference and return a detached :class:`Trace`.

        Not implemented at M0 — lands at M1 with the hook-purity guarantee.
        """

        raise NotImplementedError("Instrumenter.capture lands at M1")


__all__ = ["Trace", "Instrumenter"]
