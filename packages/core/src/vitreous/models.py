"""Model wrappers (§2, §6).

v1 targets ViT-S/16 (timm ``deit_small_patch16_224``) fine-tuned per dataset.
The wrapper keeps the rest of the system model-agnostic. torch/timm are optional
``[ml]`` extras — importing this module must not require them, so the loader is
a stub at M0 and the real implementation (which imports timm lazily) lands at
M1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelSpec:
    """Describes a loadable model (mirrors manifest ``ModelInfo``)."""

    arch: str
    hf_repo: str
    num_layers: int = 12
    num_heads: int = 6
    num_tokens: int = 197
    embed_dim: int = 384
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
    """A materialized model plus the metadata the pipeline needs."""

    spec: ModelSpec
    module: Any  # torch.nn.Module at runtime; kept Any to avoid a hard dep.


def load_model(spec: ModelSpec, *, weights: Optional[str] = None) -> LoadedModel:
    """Instantiate a model from a :class:`ModelSpec`. Not implemented at M0.

    The M1 implementation imports ``timm`` lazily (an ``[ml]`` extra) so that
    importing :mod:`vitreous` stays torch-free.
    """

    raise NotImplementedError(
        "load_model lands at M1; install the 'ml' extra (torch, timm)"
    )


__all__ = ["ModelSpec", "LoadedModel", "load_model"]
