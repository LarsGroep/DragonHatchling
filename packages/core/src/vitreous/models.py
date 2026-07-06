"""Model wrappers (§2, §6).

v1 targets ViT-S/16 (timm ``deit_small_patch16_224``) fine-tuned per dataset.
The wrapper keeps the rest of the system model-agnostic.

torch/timm are optional ``[ml]`` extras — importing this module must **not**
require them, so :func:`load_model` imports ``timm`` lazily (inside the call).
A tiny registry mirrors the dataset registry (``register_model`` / ``get_model``
/ ``list_models``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Describes a loadable ViT.

    Carries every field the Explanation Pack manifest's ``ModelInfo`` needs
    (arch, patch_size, layers, heads, embed_dim, tokens) so a pack can be
    stamped straight from the spec.
    """

    arch: str
    patch_size: int = 16
    num_layers: int = 12
    num_heads: int = 6
    num_tokens: int = 197
    embed_dim: int = 384
    image_size: int = 224
    hf_repo: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
    """A materialized model plus the metadata the pipeline needs."""

    spec: ModelSpec
    module: Any  # torch.nn.Module at runtime; kept Any to avoid a hard dep.
    num_classes: int = 0

    def to_model_info_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for the manifest ``ModelInfo`` (patch_size incl.)."""
        return {
            "arch": self.spec.arch,
            "hf_repo": self.spec.hf_repo or "local",
            "num_layers": self.spec.num_layers,
            "num_heads": self.spec.num_heads,
            "num_tokens": self.spec.num_tokens,
            "embed_dim": self.spec.embed_dim,
            "patch_size": self.spec.patch_size,
        }


# --------------------------------------------------------------------------- #
# Registry — mirrors vitreous.data's dataset registry.
# --------------------------------------------------------------------------- #

_MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(name: str, spec: ModelSpec) -> ModelSpec:
    """Register ``spec`` under ``name``. Raises on empty/duplicate names."""
    if not name:
        raise ValueError("model name must be a non-empty string")
    if name in _MODEL_REGISTRY:
        raise ValueError(f"model {name!r} is already registered")
    _MODEL_REGISTRY[name] = spec
    return spec


def get_model_spec(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` registered under ``name``."""
    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "<none>"
        raise KeyError(
            f"no model registered as {name!r}; available: {available}"
        ) from exc


def list_models() -> List[str]:
    """Return the sorted names of all registered models."""
    return sorted(_MODEL_REGISTRY)


# ViT-S/16 (DeiT-S) — the v1 default. 197 tokens @224, 12 layers, 6 heads.
register_model(
    "vit_s16",
    ModelSpec(
        arch="deit_small_patch16_224",
        patch_size=16,
        num_layers=12,
        num_heads=6,
        num_tokens=197,
        embed_dim=384,
        image_size=224,
        hf_repo="",
    ),
)


# --------------------------------------------------------------------------- #
# Loader (lazy timm import).
# --------------------------------------------------------------------------- #


def load_model(
    model: Any = "vit_s16",
    dataset_spec: Any = None,
    *,
    pretrained: bool = False,
    weights: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> LoadedModel:
    """Build a timm ViT with a fresh classification head.

    Parameters
    ----------
    model:
        A registered model name (e.g. ``"vit_s16"``) or a :class:`ModelSpec`.
    dataset_spec:
        A :class:`~vitreous.data.DatasetSpec` (or anything with a
        ``num_classes`` attribute) used to size the fresh head. Overridden by
        an explicit ``num_classes``.
    pretrained:
        Passed to ``timm.create_model``. **Keep ``False`` in tests** — never
        download weights in CI.
    weights:
        Optional path to a ``state_dict`` checkpoint to load after building.
    num_classes:
        Explicit head size; falls back to ``dataset_spec.num_classes`` then
        ``1000``.

    Returns
    -------
    LoadedModel
    """
    spec = model if isinstance(model, ModelSpec) else get_model_spec(model)

    if num_classes is None:
        if dataset_spec is not None and hasattr(dataset_spec, "num_classes"):
            num_classes = int(dataset_spec.num_classes)
        else:
            num_classes = 1000

    import timm  # lazy — only needed when actually building a model
    import torch

    module = timm.create_model(
        spec.arch, pretrained=pretrained, num_classes=num_classes
    )
    module.eval()

    if weights is not None:
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        module.load_state_dict(state)

    return LoadedModel(spec=spec, module=module, num_classes=num_classes)


__all__ = [
    "ModelSpec",
    "LoadedModel",
    "load_model",
    "register_model",
    "get_model_spec",
    "list_models",
]
