"""Shared forward-pass helpers for the XAI methods (§6).

These wrap the timm ViT so the attribution methods can (a) run the model from
raw pixels, (b) split the model into *embed* → *transformer blocks* so token-level
Integrated Gradients can integrate over the block-input token embeddings, and
(c) resolve a target class. Everything imports torch lazily so ``import
vitreous.xai`` stays torch-free (the M0 import-purity guarantee).

The embed / block split is exact: ``run_from_tokens(embed_tokens(x))`` reproduces
``model(x)`` bit-for-bit (verified against timm ``deit_small_patch16_224``).
"""

from __future__ import annotations

from typing import Any, Optional


def unwrap(model: Any) -> Any:
    """Accept a ``LoadedModel`` or a raw ``nn.Module`` and return the module."""
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "forward"):
        return module
    return model


def as_batch(image: Any) -> Any:
    """Add a leading batch dim if ``image`` is ``[C, H, W]``."""
    if hasattr(image, "dim") and image.dim() == 3:
        return image.unsqueeze(0)
    return image


def embed_tokens(model: Any, x: Any) -> Any:
    """Run the ViT stem: pixels ``[B,C,H,W]`` → block-input tokens ``[B,T,D]``.

    Mirrors timm ``VisionTransformer.forward_features`` up to (not including)
    the transformer blocks, so the result equals the input to ``blocks[0]``.
    """
    m = unwrap(model)
    e = m.patch_embed(x)
    e = m._pos_embed(e)
    e = m.patch_drop(e)
    e = m.norm_pre(e)
    return e


def run_from_tokens(model: Any, tokens: Any) -> Any:
    """Run blocks + final norm + classifier head from block-input ``tokens``."""
    m = unwrap(model)
    y = tokens
    for blk in m.blocks:
        y = blk(y)
    y = m.norm(y)
    return m.forward_head(y)


def resolve_target(logits: Any, target: Optional[int]) -> int:
    """Return ``target`` if given, else the argmax class of ``logits[0]``."""
    if target is None:
        return int(logits[0].argmax().item())
    return int(target)


__all__ = [
    "unwrap",
    "as_batch",
    "embed_tokens",
    "run_from_tokens",
    "resolve_target",
]
