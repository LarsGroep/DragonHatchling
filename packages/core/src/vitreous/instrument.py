"""Observation-only instrumentation (§6).

``Instrumenter(model).capture(image)`` registers forward hooks on a timm ViT,
runs inference, detaches, and returns a :class:`Trace` holding:

* **attention** — per-layer softmax maps ``[L, H, T, T]``
* **tokens** — the residual-stream token embeddings at each timeline step
  ``[L+1, T, D]``: the input to each of the ``L`` blocks (steps ``0..L-1``)
  plus the final-norm output (step ``L``)
* **logits** — the classifier output
* **timings** — wall-clock stage timings

**Hard guarantee (enforced by test):** logits are *bit-identical* whether or
not the Instrumenter is attached. This holds by construction — the hooks only
*read* module inputs and never return a modified output. In particular,
modern timm attention uses a fused ``scaled_dot_product_attention`` kernel that
does not expose its internal softmax; rather than switch the model to the
slower unfused path (which would change the numerics), the attention hook
*recomputes* the softmax from the attention module's own inputs/weights. The
model's real forward pass runs untouched, so the logits are unperturbed while
the captured attention is exactly the softmax the fused kernel computed.

The Instrumenter is a context manager; hooks are removed on ``__exit__`` (and
by ``capture`` when it owns the hooks), leaving no lingering handles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Trace:
    """Captured forward-pass internals for one image.

    Attributes
    ----------
    attention:
        ``[L, H, T, T]`` tensor of per-layer/head attention probabilities.
    tokens:
        ``[L+1, T, D]`` tensor of residual-stream token embeddings.
    logits:
        Classifier logits (kept with the batch dim, e.g. ``[B, num_classes]``).
    attention_grad:
        ``[L, H, T, T]`` gradient of a target logit w.r.t. the attention maps,
        populated only by :meth:`Instrumenter.capture_with_grad` (used by
        Chefer relevance). ``None`` on the observation-only ``capture`` path.
    timings:
        Wall-clock stage timings in milliseconds.
    meta:
        Free-form capture metadata (shapes, model arch, etc.).
    """

    attention: Optional[Any] = None
    tokens: Optional[Any] = None
    logits: Optional[Any] = None
    attention_grad: Optional[Any] = None
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def _unwrap(model: Any) -> Any:
    """Accept a ``LoadedModel`` or a raw ``nn.Module`` and return the module."""
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "forward"):
        return module
    return model


class Instrumenter:
    """Attaches observation-only hooks to a timm ViT and captures a :class:`Trace`.

    Usage::

        with Instrumenter(model) as inst:
            trace = inst.capture(image_tensor)
        # hooks are gone here

    or standalone (``capture`` registers and removes its own hooks)::

        trace = Instrumenter(model).capture(image_tensor)
    """

    def __init__(self, model: Any) -> None:
        self.model = _unwrap(model)
        if not hasattr(self.model, "blocks"):
            raise TypeError(
                "Instrumenter expects a timm ViT with a `.blocks` ModuleList"
            )
        self._handles: List[Any] = []
        self._attn: Dict[int, Any] = {}
        self._tokens: Dict[int, Any] = {}
        self._final_norm: Optional[Any] = None

    # -- hook registration --------------------------------------------------- #

    def _make_token_pre_hook(self, layer: int):
        def _hook(module, args):  # forward_pre_hook
            # args[0] is the residual-stream tensor entering this block.
            self._tokens[layer] = args[0].detach()
            return None  # never modify the input

        return _hook

    def _make_attn_hook(self, layer: int):
        import torch  # lazy

        def _hook(module, args, output):  # forward_hook
            # Recompute the softmax attention from this module's own inputs so
            # the model's (possibly fused) forward stays untouched.
            x = args[0]
            b, n, c = x.shape
            qkv = (
                module.qkv(x)
                .reshape(b, n, 3, module.num_heads, module.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, _v = qkv.unbind(0)
            q = module.q_norm(q)
            k = module.k_norm(k)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            self._attn[layer] = attn.detach()
            return None  # never modify the output

        return _hook

    def _make_final_norm_hook(self):
        def _hook(module, args, output):
            self._final_norm = output.detach()
            return None

        return _hook

    def register(self) -> "Instrumenter":
        """Register all hooks. Idempotent-safe (no-op if already registered)."""
        if self._handles:
            return self
        blocks = self.model.blocks
        for i, blk in enumerate(blocks):
            self._handles.append(
                blk.register_forward_pre_hook(self._make_token_pre_hook(i))
            )
            self._handles.append(
                blk.attn.register_forward_hook(self._make_attn_hook(i))
            )
        # Final norm output = the last timeline step (L).
        self._handles.append(
            self.model.norm.register_forward_hook(self._make_final_norm_hook())
        )
        return self

    def remove(self) -> None:
        """Remove all registered hooks, leaving no lingering handles."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "Instrumenter":
        return self.register()

    def __exit__(self, *exc: Any) -> None:
        self.remove()

    # -- capture ------------------------------------------------------------- #

    def capture(self, image: Any) -> Trace:
        """Run instrumented inference and return a detached :class:`Trace`.

        ``image`` may be ``[C, H, W]`` or ``[B, C, H, W]``; a leading batch dim
        is added if missing. Attention/token arrays are returned with the batch
        dim squeezed when ``B == 1`` so a single image yields the canonical
        ``[L, H, T, T]`` / ``[L+1, T, D]`` shapes.
        """
        import torch  # lazy

        owns_hooks = not self._handles
        if owns_hooks:
            self.register()

        # Reset per-capture buffers.
        self._attn.clear()
        self._tokens.clear()
        self._final_norm = None

        x = image
        if hasattr(x, "dim") and x.dim() == 3:
            x = x.unsqueeze(0)

        was_training = self.model.training
        self.model.eval()
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(x)
            forward_ms = (time.perf_counter() - t0) * 1000.0

            n_layers = len(self.model.blocks)
            # tokens: inputs to blocks 0..L-1, then final-norm output = step L.
            token_steps = [self._tokens[i] for i in range(n_layers)]
            if self._final_norm is not None:
                token_steps.append(self._final_norm)
            tokens = torch.stack(token_steps, dim=0)  # [L+1, B, T, D]
            attention = torch.stack(
                [self._attn[i] for i in range(n_layers)], dim=0
            )  # [L, B, H, T, T]

            # Squeeze the batch dim for the common single-image case.
            if tokens.shape[1] == 1:
                tokens = tokens.squeeze(1)  # [L+1, T, D]
            if attention.shape[1] == 1:
                attention = attention.squeeze(1)  # [L, H, T, T]

            trace = Trace(
                attention=attention.contiguous(),
                tokens=tokens.contiguous(),
                logits=logits.detach(),
                timings={"forward_ms": forward_ms},
                meta={
                    "num_layers": n_layers,
                    "num_heads": int(self.model.blocks[0].attn.num_heads),
                    "num_tokens": int(tokens.shape[-2]),
                    "embed_dim": int(tokens.shape[-1]),
                    "attention_shape": tuple(attention.shape),
                    "tokens_shape": tuple(tokens.shape),
                },
            )
        finally:
            if was_training:
                self.model.train()
            if owns_hooks:
                self.remove()

        return trace

    # -- grad-enabled capture (Chefer relevance, §6 / DECISION-LOG) ----------- #

    def capture_with_grad(self, image: Any, target: Optional[int] = None) -> Trace:
        """Run a *differentiable* forward and return attention maps + their grads.

        Chefer relevance (§6) needs the gradient of a target logit w.r.t. each
        layer's attention softmax. The observation-only :meth:`capture` recomputes
        attention *outside* the autograd graph (so it has no path to the logits);
        here we instead temporarily replace each attention module's ``forward``
        with a numerically-equivalent *unfused* version that materializes the
        softmax matrix, calls ``retain_grad`` on it, and computes the block output
        the standard way — so the target logit backpropagates into every
        attention map.

        This path deliberately drops ``no_grad`` and uses the unfused attention
        (equal to the fused SDPA kernel up to float epsilon). It is entirely
        separate from :meth:`capture`; the bit-identical hook-purity guarantee on
        ``capture`` is untouched. The monkeypatch is installed and removed within
        this call, leaving the model exactly as found.

        Returns a :class:`Trace` with ``attention`` (softmax maps, detached),
        ``attention_grad`` (their grads, detached), ``logits`` and
        ``meta['target']``. ``target`` defaults to the argmax class.
        """
        import torch  # lazy

        model = self.model
        x = image
        if hasattr(x, "dim") and x.dim() == 3:
            x = x.unsqueeze(0)

        blocks = model.blocks
        n_layers = len(blocks)
        attn_store: Dict[int, Any] = {}
        originals: Dict[int, Any] = {}

        def _make_forward(layer: int, attn_mod: Any):
            def _forward(x, attn_mask=None, is_causal=False):
                b, n, c = x.shape
                qkv = (
                    attn_mod.qkv(x)
                    .reshape(b, n, 3, attn_mod.num_heads, attn_mod.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv.unbind(0)
                q = attn_mod.q_norm(q)
                k = attn_mod.k_norm(k)
                q = q * attn_mod.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn.retain_grad()
                attn_store[layer] = attn
                a = attn_mod.attn_drop(attn)
                out = a @ v
                out = out.transpose(1, 2).reshape(b, n, attn_mod.attn_dim)
                out = attn_mod.norm(out)
                out = attn_mod.proj(out)
                out = attn_mod.proj_drop(out)
                return out

            return _forward

        was_training = model.training
        model.eval()
        try:
            for i, blk in enumerate(blocks):
                originals[i] = blk.attn.forward
                blk.attn.forward = _make_forward(i, blk.attn)

            t0 = time.perf_counter()
            logits = model(x)
            if target is None:
                target = int(logits[0].argmax().item())
            model.zero_grad(set_to_none=True)
            logits[0, target].backward()
            forward_ms = (time.perf_counter() - t0) * 1000.0

            attention = torch.stack(
                [attn_store[i].detach() for i in range(n_layers)], dim=0
            )  # [L, B, H, T, T]
            grads = torch.stack(
                [attn_store[i].grad.detach() for i in range(n_layers)], dim=0
            )  # [L, B, H, T, T]
            if attention.shape[1] == 1:
                attention = attention.squeeze(1)  # [L, H, T, T]
                grads = grads.squeeze(1)

            trace = Trace(
                attention=attention.contiguous(),
                attention_grad=grads.contiguous(),
                logits=logits.detach(),
                timings={"forward_ms": forward_ms},
                meta={
                    "num_layers": n_layers,
                    "num_heads": int(blocks[0].attn.num_heads),
                    "num_tokens": int(attention.shape[-1]),
                    "target": int(target),
                    "grad_capture": True,
                },
            )
        finally:
            for i, blk in enumerate(blocks):
                if i in originals:
                    # Restore the class-level forward by dropping the instance attr.
                    try:
                        del blk.attn.forward
                    except AttributeError:
                        blk.attn.forward = originals[i]
            model.zero_grad(set_to_none=True)
            if was_training:
                model.train()

        return trace


__all__ = ["Trace", "Instrumenter"]
