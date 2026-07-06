"""XAI attribution suite (§6).

Each method is a pure function ``(Trace | model, image) -> Attribution``.
v1 ships: attention rollout, Chefer relevance (the default lens), Grad-CAM, and
Integrated Gradients. Faithfulness metrics live in :mod:`vitreous.xai.eval`.

All methods return numpy ``float32`` arrays and default ``class_idx`` to the
argmax class. Every torch import is lazy, so ``import vitreous.xai`` stays
torch-free (the M0 import-purity guarantee).

Formulation notes
-----------------
* **attention_rollout** — Abnar & Zuidema (2020): per layer, average heads,
  mix in the residual (``0.5·A + 0.5·I``), row-normalize, then take the running
  prefix product. The CLS row of the cumulative product at each layer is the
  per-layer relevance ``[L, T]``.
* **chefer_relevance** — the *generic gradient-weighted attention rollout*
  restricted to self-attention (Chefer et al., ICCV 2021, "Generic
  Attention-model Explainability"). This is the LRP-free variant the task
  prefers for robustness: ``R = I``; per layer ``cam = E_h[(∇A ⊙ A)^+]`` then
  ``R ← R + cam·R``; relevance is the CLS row of ``R``. Class-specific through
  ``∇A`` (gradient of the target logit w.r.t. attention).
* **grad_cam** — Selvaraju et al. (2017) on the last block's token grid.
* **integrated_gradients** — Sundararajan et al. (2017), black baseline, both
  pixel-level ``[224,224]`` and token-level ``[T]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from ._common import as_batch, embed_tokens, resolve_target, run_from_tokens, unwrap

Method = Literal["attention", "rollout", "chefer", "gradcam", "ig"]


@dataclass
class Attribution:
    """A single attribution result over tokens and/or pixels.

    Attributes
    ----------
    method:
        Which method produced this attribution.
    token_scores:
        Per-token relevance, shape ``[T]`` (or ``[L, T]`` for per-layer methods).
        numpy ``float32``.
    pixel_map:
        Optional dense pixel-level (or token-grid) heatmap, numpy ``float32``.
    meta:
        Method parameters and provenance.
    """

    method: Method
    token_scores: Optional[Any] = None
    pixel_map: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def attention_rollout(trace: Any, *, residual_ratio: float = 0.5) -> Attribution:
    """Cumulative attention rollout, per-layer CLS-to-patch relevance ``[L, T]``.

    Parameters
    ----------
    trace:
        A :class:`~vitreous.instrument.Trace` carrying ``attention`` ``[L,H,T,T]``
        (true softmax rows).
    residual_ratio:
        Weight on the attention term in the residual mixing ``r·A + (1-r)·I``;
        ``0.5`` is the standard rollout choice.
    """
    import numpy as np
    import torch

    A = trace.attention
    if hasattr(A, "detach"):
        A = A.detach()
    A = A.float()
    if A.dim() != 4:
        raise ValueError(f"expected attention [L,H,T,T], got shape {tuple(A.shape)}")
    L, H, T, _ = A.shape
    eye = torch.eye(T, dtype=A.dtype)

    cumulative: Optional[Any] = None
    rows = []
    for layer in range(L):
        a = A[layer].mean(dim=0)  # head-average [T,T]
        a = residual_ratio * a + (1.0 - residual_ratio) * eye
        a = a / a.sum(dim=-1, keepdim=True)
        cumulative = a if cumulative is None else a @ cumulative
        rows.append(cumulative[0].clone())  # CLS row -> relevance to every token
    out = torch.stack(rows, dim=0).cpu().numpy().astype(np.float32)  # [L, T]
    return Attribution(
        method="rollout",
        token_scores=out,
        meta={"residual_ratio": residual_ratio, "num_layers": int(L)},
    )


def chefer_relevance(
    model: Any, image: Any, class_idx: Optional[int] = None
) -> Attribution:
    """Class-specific gradient×attention relevance (the default lens).

    Implements the generic gradient-weighted attention rollout for self-attention
    (Chefer et al., ICCV 2021). Returns per-layer cumulative relevance
    ``token_scores`` ``[L, T]`` (row ``l`` = CLS relevance after layer ``l``);
    ``meta['final']`` is the last row ``[T]`` (the reported relevance).
    """
    import numpy as np
    import torch

    from ..instrument import Instrumenter

    x = as_batch(image)
    trace = Instrumenter(model).capture_with_grad(x, target=class_idx)
    A = trace.attention.float()  # [L,H,T,T]
    G = trace.attention_grad.float()  # [L,H,T,T]
    L, H, T, _ = A.shape

    R = torch.eye(T, dtype=A.dtype)
    rows = []
    for layer in range(L):
        cam = (G[layer] * A[layer]).clamp(min=0).mean(dim=0)  # [T,T]
        R = R + cam @ R
        rows.append(R[0].clone())  # CLS relevance row (incl. CLS)
    per_layer = torch.stack(rows, dim=0).cpu().numpy().astype(np.float32)  # [L,T]
    final = per_layer[-1].copy()
    return Attribution(
        method="chefer",
        token_scores=per_layer,
        meta={
            "final": final,
            "target": int(trace.meta.get("target", 0)),
            "formulation": "grad_weighted_rollout_iccv2021",
        },
    )


def grad_cam(model: Any, image: Any, class_idx: Optional[int] = None) -> Attribution:
    """Grad-CAM over the last block's token grid → ``pixel_map`` ``[14, 14]``.

    Uses the last transformer block's output tokens as the activation map and the
    gradient of the target logit w.r.t. them; CLS is dropped and the 196 patch
    tokens are reshaped to the ``14×14`` grid.
    """
    import numpy as np
    import torch

    m = unwrap(model)
    x = as_batch(image)
    store: Dict[str, Any] = {}

    def _hook(module, inp, out):
        out.retain_grad()
        store["act"] = out
        return None

    handle = m.blocks[-1].register_forward_hook(_hook)
    was_training = m.training
    m.eval()
    try:
        logits = m(x)
        target = resolve_target(logits, class_idx)
        m.zero_grad(set_to_none=True)
        logits[0, target].backward()

        act = store["act"].detach()[0]  # [T, D]
        grad = store["act"].grad.detach()[0]  # [T, D]
        act_p = act[1:]  # drop CLS -> [196, D]
        grad_p = grad[1:]
        weights = grad_p.mean(dim=0)  # [D]
        cam = (act_p * weights).sum(dim=-1).clamp(min=0)  # [196]
        grid = int(round(cam.shape[0] ** 0.5))
        cam = cam.reshape(grid, grid)
    finally:
        handle.remove()
        m.zero_grad(set_to_none=True)
        if was_training:
            m.train()

    return Attribution(
        method="gradcam",
        pixel_map=cam.cpu().numpy().astype(np.float32),
        meta={"target": int(target), "grid": int(grid)},
    )


def integrated_gradients(
    model: Any,
    image: Any,
    class_idx: Optional[int] = None,
    steps: int = 20,
) -> Attribution:
    """Integrated Gradients with a **black (zero) baseline**, pixel + token level.

    Returns ``token_scores`` ``[T]`` (IG over block-input token embeddings, with
    the black image's embedding as the token baseline) and ``pixel_map``
    ``[224,224]`` (signed IG over input pixels, summed over channels). Both are
    numpy ``float32``. Deterministic given a fixed ``class_idx`` (eval mode; no
    sampling).
    """
    import numpy as np
    import torch

    m = unwrap(model)
    x = as_batch(image)
    was_training = m.training
    m.eval()
    try:
        with torch.no_grad():
            logits = m(x)
        target = resolve_target(logits, class_idx)

        alphas = torch.linspace(1.0 / steps, 1.0, steps, dtype=x.dtype)

        # -- pixel-level IG over the raw image (black baseline) --------------- #
        baseline_img = torch.zeros_like(x)
        diff_px = x - baseline_img
        grad_acc = torch.zeros_like(x)
        for a in alphas:
            xi = (baseline_img + a * diff_px).clone().requires_grad_(True)
            out = m(xi)
            m.zero_grad(set_to_none=True)
            out[0, target].backward()
            grad_acc = grad_acc + xi.grad.detach()
        ig_px = (diff_px * (grad_acc / steps))[0]  # [C,H,W]
        pixel_map = ig_px.sum(dim=0).cpu().numpy().astype(np.float32)  # [H,W]

        # -- token-level IG over block-input embeddings --------------------- #
        tok = embed_tokens(m, x).detach()  # [1,T,D]
        base_tok = embed_tokens(m, baseline_img).detach()  # black-image embedding
        diff_tok = tok - base_tok
        grad_acc_t = torch.zeros_like(tok)
        for a in alphas:
            ti = (base_tok + a * diff_tok).clone().requires_grad_(True)
            out = run_from_tokens(m, ti)
            m.zero_grad(set_to_none=True)
            out[0, target].backward()
            grad_acc_t = grad_acc_t + ti.grad.detach()
        ig_tok = (diff_tok * (grad_acc_t / steps))[0].sum(dim=-1)  # [T]
        token_scores = ig_tok.cpu().numpy().astype(np.float32)
    finally:
        m.zero_grad(set_to_none=True)
        if was_training:
            m.train()

    return Attribution(
        method="ig",
        token_scores=token_scores,
        pixel_map=pixel_map,
        meta={"target": int(target), "steps": int(steps), "baseline": "black"},
    )


__all__ = [
    "Method",
    "Attribution",
    "attention_rollout",
    "chefer_relevance",
    "grad_cam",
    "integrated_gradients",
]
