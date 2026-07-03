"""Unit → class influence for the Hebbian-observed layer.

Answers "how much does each tracked unit push each class logit?" so the web
app can show per-concept SHAP contributions without a Python runtime.

For a background batch we compute the gradient of every class logit with
respect to the observed layer's pooled activations (the same pooled,
rectified, subsampled units the :class:`HebbianFeatureMemory` tracks and the
ONNX bundle emits as ``act_<layer>``) and average it:

    weights[k, u] = E_bg[ d logit_k / d act_u ]
    baseline[u]   = E_bg[ act_u ]

A per-image attribution is then ``phi[k, u] = weights[k, u] * (act_u -
baseline[u])``, which approximates ``logit_k - E_bg[logit_k]``.

When the logits are *linear* in the tracked activations — true for the
``hybrid`` and ``bdh`` readouts, where the observed neuron layer feeds a
residual linear path into a linear head — the gradient is constant, the
first-order expansion is the function itself, and ``phi`` equals the exact
Shapley values of the linear model (independent players, background-mean
reference). We detect that case by checking gradient constancy across the
background batch and label the result ``exact-linear``; anything else is
reported honestly as ``expected-gradients``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from hatchvision.hebbian.memory import HebbianFeatureMemory


@dataclass
class UnitInfluence:
    layer: str
    weights: torch.Tensor          # [num_classes, units]
    baseline: torch.Tensor         # [units] mean background activation
    expected_logits: torch.Tensor  # [num_classes] mean background logits
    method: str                    # "exact-linear" | "expected-gradients"


def _pool_grad(g: torch.Tensor) -> torch.Tensor:
    """Gradient w.r.t. the *pooled* per-unit activation.

    The tracked activation of a conv/token layer is the spatial/token mean;
    a uniform shift of one unit by eps shifts every position by eps, so the
    pooled gradient is the sum of the positional gradients.
    """
    if g.dim() == 4:
        return g.sum(dim=(2, 3))
    if g.dim() == 3:
        return g.sum(dim=1)
    return g


def unit_class_influence(
    model: nn.Module,
    memory: HebbianFeatureMemory,
    layer: str,
    background: torch.Tensor,
    batch_size: int = 32,
    linear_tol: float = 1e-4,
) -> UnitInfluence:
    """Expected-gradients influence of the tracked units on every logit.

    ``background`` is a small batch of reference images (32-128 works well;
    the training probe batch is a natural choice). The model is evaluated in
    ``eval()`` mode (dropout off) and restored afterwards; the memory is
    paused so the pass leaves its statistics untouched.
    """
    layers = model.hebbian_layers()
    if layer not in layers:
        raise KeyError(f"model has no hebbian layer {layer!r}; options: {list(layers)}")
    st = memory.stats[layer]
    unit_index = st.unit_index

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    captured: dict = {}

    def hook(_m, _i, out):
        captured["z"] = out

    handle = layers[layer].register_forward_hook(hook)

    grads = []      # per-sample gradients, [B, num_classes, units]
    acts = []       # per-sample tracked activations, [B, units]
    logit_sum = None
    n = 0
    try:
        with memory.paused():
            for start in range(0, background.shape[0], batch_size):
                x = background[start : start + batch_size].to(device)
                captured.clear()
                logits = model(x)
                z = captured.get("z")
                if z is None or not z.requires_grad:
                    raise RuntimeError(
                        f"layer {layer!r} produced no differentiable output — "
                        "is the observed layer inside a no_grad() block?"
                    )
                num_classes = logits.shape[1]
                batch_grads = []
                for k in range(num_classes):
                    (g,) = torch.autograd.grad(
                        logits[:, k].sum(), z, retain_graph=k < num_classes - 1
                    )
                    pg = _pool_grad(g.detach())
                    if unit_index is not None:
                        pg = pg[:, unit_index]
                    batch_grads.append(pg.cpu())
                grads.append(torch.stack(batch_grads, dim=1))  # [b, K, U]

                a = torch.relu(HebbianFeatureMemory._pool(z.detach().float()))
                if unit_index is not None:
                    a = a[:, unit_index]
                acts.append(a.cpu())

                ls = logits.detach().sum(dim=0).cpu()
                logit_sum = ls if logit_sum is None else logit_sum + ls
                n += x.shape[0]
    finally:
        handle.remove()
        model.train(was_training)

    all_grads = torch.cat(grads)            # [N, K, U]
    weights = all_grads.mean(dim=0)         # [K, U]
    baseline = torch.cat(acts).mean(dim=0)  # [U]
    expected_logits = logit_sum / n

    # Linear readout ⇔ the gradient does not depend on the input.
    spread = all_grads.std(dim=0).max()
    scale = all_grads.abs().mean().clamp(min=1e-12)
    method = "exact-linear" if (spread / scale) < linear_tol else "expected-gradients"

    return UnitInfluence(
        layer=layer,
        weights=weights,
        baseline=baseline,
        expected_logits=expected_logits,
        method=method,
    )


def class_fingerprints(
    memory: HebbianFeatureMemory,
    layer: str,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-class Hebbian firing fingerprint over the tracked units.

    Returns ``[num_classes, units]`` — the mean (L2-normalized) firing rate
    of every unit on images of each class, i.e. the class's *activation
    region* in neuron space. With ``normalize=True`` each class row is
    scaled so its strongest unit is 1.0, which keeps regions comparable
    across frequent and rare classes.
    """
    fp = memory.class_affinity(layer).clone()
    if normalize:
        peak = fp.max(dim=1, keepdim=True).values.clamp(min=1e-12)
        fp = fp / peak
    return fp
