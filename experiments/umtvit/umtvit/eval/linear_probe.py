"""Linear probe on frozen features (ARCHITECTURE §6.1).

The standard SSL yardstick: a single logistic-regression layer trained on the
*frozen* pooled features (no gradient reaches the encoder), read out on the test
split. Pure torch — a ``Linear`` layer optimised by AdamW for a fixed number of
steps from a fixed seed, so the number is reproducible on CPU.

Features are standardised by the train split's statistics
(:func:`umtvit.eval.features.standardize`) before probing, matching the
notebook reference. The probe returns the test accuracy plus a per-class
accuracy dict, and **skips gracefully** (returns ``None``) when labels are
absent — the SSL training and topology metrics do not depend on it.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from umtvit.eval.features import FrozenFeatures, standardize

__all__ = ["linear_probe"]


def _per_class_accuracy(pred: Tensor, target: Tensor, num_classes: int) -> Dict[int, float]:
    """Recall per class: fraction of each true class predicted correctly."""
    acc: Dict[int, float] = {}
    for c in range(num_classes):
        mask = target == c
        n = int(mask.sum().item())
        acc[c] = float((pred[mask] == c).float().mean().item()) if n > 0 else float("nan")
    return acc


def linear_probe(
    train: FrozenFeatures,
    test: FrozenFeatures,
    *,
    steps: int = 400,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, object]]:
    """Train a logistic-regression probe on frozen features; return test metrics.

    Args:
        train: Frozen features + labels for the probe's training split.
        test: Frozen features + labels for the held-out evaluation split.
        steps: Fixed number of full-batch AdamW steps.
        lr: Probe learning rate.
        weight_decay: Probe weight decay.
        seed: Torch seed for the probe's parameter init (reproducibility).
        device: Where to run the probe; defaults to the features' device (CPU).

    Returns:
        ``None`` if either split is unlabeled or empty (graceful skip). Otherwise
        a dict with ``"accuracy"`` (float), ``"per_class_accuracy"``
        (``{class_index: recall}``), ``"chance"`` (1/num_classes), and
        ``"num_classes"``.
    """
    if not train.labeled or not test.labeled or len(train) == 0 or len(test) == 0:
        return None
    num_classes = max(train.num_classes, test.num_classes)
    if num_classes < 2:
        return None

    device = device if device is not None else train.pooled.device
    xtr, xte = standardize(train.pooled, test.pooled)
    xtr, ytr = xtr.to(device), train.labels.to(device)
    xte, yte = xte.to(device), test.labels

    torch.manual_seed(seed)
    probe = torch.nn.Linear(xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(probe(xtr), ytr).backward()
        opt.step()

    with torch.no_grad():
        pred = probe(xte).argmax(1).cpu()
    accuracy = float((pred == yte).float().mean().item())
    return {
        "accuracy": accuracy,
        "per_class_accuracy": _per_class_accuracy(pred, yte, num_classes),
        "chance": 1.0 / num_classes,
        "num_classes": num_classes,
    }
