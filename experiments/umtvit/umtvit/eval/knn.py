"""Cosine k-NN read-out of frozen features (ARCHITECTURE §6.1).

A non-parametric companion to the linear probe: each test feature votes with
the labels of its ``k`` nearest train features under cosine similarity. Like the
probe it standardises by the train split's statistics and **skips gracefully**
(returns ``None``) when labels are absent. Semantics mirror the notebook's
``knn_accuracy`` (normalise → similarity → top-k → majority vote).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from umtvit.eval.features import FrozenFeatures, standardize

__all__ = ["knn_accuracy"]


def knn_accuracy(
    train: FrozenFeatures,
    test: FrozenFeatures,
    *,
    k: int = 5,
) -> Optional[Dict[str, object]]:
    """Cosine k-NN accuracy on frozen features.

    Args:
        train: Frozen features + labels used as the reference set.
        test: Frozen features + labels to classify.
        k: Number of neighbours to vote (clamped to the reference size).

    Returns:
        ``None`` if either split is unlabeled or empty. Otherwise a dict with
        ``"accuracy"``, the effective ``"k"``, ``"chance"``, and ``"num_classes"``.
    """
    if not train.labeled or not test.labeled or len(train) == 0 or len(test) == 0:
        return None
    num_classes = max(train.num_classes, test.num_classes)
    if num_classes < 2:
        return None

    xtr, xte = standardize(train.pooled, test.pooled)
    a = F.normalize(xtr, dim=1)
    b = F.normalize(xte, dim=1)
    sim = b @ a.T  # [N_test, N_train]
    eff_k = int(min(k, a.shape[0]))
    nn_idx = sim.topk(eff_k, dim=1).indices  # [N_test, k]
    votes = train.labels[nn_idx]  # [N_test, k]
    pred = torch.mode(votes, dim=1).values
    accuracy = float((pred == test.labels).float().mean().item())
    return {
        "accuracy": accuracy,
        "k": eff_k,
        "chance": 1.0 / num_classes,
        "num_classes": num_classes,
    }
