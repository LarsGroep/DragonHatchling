"""Faithfulness evaluation (§6, §10).

Deletion & insertion curves (patch-masking in ranked order, 20 steps), AUC per
method, and pairwise Spearman rank correlation between methods (method
agreement). Precomputed for gallery packs; computed on demand for uploads.

M0 ships the result container and function signatures as stubs; implementations
land at M2 (acceptance: ``AUC(chefer) > AUC(random)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FaithfulnessResult:
    """Deletion/insertion curves + AUCs + method-agreement matrix."""

    deletion_curves: Dict[str, List[float]] = field(default_factory=dict)
    insertion_curves: Dict[str, List[float]] = field(default_factory=dict)
    auc: Dict[str, float] = field(default_factory=dict)
    agreement: Dict[str, Dict[str, float]] = field(default_factory=dict)
    steps: int = 20


def deletion_insertion(model: Any, image: Any, attributions: Any, *, steps: int = 20) -> FaithfulnessResult:
    """Compute deletion/insertion curves and AUC. Not implemented at M0."""
    raise NotImplementedError("deletion_insertion lands at M2")


def method_agreement(attributions: Any) -> Dict[str, Dict[str, float]]:
    """Pairwise Spearman rank correlation between methods. Not implemented at M0."""
    raise NotImplementedError("method_agreement lands at M2")


__all__ = ["FaithfulnessResult", "deletion_insertion", "method_agreement"]
