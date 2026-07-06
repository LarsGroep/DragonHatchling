"""Concept tier (§9).

A k-sparse autoencoder (4096 features, k=32) over layer-9 token activations
yields a concept dictionary per (model, dataset): each feature gets exemplar
patches, class-conditional firing rates, and an optional CLIP-text probe label
(marked *suggested*; exemplars are ground truth). A quality gate decides per
dataset whether to fall back to PCA/k-means clustering via the same
:class:`ConceptProvider` interface.

M0 ships the dataclasses and the Protocol; providers land at M4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable


@dataclass
class ConceptFeature:
    """One dictionary feature (§9)."""

    id: int
    exemplars: List[Any] = field(default_factory=list)  # (image_id, token_idx, act)
    firing_rates: List[float] = field(default_factory=list)  # per class
    suggested_label: Optional[str] = None  # CLIP-probe suggestion, human-editable


@dataclass
class ConceptDictionary:
    """A concept dictionary artifact for a (model, dataset, layer)."""

    model: str
    dataset: str
    layer: int
    features: List[ConceptFeature] = field(default_factory=list)


@runtime_checkable
class ConceptProvider(Protocol):
    """Produces per-token top-k concept activations and a dictionary (§9)."""

    def dictionary(self) -> ConceptDictionary:
        ...

    def encode(self, activations: Any) -> Any:
        ...


def train_sae(activations: Any, *, n_features: int = 4096, k: int = 32) -> ConceptProvider:
    """Train the k-sparse autoencoder. Not implemented at M0 — lands at M4."""
    raise NotImplementedError("train_sae lands at M4")


__all__ = [
    "ConceptFeature",
    "ConceptDictionary",
    "ConceptProvider",
    "train_sae",
]
