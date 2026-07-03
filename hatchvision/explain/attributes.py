"""Ground Hebbian concepts in human-readable dataset attributes.

Datasets like CUB-200-2011 annotate every image with binary visual
attributes ("wing color: yellow", "bill shape: hooked", ...).  When the
dataset loader provides them, each Hebbian concept can be *grounded*: we
measure how much more strongly the concept fires on images that have an
attribute than on images that don't, and name the concept after its most
discriminative attributes.  This is the "translate activation patterns to
features" step — instead of "concept 7", the graph shows
"wing color: yellow · bill shape: hooked".

The mechanism is dataset-agnostic: any loader that implements
``attribute_names()`` / ``val_attribute_matrix()`` gets grounding for free;
datasets without attributes keep class-affinity + exemplar labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from hatchvision.explain.concepts import Concept, concept_scores, probe_activations
from hatchvision.hebbian.memory import HebbianFeatureMemory


def ground_concepts(
    concepts: List[Concept],
    memory: HebbianFeatureMemory,
    model: nn.Module,
    probe_images: torch.Tensor,
    attribute_matrix: torch.Tensor,
    attribute_names: Sequence[str],
    top_k: int = 4,
    min_support: int = 5,
    min_effect: float = 0.25,
    relabel: bool = True,
    batch_size: int = 64,
) -> List[Concept]:
    """Attach discriminative attributes to each concept (in place).

    For every (concept, attribute) pair we compute a standardized effect
    size: mean concept activation on images *with* the attribute minus mean
    activation on images *without* it, divided by the activation's standard
    deviation (Cohen's d against the pooled spread).  Attributes need at
    least ``min_support`` positive and negative probe images to be scored,
    and only effects above ``min_effect`` are kept.

    Parameters
    ----------
    probe_images / attribute_matrix:
        Aligned: row ``i`` of the matrix annotates image ``i``.  Use
        ``loader.probe_batch(n)`` with ``loader.probe_attributes(n)``.
    relabel:
        Rewrite ``concept.label`` to its top attributes when any pass the
        threshold (the class-affinity label is kept as a fallback).
    """
    if attribute_matrix.shape[0] != probe_images.shape[0]:
        raise ValueError(
            f"attribute matrix rows ({attribute_matrix.shape[0]}) must match "
            f"probe images ({probe_images.shape[0]})"
        )
    if attribute_matrix.shape[1] != len(attribute_names):
        raise ValueError("attribute matrix columns must match attribute_names")

    acts = probe_activations(model, probe_images, memory=memory, batch_size=batch_size)
    scores = concept_scores(concepts, acts)          # [n_images, n_concepts]

    mask = attribute_matrix > 0                      # [n_images, n_attrs]
    pos_count = mask.sum(dim=0)                      # [n_attrs]
    neg_count = mask.shape[0] - pos_count
    valid = (pos_count >= min_support) & (neg_count >= min_support)

    std = scores.std(dim=0, keepdim=True) + 1e-8     # [1, n_concepts]
    m = mask.float()
    pos_mean = (m.t() @ scores) / pos_count.clamp(min=1).unsqueeze(1)
    neg_mean = ((1 - m).t() @ scores) / neg_count.clamp(min=1).unsqueeze(1)
    effect = (pos_mean - neg_mean) / std             # [n_attrs, n_concepts]
    effect[~valid] = float("-inf")

    for ci, concept in enumerate(concepts):
        col = effect[:, ci]
        k = min(top_k, int(torch.isfinite(col).sum().item()))
        concept.attributes = {}
        if k:
            vals, idx = col.topk(k)
            concept.attributes = {
                attribute_names[i]: round(float(v), 4)
                for i, v in zip(idx.tolist(), vals.tolist())
                if v >= min_effect
            }
        if relabel and concept.attributes:
            top = list(concept.attributes)[:2]
            concept.label = " · ".join(top)
    return concepts


def ground_concepts_from_class_attributes(
    concepts: List[Concept],
    class_attribute_matrix: torch.Tensor,
    attribute_names: Sequence[str],
    class_names: Sequence[str],
    top_k: int = 4,
    relabel: bool = True,
) -> List[Concept]:
    """Cheaper grounding via class-level attribute frequencies.

    No forward passes: a concept's attribute profile is the class-affinity-
    weighted average of per-class attribute frequencies, contrasted against
    the dataset mean.  Less faithful than :func:`ground_concepts` (it can
    only name attributes that covary with whole classes) but free.
    """
    freq = class_attribute_matrix.float()
    if freq.max() > 1.0:
        freq = freq / 100.0                           # CUB stores percentages
    base = freq.mean(dim=0)                           # [n_attrs]
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    for concept in concepts:
        weights = torch.zeros(len(class_names))
        for cls, score in concept.class_affinity.items():
            if cls in name_to_idx:
                weights[name_to_idx[cls]] = max(float(score), 0.0)
        total = weights.sum()
        if total <= 0:
            concept.attributes = {}
            continue
        profile = (weights / total) @ freq            # [n_attrs]
        lift = profile - base
        k = min(top_k, lift.shape[0])
        vals, idx = lift.topk(k)
        concept.attributes = {
            attribute_names[i]: round(float(v), 4)
            for i, v in zip(idx.tolist(), vals.tolist())
            if v > 0
        }
        if relabel and concept.attributes:
            concept.label = " · ".join(list(concept.attributes)[:2])
    return concepts
