"""Map Hebbian co-activation structure to learned visual concepts.

Units that consistently fire together form clusters; each cluster is
interpreted as a *concept*.  A concept is characterized by

* its member units,
* internal coherence (mean intra-cluster co-activation),
* class affinity (which labels its units fire for), and
* exemplar images from a probe set that activate it most strongly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from hatchvision.hebbian.memory import HebbianFeatureMemory


@dataclass
class Concept:
    concept_id: int
    layer: str
    units: List[int]                    # indices into the tracked units
    coherence: float                    # mean intra-cluster correlation
    importance: float                   # mean firing rate of member units
    class_affinity: Dict[str, float]    # class name -> normalized affinity
    label: str = ""                     # human-readable name
    exemplars: List[int] = field(default_factory=list)  # probe-set indices


def cluster_concepts(
    memory: HebbianFeatureMemory,
    layer: str,
    class_names: Sequence[str],
    n_concepts: int = 8,
    min_units: int = 2,
) -> List[Concept]:
    """Agglomerative clustering of units by Hebbian correlation distance."""
    from sklearn.cluster import AgglomerativeClustering

    corr = memory.correlation(layer).numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    n_units = corr.shape[0]
    n_clusters = min(n_concepts, n_units)
    dist = 1.0 - np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(dist)

    affinity = memory.class_affinity(layer).numpy()   # [classes, units]
    mean_act = memory.stats[layer].mean_act.numpy()

    concepts: List[Concept] = []
    for cid in range(n_clusters):
        units = np.where(labels == cid)[0]
        if len(units) < min_units:
            continue
        sub = corr[np.ix_(units, units)]
        off_diag = sub[~np.eye(len(units), dtype=bool)]
        coherence = float(off_diag.mean()) if off_diag.size else 0.0
        cls_scores = affinity[:, units].mean(axis=1)
        total = cls_scores.sum()
        norm = cls_scores / total if total > 0 else cls_scores
        order = np.argsort(-norm)
        top = [(class_names[i], float(norm[i])) for i in order[:3] if norm[i] > 0]
        label = " / ".join(name for name, _ in top[:2]) or f"concept {cid}"
        concepts.append(
            Concept(
                concept_id=cid,
                layer=layer,
                units=units.tolist(),
                coherence=coherence,
                importance=float(mean_act[units].mean()),
                class_affinity={name: score for name, score in top},
                label=label,
            )
        )
    concepts.sort(key=lambda c: -c.importance)
    return concepts


@torch.no_grad()
def find_exemplars(
    concepts: List[Concept],
    memory: HebbianFeatureMemory,
    model: nn.Module,
    probe_images: torch.Tensor,
    layers: Optional[Dict[str, nn.Module]] = None,
    top_k: int = 6,
    batch_size: int = 64,
) -> List[Concept]:
    """Attach, per concept, the probe images that activate it most.

    Runs the probe set through the model once, capturing the observed layers'
    pooled activations, then scores each image by the mean activation of each
    concept's member units.  Fills ``Concept.exemplars`` in place.
    """
    if layers is None:
        layers = model.hebbian_layers()
    captured: Dict[str, List[torch.Tensor]] = {name: [] for name in layers}

    def make_hook(name):
        def hook(_m, _i, out):
            a = torch.relu(HebbianFeatureMemory._pool(out.detach().float()))
            captured[name].append(a.cpu())
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in layers.items()]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        # Pause the memory so probing doesn't contaminate training statistics.
        pause = memory.paused() if memory is not None else None
        if pause:
            pause.__enter__()
        try:
            for start in range(0, probe_images.shape[0], batch_size):
                model(probe_images[start : start + batch_size].to(device))
        finally:
            if pause:
                pause.__exit__(None, None, None)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    acts = {name: torch.cat(chunks) for name, chunks in captured.items()}
    for concept in concepts:
        a = acts[concept.layer]
        st = memory.stats[concept.layer]
        if st.unit_index is not None:
            a = a[:, st.unit_index]
        score = a[:, concept.units].mean(dim=1)
        k = min(top_k, a.shape[0])
        concept.exemplars = score.topk(k).indices.tolist()
    return concepts
