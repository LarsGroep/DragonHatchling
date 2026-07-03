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
    attributes: Dict[str, float] = field(default_factory=dict)  # attr -> corr


def cluster_concepts(
    memory: HebbianFeatureMemory,
    layer: str,
    class_names: Sequence[str],
    n_concepts: int = 8,
    min_units: int = 2,
    activity_threshold: float = 0.02,
) -> List[Concept]:
    """Cluster units into concepts by their Hebbian co-activation fingerprint.

    Dead and near-dead units (mean activation below ``activity_threshold`` ×
    the most active unit) are excluded first — with wide sparse neuron
    spaces they otherwise dominate the geometry and everything alive chains
    into one giant cluster.  The remaining units are clustered by Ward
    agglomeration on the rows of the correlation matrix (each unit's
    "who do I fire with" fingerprint), which produces balanced clusters
    where average-linkage on raw correlation distance degenerates.

    Class affinity is normalized per concept relative to its strongest
    class (top class = 1.0), so scores stay meaningful for datasets with
    hundreds of classes.
    """
    from sklearn.cluster import AgglomerativeClustering

    corr = memory.correlation(layer).numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    mean_act = memory.stats[layer].mean_act.numpy()

    active = np.where(mean_act > activity_threshold * max(mean_act.max(), 1e-12))[0]
    if len(active) < max(min_units * 2, 4):        # degenerate memory; keep all
        active = np.arange(corr.shape[0])
    n_clusters = max(1, min(n_concepts, len(active) // min_units, len(active)))

    fingerprints = corr[np.ix_(active, active)]
    if n_clusters == 1 or len(active) <= n_clusters:
        labels = np.zeros(len(active), dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ).fit_predict(fingerprints)

    affinity = memory.class_affinity(layer).numpy()   # [classes, units]

    concepts: List[Concept] = []
    for cid in range(labels.max() + 1):
        units = active[labels == cid]
        if len(units) < min_units:
            continue
        sub = corr[np.ix_(units, units)]
        off_diag = sub[~np.eye(len(units), dtype=bool)]
        coherence = float(off_diag.mean()) if off_diag.size else 0.0
        cls_scores = affinity[:, units].mean(axis=1)
        peak = cls_scores.max()
        norm = cls_scores / peak if peak > 0 else cls_scores
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
def probe_activations(
    model: nn.Module,
    probe_images: torch.Tensor,
    layers: Optional[Dict[str, nn.Module]] = None,
    memory: Optional[HebbianFeatureMemory] = None,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """Pooled, rectified activations of the observed layers on a probe set.

    Returns ``{layer_name: [n_images, units]}``.  If a memory is given its
    statistics are paused during the pass (probing must not contaminate
    training statistics) and wide layers are subsampled to the same tracked
    units the memory uses, so indices line up with concept ``units``.
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
    if memory is not None:
        for name, a in acts.items():
            st = memory.stats.get(name)
            if st is not None and st.unit_index is not None:
                acts[name] = a[:, st.unit_index]
    return acts


def concept_scores(
    concepts: Sequence[Concept],
    acts: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Per-image concept activation: ``[n_images, n_concepts]``."""
    cols = [acts[c.layer][:, c.units].mean(dim=1) for c in concepts]
    return torch.stack(cols, dim=1)


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
    acts = probe_activations(model, probe_images, layers, memory, batch_size)
    scores = concept_scores(concepts, acts)
    for i, concept in enumerate(concepts):
        k = min(top_k, scores.shape[0])
        concept.exemplars = scores[:, i].topk(k).indices.tolist()
    return concepts
