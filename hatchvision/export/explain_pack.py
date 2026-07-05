"""Export the demo "explain pack" (``explain.json``) for the web app.

Everything the browser needs to demo **Hebbian activation regions** and
**SHAP influence** without a Python runtime:

* ``fingerprints`` — per-class Hebbian firing fingerprints over the tracked
  units (``[num_classes, units]``): each class's *activation region* in
  neuron space. Rebuildable from ``hebbian_state.pt`` alone.
* ``shap`` — a unit→class influence matrix plus background baseline
  (:func:`hatchvision.explain.influence.unit_class_influence`), from which
  the browser computes per-image, per-concept SHAP contributions as
  ``weights[k] · (act - baseline)``. Requires the trained model and a
  background batch, so this section is optional.

Unit ``i`` in every matrix corresponds to graph node ``<node_prefix>i`` and
to element ``i`` of the ONNX ``act_<layer>`` output — the same order the
memory tracks and the bundle emits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
from torch import nn

from hatchvision.explain.influence import class_fingerprints, unit_class_influence
from hatchvision.hebbian.heads import HebbianPrototypeHead
from hatchvision.hebbian.hierarchy import ConceptNode
from hatchvision.hebbian.memory import HebbianFeatureMemory


def _rounded(t: torch.Tensor, decimals: int, prune_below: float = 0.0) -> list:
    """Tensor → nested lists of compactly rounded floats (small JSON)."""
    t = t.detach().float()
    if prune_below > 0:
        t = torch.where(t.abs() < prune_below, torch.zeros_like(t), t)
    r = torch.round(t * 10**decimals) / 10**decimals
    return r.tolist()


def build_explain_pack(
    memory: HebbianFeatureMemory,
    layer: str,
    class_names: Sequence[str],
    model: Optional[nn.Module] = None,
    background: Optional[torch.Tensor] = None,
    fingerprint_decimals: int = 3,
    weight_decimals: int = 5,
) -> Dict:
    """Assemble the explain-pack document as a plain dict.

    Fingerprints always; the ``shap`` section only when both ``model`` and
    ``background`` are given.
    """
    st = memory.stats[layer]
    fp = class_fingerprints(memory, layer)
    doc: Dict = {
        "format": "hatchvision-explain",
        "version": "1.0",
        "layer": layer,
        "node_prefix": f"u:{layer}:",
        "units": st.dim,
        "num_classes": len(class_names),
        "fingerprints": {
            "norm": "per-class-max",
            "matrix": _rounded(fp, fingerprint_decimals, prune_below=0.005),
        },
    }
    if model is not None and background is not None:
        inf = unit_class_influence(model, memory, layer, background)
        doc["shap"] = {
            "method": inf.method,
            "reference": "background-mean",
            "weights": _rounded(inf.weights, weight_decimals),
            "baseline": _rounded(inf.baseline, weight_decimals),
            "expected_logits": _rounded(inf.expected_logits, 4),
        }
    return doc


def export_explain_pack(
    memory: HebbianFeatureMemory,
    layer: str,
    class_names: Sequence[str],
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    background: Optional[torch.Tensor] = None,
    **kwargs,
) -> Path:
    """Build and write ``explain.json``; returns the written path.

    Written compactly (no indent) — the matrices dominate the size and
    gzip on any static host shrinks them a further ~5×.
    """
    doc = build_explain_pack(
        memory, layer, class_names, model=model, background=background, **kwargs
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return path


def build_hierarchy_pack(
    memory: HebbianFeatureMemory,
    layer: str,
    class_names: Sequence[str],
    tree: ConceptNode,
    prototype_head: Optional[HebbianPrototypeHead] = None,
    config: Optional[Dict] = None,
    prototype_decimals: int = 5,
) -> Dict:
    """Assemble ``hierarchy.json`` — the browser-side classifier document.

    Contains everything a browser needs to (a) render the concept tree with
    per-node pixel patches, (b) route an ONNX ``act_<layer>`` vector through
    the tree, and (c) run the prototype head / enroll new classes client-side:

    * ``tree`` — :meth:`ConceptNode.to_dict` of the root (with any attached
      ``patches`` data-URIs);
    * ``prototypes`` — ``{class_name: unit vector}`` (L2-normalized firing
      prototype per class) for the gradient-free prototype head;
    * ``layer`` / ``unit_ids`` — the tracked units, aligned with
      ``act_<layer>`` and graph nodes ``u:<layer>:<i>``;
    * ``config`` — temperature and routing settings.

    This section is purely additive; ``graph.json`` / ``explain.json`` are
    unchanged and remain valid on their own.
    """
    if prototype_head is None:
        prototype_head = HebbianPrototypeHead.from_memory(memory, layer, class_names)
    prototypes = {
        name: _rounded(prototype_head.prototypes[i], prototype_decimals)
        for i, name in enumerate(prototype_head.class_names)
    }
    doc: Dict = {
        "format": "hatchvision-hierarchy",
        "version": "1.0",
        "layer": layer,
        "node_prefix": f"u:{layer}:",
        "unit_ids": memory.unit_ids(layer),
        "num_classes": len(prototype_head.class_names),
        "class_names": list(prototype_head.class_names),
        "tree": tree.to_dict(),
        "prototypes": prototypes,
        "config": {
            "temperature": prototype_head.temperature,
            **(config or {}),
        },
    }
    return doc


def export_hierarchy_pack(
    memory: HebbianFeatureMemory,
    layer: str,
    class_names: Sequence[str],
    tree: ConceptNode,
    path: Union[str, Path],
    **kwargs,
) -> Path:
    """Build and write ``hierarchy.json``; returns the written path."""
    doc = build_hierarchy_pack(memory, layer, class_names, tree, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return path
