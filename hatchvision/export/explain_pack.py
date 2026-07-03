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
