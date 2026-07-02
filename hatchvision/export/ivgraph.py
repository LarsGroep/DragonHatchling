"""Export the Hebbian concept graph as IVGraph JSON.

Schema (``format: "ivgraph", version: "1.0"``)::

    {
      "format": "ivgraph",
      "version": "1.0",
      "meta":  {"dataset": ..., "backbone": ..., "layer": ..., "created": ...},
      "nodes": [
        {"id": "u:<layer>:<unit>", "type": "unit",    "label", "cluster",
         "activation", "size"},
        {"id": "c:<cid>",          "type": "concept", "label", "cluster",
         "coherence", "importance", "classes": {name: affinity}, "size"},
        {"id": "k:<class>",        "type": "class",   "label", "size"}
      ],
      "edges": [
        {"source", "target", "weight", "kind": "coactivation" | "membership"
                                              | "affinity"}
      ]
    }

The bundled web app (``webapp/``) renders exactly this format.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from hatchvision.explain.concepts import Concept
from hatchvision.hebbian.memory import HebbianFeatureMemory


def build_ivgraph(
    memory: HebbianFeatureMemory,
    concepts: Sequence[Concept],
    layer: str,
    class_names: Sequence[str],
    meta: Optional[Dict] = None,
    max_edges: int = 400,
    min_edge_weight: float = 0.1,
) -> Dict:
    """Assemble the IVGraph document as a plain dict."""
    unit_cluster: Dict[int, int] = {}
    for c in concepts:
        for u in c.units:
            unit_cluster[u] = c.concept_id

    mean_act = memory.stats[layer].mean_act
    channel_ids = memory.unit_ids(layer)
    max_act = float(mean_act.max().item()) or 1.0

    nodes: List[Dict] = []
    edges: List[Dict] = []

    tracked_units = sorted(unit_cluster)
    for u in tracked_units:
        act = float(mean_act[u].item())
        nodes.append(
            {
                "id": f"u:{layer}:{u}",
                "type": "unit",
                "label": f"{layer} · ch {channel_ids[u]}",
                "cluster": unit_cluster[u],
                "activation": round(act, 5),
                "size": round(0.4 + 0.6 * act / max_act, 4),
            }
        )

    used_classes = set()
    for c in concepts:
        nodes.append(
            {
                "id": f"c:{c.concept_id}",
                "type": "concept",
                "label": c.label,
                "cluster": c.concept_id,
                "coherence": round(c.coherence, 4),
                "importance": round(c.importance, 5),
                "classes": {k: round(v, 4) for k, v in c.class_affinity.items()},
                "size": 1.6,
            }
        )
        for u in c.units:
            edges.append(
                {
                    "source": f"u:{layer}:{u}",
                    "target": f"c:{c.concept_id}",
                    "weight": 1.0,
                    "kind": "membership",
                }
            )
        for cls, score in c.class_affinity.items():
            if score <= 0:
                continue
            used_classes.add(cls)
            edges.append(
                {
                    "source": f"c:{c.concept_id}",
                    "target": f"k:{cls}",
                    "weight": round(score, 4),
                    "kind": "affinity",
                }
            )

    for cls in sorted(used_classes):
        nodes.append({"id": f"k:{cls}", "type": "class", "label": cls, "size": 2.0})

    tracked = set(tracked_units)
    for i, j, w in memory.top_edges(layer, k=max_edges):
        if w < min_edge_weight or i not in tracked or j not in tracked:
            continue
        edges.append(
            {
                "source": f"u:{layer}:{i}",
                "target": f"u:{layer}:{j}",
                "weight": round(w, 4),
                "kind": "coactivation",
            }
        )

    doc_meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layer": layer,
        "num_classes": len(class_names),
        "generator": "hatchvision",
    }
    doc_meta.update(meta or {})
    return {
        "format": "ivgraph",
        "version": "1.0",
        "meta": doc_meta,
        "nodes": nodes,
        "edges": edges,
    }


def export_ivgraph(
    memory: HebbianFeatureMemory,
    concepts: Sequence[Concept],
    layer: str,
    class_names: Sequence[str],
    path: Union[str, Path],
    **kwargs,
) -> Path:
    """Build and write the IVGraph JSON; returns the written path."""
    doc = build_ivgraph(memory, concepts, layer, class_names, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return path
