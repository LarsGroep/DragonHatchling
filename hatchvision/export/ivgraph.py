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
         "coherence", "importance", "classes": {name: affinity},
         "attributes": {name: effect}?, "size"},
        {"id": "k:<class>",        "type": "class",   "label", "size"},
        {"id": "a:<attribute>",    "type": "attribute", "label", "size"}
      ],
      "edges": [
        {"source", "target", "weight", "kind": "coactivation" | "membership"
                                              | "affinity" | "attribute"}
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
    used_attributes = set()
    for c in concepts:
        node = {
            "id": f"c:{c.concept_id}",
            "type": "concept",
            "label": c.label,
            "cluster": c.concept_id,
            "coherence": round(c.coherence, 4),
            "importance": round(c.importance, 5),
            "classes": {k: round(v, 4) for k, v in c.class_affinity.items()},
            "size": 1.6,
        }
        if c.attributes:
            node["attributes"] = {k: round(v, 4) for k, v in c.attributes.items()}
        nodes.append(node)
        for attr, score in c.attributes.items():
            used_attributes.add(attr)
            edges.append(
                {
                    "source": f"c:{c.concept_id}",
                    "target": f"a:{attr}",
                    "weight": round(score, 4),
                    "kind": "attribute",
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

    for attr in sorted(used_attributes):
        nodes.append(
            {"id": f"a:{attr}", "type": "attribute", "label": attr, "size": 1.2}
        )

    tracked = set(tracked_units)
    seen_pairs = set()

    def _add_coact(i: int, j: int, w: float) -> None:
        key = (i, j) if i < j else (j, i)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        edges.append(
            {
                "source": f"u:{layer}:{key[0]}",
                "target": f"u:{layer}:{key[1]}",
                "weight": round(float(w), 4),
                "kind": "coactivation",
            }
        )

    for i, j, w in memory.top_edges(layer, k=max_edges):
        if w < min_edge_weight or i not in tracked or j not in tracked:
            continue
        _add_coact(i, j, w)

    # Always emit the co-activation structure *within* each concept, using the
    # correlation matrix restricted to the concept's units.  The global top-K
    # above ranks pairs across every tracked unit, so on wide sparse neuron
    # spaces the strongest pairs often fall outside the clustered subset and the
    # exported neuron network comes out edgeless.  These intra-concept edges
    # guarantee the "Neurons" view always has real Hebbian links to draw.
    corr = memory.correlation(layer)
    for c in concepts:
        us = c.units
        for a in range(len(us)):
            for b in range(a + 1, len(us)):
                i, j = us[a], us[b]
                w = float(corr[i, j])
                if w >= min_edge_weight:
                    _add_coact(i, j, w)

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
