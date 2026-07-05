#!/usr/bin/env python3
"""Rebuild the concept graph from saved Hebbian statistics — no retraining.

The Kaggle notebook (and any run that saves ``memory.state_dict()``) leaves a
``hebbian_state.pt`` next to the bundle.  This script re-runs the *analysis*
half of the pipeline on it — concept clustering, class affinity, optional
class-level attribute grounding — so clustering parameters can be iterated
in seconds instead of re-training for an hour.

Class names come from the bundle's ``manifest.json`` or from the dataset
loader; attribute grounding uses the dataset's class-level attribute matrix
when available (CUB-200 has one), which needs no model or images.

Examples::

    # re-cluster a Kaggle CUB bundle with different parameters
    python scripts/rebuild_graph.py --state bundle/hebbian_state.pt \
        --manifest bundle/manifest.json --dataset cub200 --root data/cub \
        --n-concepts 24 --out bundle/graph.json

    # manifest-only (no dataset on disk): clustering + class affinity
    python scripts/rebuild_graph.py --state bundle/hebbian_state.pt \
        --manifest bundle/manifest.json --out bundle/graph.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hatchvision import HebbianFeatureMemory, build_loader
from hatchvision.explain import (
    cluster_concepts,
    ground_concepts_from_class_attributes,
)
from hatchvision.export import export_ivgraph


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--state", required=True, help="hebbian_state.pt from a training run")
    p.add_argument("--manifest", default=None, help="bundle manifest.json (class names)")
    p.add_argument("--dataset", default=None, help="registered loader (attribute grounding)")
    p.add_argument("--root", default="./data")
    p.add_argument("--layer", default=None, help="observed layer (default: last)")
    p.add_argument("--n-concepts", type=int, default=24)
    p.add_argument("--min-units", type=int, default=2)
    p.add_argument("--activity-threshold", type=float, default=0.02)
    p.add_argument("--out", required=True, help="write IVGraph JSON here")
    p.add_argument(
        "--explain-out",
        default=None,
        help="also (re)write the demo explain pack here (e.g. bundle/explain.json); "
        "class fingerprints are rebuilt from the state, an existing file's "
        "shap section (which needs the model) is preserved",
    )
    p.add_argument(
        "--hierarchy-out",
        default=None,
        help="also (re)write the concept hierarchy pack here (e.g. "
        "bundle/hierarchy.json): concept tree + class prototypes, rebuilt "
        "from the state alone (node pixel patches need the model, so an "
        "existing file's patches are preserved when the tree shape matches)",
    )
    p.add_argument("--max-depth", type=int, default=3, help="concept tree depth")
    args = p.parse_args()

    memory = HebbianFeatureMemory.from_state(
        torch.load(args.state, map_location="cpu", weights_only=False)
    )
    layer = args.layer or memory.layer_names[-1]

    class_names = None
    meta = {"generator": "rebuild_graph"}
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        class_names = manifest.get("class_names")
        meta["dataset"] = manifest.get("dataset")
        meta["backbone"] = manifest.get("backbone")

    data = None
    if args.dataset:
        data = build_loader(args.dataset, root=args.root)
        class_names = class_names or list(data.spec.class_names)
        meta.setdefault("dataset", data.spec.name)
    if class_names is None:
        p.error("need --manifest or --dataset to know the class names")
    if len(class_names) != memory.num_classes:
        p.error(
            f"class name count ({len(class_names)}) != classes in state "
            f"({memory.num_classes})"
        )

    concepts = cluster_concepts(
        memory,
        layer,
        class_names,
        n_concepts=args.n_concepts,
        min_units=args.min_units,
        activity_threshold=args.activity_threshold,
    )
    print(f"{len(concepts)} concepts; sizes: {[len(c.units) for c in concepts]}")

    if data is not None:
        attr_names = data.attribute_names()
        cls_attrs = getattr(data, "class_attribute_matrix", lambda: None)()
        if attr_names and cls_attrs is not None:
            ground_concepts_from_class_attributes(
                concepts, cls_attrs, attr_names, class_names
            )
            grounded = sum(1 for c in concepts if c.attributes)
            print(f"attribute grounding (class-level): {grounded}/{len(concepts)}")

    path = export_ivgraph(memory, concepts, layer, class_names, args.out, meta=meta)
    print(f"IVGraph rebuilt at {path}")

    if args.explain_out:
        from hatchvision.export import build_explain_pack

        doc = build_explain_pack(memory, layer, class_names)
        out = Path(args.explain_out)
        if out.exists():
            try:
                prev = json.loads(out.read_text())
                if "shap" in prev:
                    doc["shap"] = prev["shap"]
                    print("kept existing shap section (rebuilding it needs the model)")
            except (json.JSONDecodeError, OSError):
                pass
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, separators=(",", ":")))
        print(f"explain pack rebuilt at {out}")

    if args.hierarchy_out:
        from hatchvision.export import build_hierarchy_pack
        from hatchvision.hebbian import build_concept_tree

        tree = build_concept_tree(
            memory,
            layer,
            class_names,
            max_depth=args.max_depth,
            min_units=args.min_units,
            activity_threshold=args.activity_threshold,
        )
        doc = build_hierarchy_pack(memory, layer, class_names, tree)
        out = Path(args.hierarchy_out)
        if out.exists():
            # node patches need the trained model + probe images; keep any
            # existing ones whose node ids still exist in the rebuilt tree
            try:
                prev = json.loads(out.read_text())

                def _patches(node, acc):
                    if node.get("patches"):
                        acc[node["node_id"]] = node["patches"]
                    for c in node.get("children", []):
                        _patches(c, acc)
                    return acc

                old = _patches(prev.get("tree", {}), {})

                def _attach(node):
                    if node["node_id"] in old:
                        node["patches"] = old[node["node_id"]]
                    for c in node.get("children", []):
                        _attach(c)

                if old:
                    _attach(doc["tree"])
                    print("kept existing node patches (rebuilding them needs the model)")
            except (json.JSONDecodeError, OSError, KeyError):
                pass
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, separators=(",", ":")))
        n_nodes = sum(1 for _ in _walk_dict(doc["tree"]))
        print(f"hierarchy pack rebuilt at {out} ({n_nodes} nodes)")


def _walk_dict(node: dict):
    yield node
    for c in node.get("children", []):
        yield from _walk_dict(c)


if __name__ == "__main__":
    main()
