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


if __name__ == "__main__":
    main()
