#!/usr/bin/env python3
"""Train an image classifier and optionally export the Hebbian IVGraph.

Examples
--------
Train the default small CNN on CIFAR-10 and export a concept graph::

    python scripts/train.py --dataset cifar10 --backbone simple_cnn \
        --epochs 3 --hebbian --export-graph exports/graph.json

Try the experimental Baby Dragon Hatchling backbone::

    python scripts/train.py --dataset cifar10 --backbone bdh --epochs 3

Use your own dataset (only the loader changes — nothing else)::

    python scripts/train.py --dataset imagefolder --root /path/to/data \
        --backbone resnet18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hatchvision import (
    HebbianFeatureMemory,
    TrainConfig,
    Trainer,
    build_loader,
    create_model,
)
from hatchvision.data import available_loaders
from hatchvision.models import available_backbones


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cifar10", choices=available_loaders())
    p.add_argument("--root", default="./data", help="dataset root directory")
    p.add_argument("--backbone", default="simple_cnn", choices=available_backbones())
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--hebbian", action="store_true", help="record Hebbian feature memory")
    p.add_argument("--export-graph", default=None, help="write IVGraph JSON here")
    p.add_argument("--n-concepts", type=int, default=8)
    p.add_argument("--checkpoint", default=None, help="save model weights here")
    args = p.parse_args()

    loader_kwargs = {}
    if args.limit_train is not None:
        loader_kwargs["limit_train"] = args.limit_train
    if args.limit_val is not None:
        loader_kwargs["limit_val"] = args.limit_val
    data = build_loader(args.dataset, root=args.root, **loader_kwargs)
    train_loader, val_loader = data.dataloaders(batch_size=args.batch_size)

    model = create_model(args.backbone, data.spec)
    memory = None
    if args.hebbian or args.export_graph:
        memory = HebbianFeatureMemory(model, num_classes=data.spec.num_classes)
        print(f"Hebbian memory attached to layers: {list(model.hebbian_layers())}")

    trainer = Trainer(model, TrainConfig(epochs=args.epochs, lr=args.lr), memory)
    trainer.fit(train_loader, val_loader)

    if args.checkpoint:
        import torch

        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"saved weights to {args.checkpoint}")

    if args.export_graph:
        from hatchvision.explain import cluster_concepts
        from hatchvision.export import export_ivgraph

        layer = memory.layer_names[-1]
        concepts = cluster_concepts(
            memory, layer, data.spec.class_names, n_concepts=args.n_concepts
        )
        path = export_ivgraph(
            memory,
            concepts,
            layer,
            data.spec.class_names,
            args.export_graph,
            meta={"dataset": data.spec.name, "backbone": args.backbone},
        )
        print(f"IVGraph exported to {path} ({len(concepts)} concepts)")


if __name__ == "__main__":
    main()
