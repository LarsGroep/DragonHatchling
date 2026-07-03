#!/usr/bin/env python3
"""Train an image classifier and export the Hebbian explainability bundle.

The universal workflow — same command, any registered dataset::

    python scripts/train.py --dataset cifar10  --backbone simple_cnn --epochs 3 \
        --export-bundle exports/cifar10
    python scripts/train.py --dataset cub200 --root data/cub --backbone hybrid \
        --epochs 15 --export-bundle exports/cub200

``--export-bundle DIR`` writes everything the web app needs into DIR:
``graph.json`` (Hebbian concept graph, attribute-grounded when the dataset
has attributes), ``model.onnx`` (with activation outputs) and
``manifest.json`` (preprocessing + node mapping).

Try the pure Baby Dragon Hatchling backbone::

    python scripts/train.py --dataset cifar10 --backbone bdh --epochs 3

Use your own dataset (only the loader changes — nothing else)::

    python scripts/train.py --dataset imagefolder --root /path/to/data \
        --backbone hybrid
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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="cifar10", choices=available_loaders())
    p.add_argument("--root", default="./data", help="dataset root directory")
    p.add_argument("--image-size", type=int, default=None, help="override input size")
    p.add_argument("--backbone", default="simple_cnn", choices=available_backbones())
    p.add_argument("--encoder", default="resnet50", help="hybrid: pretrained encoder")
    p.add_argument("--neuron-dim", type=int, default=None, help="BDH neuron space width")
    p.add_argument("--no-pretrained", action="store_true", help="hybrid: random encoder")
    p.add_argument("--unfreeze-encoder", action="store_true", help="hybrid: fine-tune encoder")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--max-units", type=int, default=256, help="Hebbian tracked units cap")
    p.add_argument("--hebbian", action="store_true", help="record Hebbian feature memory")
    p.add_argument("--export-graph", default=None, help="write IVGraph JSON here")
    p.add_argument(
        "--export-bundle",
        default=None,
        help="write graph.json + model.onnx + manifest.json into this directory",
    )
    p.add_argument("--n-concepts", type=int, default=12)
    p.add_argument("--probe", type=int, default=512, help="probe images for exemplars/grounding")
    p.add_argument("--checkpoint", default=None, help="save model weights here")
    args = p.parse_args()

    loader_kwargs = {"root": args.root}
    if args.limit_train is not None:
        loader_kwargs["limit_train"] = args.limit_train
    if args.limit_val is not None:
        loader_kwargs["limit_val"] = args.limit_val
    if args.image_size is not None:
        loader_kwargs["image_size"] = args.image_size
    data = build_loader(args.dataset, **loader_kwargs)
    train_loader, val_loader = data.dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model_kwargs = {}
    if args.backbone == "hybrid":
        model_kwargs.update(
            encoder=args.encoder,
            pretrained=not args.no_pretrained,
            freeze_encoder=not args.unfreeze_encoder,
        )
    if args.neuron_dim is not None:
        model_kwargs["neuron_dim"] = args.neuron_dim
    model = create_model(args.backbone, data.spec, **model_kwargs)

    memory = None
    if args.hebbian or args.export_graph or args.export_bundle:
        memory = HebbianFeatureMemory(
            model, num_classes=data.spec.num_classes, max_units=args.max_units
        )
        print(f"Hebbian memory attached to layers: {list(model.hebbian_layers())}")

    trainer = Trainer(model, TrainConfig(epochs=args.epochs, lr=args.lr), memory)
    trainer.fit(train_loader, val_loader)

    if args.checkpoint:
        import torch

        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"saved weights to {args.checkpoint}")

    if args.export_graph or args.export_bundle:
        from hatchvision.explain import (
            cluster_concepts,
            find_exemplars,
            ground_concepts,
        )
        from hatchvision.export import export_ivgraph, export_onnx_bundle

        layer = memory.layer_names[-1]
        concepts = cluster_concepts(
            memory, layer, data.spec.class_names, n_concepts=args.n_concepts
        )
        probe = data.probe_batch(args.probe)
        find_exemplars(concepts, memory, model, probe)

        attr_names = data.attribute_names()
        attr_matrix = data.probe_attributes(probe.shape[0])
        if attr_names and attr_matrix is not None:
            ground_concepts(
                concepts, memory, model, probe, attr_matrix, attr_names
            )
            grounded = sum(1 for c in concepts if c.attributes)
            print(f"attribute grounding: {grounded}/{len(concepts)} concepts named")

        graph_path = args.export_graph
        if args.export_bundle:
            graph_path = graph_path or str(Path(args.export_bundle) / "graph.json")
        path = export_ivgraph(
            memory,
            concepts,
            layer,
            data.spec.class_names,
            graph_path,
            meta={"dataset": data.spec.name, "backbone": args.backbone},
        )
        print(f"IVGraph exported to {path} ({len(concepts)} concepts)")

        if args.export_bundle:
            manifest = export_onnx_bundle(
                model,
                memory,
                data.spec,
                args.export_bundle,
                graph_file=Path(graph_path).name,
                extra_meta={"backbone": args.backbone},
            )
            print(f"inference bundle exported to {manifest.parent}")


if __name__ == "__main__":
    main()
