#!/usr/bin/env python3
"""Train a Hebbian-explainable skin lesion classifier and export the web bundle.

Supports HAM10000 (ISIC 2018, 7-class) and any ISIC-style imagefolder dataset.

Dataset preparation
-------------------
**HAM10000** (recommended):
    1. Download from Kaggle:
       https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
    2. Unzip so the layout is::

           data/ham10000/
               HAM10000_metadata.csv
               HAM10000_images_part_1/*.jpg
               HAM10000_images_part_2/*.jpg

**ISIC imagefolder**:
    Organise any ISIC-derived split as::

        data/isic/
            train/<diagnosis_code>/*.jpg
            val/<diagnosis_code>/*.jpg
            metadata.csv          # optional: image_id, sex, age, localization → concept grounding

Training
--------
    # Quick smoke test (CPU, ~5 min):
    python scripts/train_skin_lesion.py --dataset ham10000 --root data/ham10000 \\
        --epochs 5 --limit-train 500 --limit-val 100 --export-bundle exports/ham10000_test

    # Full HAM10000 (GPU recommended, ~30 min):
    python scripts/train_skin_lesion.py --dataset ham10000 --root data/ham10000 \\
        --epochs 30 --export-bundle webapp

    # Custom ISIC dataset:
    python scripts/train_skin_lesion.py --dataset isic --root data/isic \\
        --epochs 25 --export-bundle webapp

The exported bundle (webapp/graph.json + model.onnx + manifest.json + explain.json)
is ready to deploy to Vercel with `cd webapp && npx vercel deploy --prod`.
A medical disclaimer is shown automatically in the webapp when ham10000/isic is detected.
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
from hatchvision.engine import compute_class_weights


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="ham10000", choices=["ham10000", "isic"],
                   help="ham10000 (raw CSV + images) or isic (imagefolder)")
    p.add_argument("--root", default="./data/ham10000",
                   help="dataset root directory")
    p.add_argument("--image-size", type=int, default=224,
                   help="input image size (224 recommended for hybrid backbone)")
    p.add_argument("--backbone", default="hybrid",
                   choices=["hybrid", "resnet18", "resnet50", "bdh"],
                   help="hybrid = frozen pretrained ResNet50 + BDH lift (recommended)")
    p.add_argument("--encoder", default="resnet50",
                   help="pretrained encoder for hybrid backbone (resnet18/34/50)")
    p.add_argument("--neuron-dim", type=int, default=512,
                   help="BDH neuron space width (more = richer Hebbian concepts)")
    p.add_argument("--epochs", type=int, default=30,
                   help="training epochs (30 = good for HAM10000 frozen encoder)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="batch size (lower if GPU OOM)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="learning rate")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-concepts", type=int, default=16,
                   help="number of Hebbian concepts to cluster (12-24 works well)")
    p.add_argument("--max-units", type=int, default=512,
                   help="cap on tracked Hebbian units")
    p.add_argument("--probe", type=int, default=512,
                   help="images used for exemplar finding and concept grounding")
    p.add_argument("--limit-train", type=int, default=None,
                   help="subset training set (smoke test)")
    p.add_argument("--limit-val", type=int, default=None,
                   help="subset validation set (smoke test)")
    p.add_argument("--export-bundle", default=None,
                   help="write webapp bundle into this directory (e.g. webapp)")
    p.add_argument("--checkpoint", default=None,
                   help="save model weights here after training")
    p.add_argument("--val-ratio", type=float, default=0.15,
                   help="ham10000: fraction of lesions reserved for val (patient-level split)")
    p.add_argument("--seed", type=int, default=42,
                   help="ham10000: random seed for train/val lesion split")
    p.add_argument("--no-class-weights", action="store_true",
                   help="disable inverse-frequency class weighting (not recommended for HAM10000)")
    p.add_argument("--lr-cycle-epochs", type=int, default=10,
                   help="cosine LR restart period; 0 = fixed LR")
    args = p.parse_args()

    loader_kwargs: dict = {
        "root": args.root,
        "image_size": args.image_size,
    }
    if args.dataset == "ham10000":
        loader_kwargs["val_ratio"] = args.val_ratio
        loader_kwargs["seed"] = args.seed
    if args.limit_train is not None:
        loader_kwargs["limit_train"] = args.limit_train
    if args.limit_val is not None:
        loader_kwargs["limit_val"] = args.limit_val

    print(f"Loading {args.dataset} from {args.root}…")
    data = build_loader(args.dataset, **loader_kwargs)
    train_loader, val_loader = data.dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"  {data.spec.num_classes} classes: {', '.join(data.spec.class_names)}")
    print(f"  train {len(train_loader.dataset)} · val {len(val_loader.dataset)}")

    # Class-weighted loss: HAM10000 is severely imbalanced (nv dominates 67%)
    class_weights = None
    if not args.no_class_weights:
        train_ds = train_loader.dataset
        all_labels = [int(train_ds[i][1]) for i in range(len(train_ds))]
        class_weights = compute_class_weights(all_labels, data.spec.num_classes)
        print("  class weights:", {data.spec.class_names[i]: f"{w:.2f}" for i, w in enumerate(class_weights)})

    model_kwargs: dict = {}
    if args.backbone == "hybrid":
        model_kwargs.update(encoder=args.encoder, pretrained=True, freeze_encoder=True)
    if args.neuron_dim is not None:
        model_kwargs["neuron_dim"] = args.neuron_dim

    model = create_model(args.backbone, data.spec, **model_kwargs)
    memory = HebbianFeatureMemory(
        model, num_classes=data.spec.num_classes, max_units=args.max_units
    )
    print(f"Hebbian memory attached to: {list(model.hebbian_layers())}")

    trainer = Trainer(
        model,
        TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            class_weights=class_weights,
            lr_cycle_epochs=args.lr_cycle_epochs,
        ),
        memory,
    )
    trainer.fit(train_loader, val_loader)

    if args.checkpoint:
        import torch
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved weights → {args.checkpoint}")

    if args.export_bundle:
        from hatchvision.explain import cluster_concepts, find_exemplars, ground_concepts
        from hatchvision.export import export_explain_pack, export_ivgraph, export_onnx_bundle

        layer = memory.layer_names[-1]
        print(f"\nClustering {args.n_concepts} concepts from layer '{layer}'…")
        concepts = cluster_concepts(
            memory, layer, data.spec.class_names, n_concepts=args.n_concepts
        )
        probe = data.probe_batch(args.probe)
        find_exemplars(concepts, memory, model, probe)

        attr_names = data.attribute_names()
        attr_matrix = data.probe_attributes(probe.shape[0])
        if attr_names and attr_matrix is not None:
            print(f"Grounding concepts against {len(attr_names)} attributes "
                  f"(sex, age group, localization)…")
            ground_concepts(concepts, memory, model, probe, attr_matrix, attr_names)
            grounded = sum(1 for c in concepts if c.attributes)
            print(f"  {grounded}/{len(concepts)} concepts named")

        bundle_dir = Path(args.export_bundle)
        graph_path = str(bundle_dir / "graph.json")
        path = export_ivgraph(
            memory, concepts, layer, data.spec.class_names, graph_path,
            meta={"dataset": data.spec.name, "backbone": args.backbone},
        )
        print(f"IVGraph → {path}")

        manifest = export_onnx_bundle(
            model, memory, data.spec, args.export_bundle,
            graph_file="graph.json", explain_file="explain.json",
            extra_meta={"backbone": args.backbone},
        )
        print(f"ONNX bundle → {manifest.parent}/")

        explain_path = export_explain_pack(
            memory, layer, data.spec.class_names,
            bundle_dir / "explain.json",
            model=model,
            background=probe[: min(64, probe.shape[0])],
        )
        print(f"Explain pack → {explain_path}")

        import torch
        torch.save(memory.state_dict(), bundle_dir / "hebbian_state.pt")
        print(f"\nBundle ready in {bundle_dir}/")
        print("Deploy: cd webapp && npx vercel deploy --prod")


if __name__ == "__main__":
    main()
