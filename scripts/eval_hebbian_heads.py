#!/usr/bin/env python3
"""Evaluate gradient-free Hebbian classifiers against the gradient head.

Turns the observation-only Hebbian memory into *actual* classifiers and puts
them head-to-head with the trained gradient head on the same features:

* **prototype head** — nearest per-class mean-firing prototype (cosine);
* **tree-routed head** — route down the concept hierarchy (hard and soft);
* **concept-bottleneck head** — flat concepts → class affinity (and an
  optional logistic upper bound).

It also runs a **few-shot enrollment** experiment: retrain with N classes
removed, then teach those classes to the prototype head from K validation
images (no gradients) and measure held-out accuracy and interference on the
seen classes.

Examples::

    # fast, self-contained shapes demo (auto-generates the dataset)
    python scripts/eval_hebbian_heads.py --dataset shapes --epochs 4 \
        --holdout-classes 1 --shots 5 --out results_shapes.json

    # cifar10 subset with a 2-class few-shot holdout
    python scripts/eval_hebbian_heads.py --dataset cifar10 --backbone simple_cnn \
        --epochs 4 --limit-train 2000 --limit-val 1000 \
        --holdout-classes 2 --shots 5 --out results_cifar10.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from hatchvision import (
    HebbianFeatureMemory,
    TrainConfig,
    Trainer,
    build_loader,
    create_model,
)
from hatchvision.data.base import DatasetSpec
from hatchvision.explain import probe_activations
from hatchvision.hebbian import (
    ConceptBottleneckHead,
    HebbianPrototypeHead,
    TreeRoutedHead,
    build_concept_tree,
)


# --------------------------------------------------------------------- data


def ensure_shapes(root: str) -> None:
    """Generate the procedural shapes dataset if it is not already there."""
    root_p = Path(root)
    if (root_p / "train").is_dir() and (root_p / "val").is_dir():
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_shapes_dataset as msd

    rng = __import__("random").Random(0)
    for split, count in (("train", 120), ("val", 40)):
        for cls in msd.CLASSES:
            d = root_p / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                msd.make_image(cls, 64, rng).save(d / f"{cls}_{i:04d}.png")
    print(f"generated shapes dataset at {root}")


def make_loader(args) -> Tuple[object, DatasetSpec]:
    if args.dataset == "shapes":
        root = args.root if args.root != "./data" else "data/shapes_eval"
        ensure_shapes(root)
        kwargs = {"root": root, "image_size": args.image_size or 64}
        loader = build_loader("imagefolder", **kwargs)
    else:
        kwargs = {"root": args.root}
        if args.limit_train is not None:
            kwargs["limit_train"] = args.limit_train
        if args.limit_val is not None:
            kwargs["limit_val"] = args.limit_val
        if args.image_size is not None:
            kwargs["image_size"] = args.image_size
        loader = build_loader(args.dataset, **kwargs)
    return loader, loader.spec


def dataset_targets(ds: Dataset) -> List[int]:
    if isinstance(ds, Subset):
        base = dataset_targets(ds.dataset)
        return [base[i] for i in ds.indices]
    if hasattr(ds, "targets"):
        return [int(t) for t in ds.targets]
    if hasattr(ds, "samples"):
        return [int(s[1]) for s in ds.samples]
    return [int(ds[i][1]) for i in range(len(ds))]


class RemapDataset(Dataset):
    """Keep only samples whose label is in ``keep`` and remap to 0..k-1."""

    def __init__(self, base: Dataset, keep: Dict[int, int]) -> None:
        self.base = base
        self.keep = keep
        targets = dataset_targets(base)
        self.indices = [i for i, t in enumerate(targets) if t in keep]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, label = self.base[self.indices[i]]
        return img, self.keep[int(label)]


def collect_tensors(loader: DataLoader, limit: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []
    n = 0
    for xb, yb in loader:
        images.append(xb)
        labels.append(yb)
        n += xb.shape[0]
        if limit is not None and n >= limit:
            break
    x = torch.cat(images)
    y = torch.cat(labels)
    if limit is not None:
        x, y = x[:limit], y[:limit]
    return x, y


# ----------------------------------------------------------------- training


def train_model(spec: DatasetSpec, train_loader, val_loader, args, num_classes: int):
    model = create_model(args.backbone, _spec_with_classes(spec, num_classes), **_backbone_kwargs(args))
    memory = HebbianFeatureMemory(model, num_classes=num_classes, max_units=args.max_units)
    trainer = Trainer(model, TrainConfig(epochs=args.epochs, lr=args.lr, log_every=0), memory)
    trainer.fit(train_loader, val_loader)
    return model, memory, trainer


def _spec_with_classes(spec: DatasetSpec, num_classes: int) -> DatasetSpec:
    import dataclasses

    if num_classes == spec.num_classes:
        return spec
    names = tuple(spec.class_names[:num_classes])
    return dataclasses.replace(spec, num_classes=num_classes, class_names=names)


def _backbone_kwargs(args) -> dict:
    kw = {}
    if args.backbone == "hybrid":
        kw.update(encoder=args.encoder, pretrained=not args.no_pretrained,
                  freeze_encoder=True)
    if args.neuron_dim is not None:
        kw["neuron_dim"] = args.neuron_dim
    return kw


# --------------------------------------------------------------- evaluation


@torch.no_grad()
def gradient_accuracy(model, images, labels, device, batch_size=128) -> float:
    model.eval()
    correct = 0
    for s in range(0, images.shape[0], batch_size):
        xb = images[s : s + batch_size].to(device)
        logits = model(xb)
        correct += (logits.argmax(1).cpu() == labels[s : s + batch_size]).sum().item()
    return correct / images.shape[0]


def _acc(pred: torch.Tensor, labels: torch.Tensor) -> float:
    return float((pred == labels).float().mean())


def tree_stats(tree) -> Dict:
    leaves = tree.leaves()
    depths = [n.depth for n in tree.walk()]
    return {
        "n_nodes": len(list(tree.walk())),
        "n_leaves": len(leaves),
        "max_depth": max(depths),
        "root_coherence": round(float(tree.coherence), 4),
        "mean_leaf_coherence": round(sum(l.coherence for l in leaves) / len(leaves), 4),
        "root_units": len(tree.units),
    }


def evaluate_heads(model, memory, layer, class_names, val_images, val_labels,
                   train_images, train_labels, args, device) -> Dict:
    acts = probe_activations(model, val_images, memory=memory)[layer]
    train_acts = probe_activations(model, train_images, memory=memory)[layer]
    results: Dict = {}

    results["gradient"] = round(gradient_accuracy(model, val_images, val_labels, device), 4)

    proto = HebbianPrototypeHead.from_memory(memory, layer, class_names, temperature=args.temperature)
    results["prototype"] = round(_acc(proto.predict(acts), val_labels), 4)

    # prototypes rebuilt from one final-model pass over train images —
    # removes the "memory averages the whole training run" staleness
    refreshed = HebbianPrototypeHead.from_activations(
        layer, class_names, train_acts, train_labels, temperature=args.temperature
    )
    results["prototype_refreshed"] = round(_acc(refreshed.predict(acts), val_labels), 4)

    tree = build_concept_tree(memory, layer, class_names,
                              max_depth=args.max_depth, min_units=args.min_units)
    routed = TreeRoutedHead(tree, class_names, temperature=args.tree_temperature,
                            use_importance_prior=not args.no_importance_prior)
    results["tree_hard"] = round(_acc(routed.predict(acts, mode="hard"), val_labels), 4)
    results["tree_soft"] = round(_acc(routed.predict(acts, mode="soft"), val_labels), 4)

    cb = ConceptBottleneckHead.from_memory(memory, layer, class_names, n_concepts=args.n_concepts)
    results["concept_affinity"] = round(_acc(cb.predict(acts), val_labels), 4)
    if args.fit_logistic:
        cb.fit_logistic(train_acts, train_labels)
        results["concept_logistic"] = round(_acc(cb.predict_logistic(acts), val_labels), 4)

    results["tree_stats"] = tree_stats(tree)
    return results, tree, proto


# ---------------------------------------------------------------- few-shot


def few_shot_experiment(spec, loader, args, device) -> Dict:
    full_names = list(spec.class_names)
    n_hold = args.holdout_classes
    held = list(range(spec.num_classes - n_hold, spec.num_classes))
    seen = [c for c in range(spec.num_classes) if c not in held]
    seen_map = {g: i for i, g in enumerate(seen)}          # global -> compact
    seen_names = [full_names[g] for g in seen]
    held_names = [full_names[g] for g in held]

    # training data: seen classes only, compact labels (same limit as main run)
    train_base = loader._maybe_limit(loader.train_dataset(), args.limit_train)
    train_seen = RemapDataset(train_base, seen_map)
    train_loader = DataLoader(train_seen, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = create_model(args.backbone, _spec_with_classes(spec, len(seen)), **_backbone_kwargs(args))
    memory = HebbianFeatureMemory(model, num_classes=len(seen), max_units=args.max_units)
    Trainer(model, TrainConfig(epochs=args.epochs, lr=args.lr, log_every=0), memory).fit(train_loader)
    layer = memory.layer_names[-1]

    # full validation tensors with global labels
    val_loader = DataLoader(loader.val_dataset(), batch_size=args.batch_size, num_workers=0)
    val_x, val_g = collect_tensors(val_loader, limit=args.limit_val)

    # Prototypes from a single *final-model* pass over the seen train images.
    # Prototypes read straight off the memory average the network over the
    # whole training run; a class enrolled later (final-model activations)
    # then dominates cosine similarity and swallows the stale classes —
    # catastrophic interference that vanishes once every prototype comes from
    # the same final-model footing (still gradient-free).
    tr_x, tr_c = collect_tensors(
        DataLoader(train_seen, batch_size=args.batch_size, num_workers=0),
        limit=args.limit_train or 2000,
    )
    tr_acts = probe_activations(model, tr_x, memory=memory)[layer]
    proto = HebbianPrototypeHead.from_activations(
        layer, seen_names, tr_acts, tr_c, temperature=args.temperature
    )

    # seen accuracy BEFORE enrollment (seen-only prototype head, seen val images)
    seen_mask = torch.tensor([int(g) in seen_map for g in val_g.tolist()])
    seen_x, seen_g = val_x[seen_mask], val_g[seen_mask]
    seen_compact = torch.tensor([seen_map[int(g)] for g in seen_g.tolist()])
    seen_acts = probe_activations(model, seen_x, memory=memory)[layer]
    seen_before = _acc(proto.predict(seen_acts), seen_compact)

    # enroll each held-out class from K disjoint val images
    enroll_used: Dict[int, List[int]] = {}
    for g, name in zip(held, held_names):
        idx = [i for i, gg in enumerate(val_g.tolist()) if gg == g]
        shots = idx[: args.shots]
        enroll_used[g] = shots
        if shots:
            en_acts = probe_activations(model, val_x[shots], memory=memory)[layer]
            proto.enroll(name, en_acts)

    global_of_class = {i: g for i, g in enumerate(seen)}      # compact idx -> global
    for j, g in enumerate(held):
        global_of_class[len(seen) + j] = g

    def global_pred(x):
        acts = probe_activations(model, x, memory=memory)[layer]
        pred = proto.predict(acts)
        return torch.tensor([global_of_class[int(p)] for p in pred.tolist()])

    # seen accuracy AFTER enrollment (all prototypes present → interference?)
    seen_after = _acc(global_pred(seen_x), seen_g)

    # held-out accuracy on non-enrollment images
    used_flat = {i for shots in enroll_used.values() for i in shots}
    held_eval_idx = [i for i, gg in enumerate(val_g.tolist())
                     if gg in held and i not in used_flat]
    held_x = val_x[held_eval_idx]
    held_g = val_g[held_eval_idx]
    held_after = _acc(global_pred(held_x), held_g) if len(held_eval_idx) else float("nan")

    # overall accuracy (exclude enrollment images to keep eval honest)
    overall_idx = [i for i in range(val_x.shape[0]) if i not in used_flat]
    overall_after = _acc(global_pred(val_x[overall_idx]), val_g[overall_idx])

    return {
        "held_out_classes": held_names,
        "seen_classes": len(seen),
        "shots": args.shots,
        "seen_acc_before_enroll": round(seen_before, 4),
        "seen_acc_after_enroll": round(seen_after, 4),
        "held_out_acc": round(held_after, 4),
        "overall_acc_after_enroll": round(overall_after, 4),
        "seen_interference_drop": round(seen_before - seen_after, 4),
    }


# -------------------------------------------------------------------- print


def print_table(title: str, rows: List[Tuple[str, str]]) -> None:
    print(f"\n{title}")
    print("-" * max(len(title), 40))
    for name, val in rows:
        print(f"  {name:<28} {val}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="shapes",
                   help="'shapes' (auto-generated), or any registered loader")
    p.add_argument("--root", default="./data")
    p.add_argument("--backbone", default="simple_cnn")
    p.add_argument("--encoder", default="resnet18")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--neuron-dim", type=int, default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--max-units", type=int, default=128)
    p.add_argument("--holdout-classes", type=int, default=0)
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--n-concepts", type=int, default=8)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-units", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.1, help="prototype cosine temperature")
    p.add_argument("--tree-temperature", type=float, default=0.25, help="soft routing temperature")
    p.add_argument("--no-importance-prior", action="store_true",
                   help="do not divide child scores by cluster importance")
    p.add_argument("--fit-logistic", action="store_true",
                   help="also fit a logistic upper bound on concept scores")
    p.add_argument("--export-hierarchy", default=None,
                   help="write hierarchy.json (with node patches) to this path")
    p.add_argument("--out", default="results.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    loader, spec = make_loader(args)
    device = torch.device("cpu")
    print(f"dataset={spec.name} classes={spec.num_classes} backbone={args.backbone} "
          f"epochs={args.epochs}")

    # ---- main comparison: all classes ----
    train_loader, val_loader = loader.dataloaders(batch_size=args.batch_size, num_workers=0)
    model, memory, _ = train_model(spec, train_loader, val_loader, args, spec.num_classes)
    layer = memory.layer_names[-1]

    val_x, val_y = collect_tensors(
        DataLoader(loader.val_dataset(), batch_size=args.batch_size, num_workers=0),
        limit=args.limit_val,
    )
    train_x, train_y = collect_tensors(
        DataLoader(loader.train_dataset(), batch_size=args.batch_size, num_workers=0),
        limit=args.limit_train or 2000,
    )

    head_results, tree, proto = evaluate_heads(
        model, memory, layer, list(spec.class_names),
        val_x, val_y, train_x, train_y, args, device,
    )
    chance = 1.0 / spec.num_classes

    rows = [
        ("gradient head", f"{head_results['gradient']:.3f}"),
        ("prototype (memory)", f"{head_results['prototype']:.3f}"),
        ("prototype (refreshed)", f"{head_results['prototype_refreshed']:.3f}"),
        ("tree-routed (hard)", f"{head_results['tree_hard']:.3f}"),
        ("tree-routed (soft)", f"{head_results['tree_soft']:.3f}"),
        ("concept-bottleneck (aff)", f"{head_results['concept_affinity']:.3f}"),
    ]
    if "concept_logistic" in head_results:
        rows.append(("concept-bottleneck (logreg)", f"{head_results['concept_logistic']:.3f}"))
    rows.append(("chance", f"{chance:.3f}"))
    print_table(f"Head accuracy — {spec.name} (val n={val_x.shape[0]})", rows)

    ts = head_results["tree_stats"]
    print_table("Concept tree", [
        ("nodes / leaves", f"{ts['n_nodes']} / {ts['n_leaves']}"),
        ("max depth", str(ts["max_depth"])),
        ("root units", str(ts["root_units"])),
        ("root coherence", f"{ts['root_coherence']:.3f}"),
        ("mean leaf coherence", f"{ts['mean_leaf_coherence']:.3f}"),
    ])

    doc: Dict = {
        "dataset": spec.name,
        "num_classes": spec.num_classes,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "chance": round(chance, 4),
        "val_n": int(val_x.shape[0]),
        "heads": head_results,
    }

    # ---- few-shot enrollment ----
    if args.holdout_classes > 0:
        fs = few_shot_experiment(spec, loader, args, device)
        doc["few_shot"] = fs
        print_table(
            f"Few-shot enrollment — hold out {fs['held_out_classes']} "
            f"({fs['seen_classes']} seen, {fs['shots']} shots)",
            [
                ("seen acc before enroll", f"{fs['seen_acc_before_enroll']:.3f}"),
                ("seen acc after enroll", f"{fs['seen_acc_after_enroll']:.3f}"),
                ("seen interference drop", f"{fs['seen_interference_drop']:+.3f}"),
                ("held-out acc (enrolled)", f"{fs['held_out_acc']:.3f}"),
                ("overall acc after enroll", f"{fs['overall_acc_after_enroll']:.3f}"),
            ],
        )

    # ---- optional hierarchy export ----
    if args.export_hierarchy:
        from hatchvision.explain import attach_patches, node_patch_uris
        from hatchvision.export import export_hierarchy_pack

        mean, std = spec.normalization()
        patches = node_patch_uris(
            tree, model, memory, val_x[: min(64, val_x.shape[0])], mean, std,
            exemplars=3, max_depth=args.max_depth,
        )
        attach_patches(tree, patches)
        path = export_hierarchy_pack(
            memory, layer, list(spec.class_names), tree, args.export_hierarchy,
            prototype_head=proto,
            config={"tree_temperature": args.tree_temperature,
                    "use_importance_prior": not args.no_importance_prior},
        )
        print(f"\nhierarchy pack exported to {path}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2))
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
