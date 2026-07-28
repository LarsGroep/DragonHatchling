#!/usr/bin/env python3
"""Build the interpretability bundle from saved measurements — no retraining.

The analysis half of the pipeline, run offline: re-cluster the Hebbian
co-activation memory, re-ground its concepts against the ISIC 2018 Task 2
dermoscopic attributes, re-probe them for the body-site confound, and re-score
the held-out split — all from files a run leaves behind. This is the
``scripts/rebuild_graph.py`` workflow (iterate clustering/grounding parameters
in seconds instead of re-training for an hour), brought up to the honest
vocabulary and made **torch-free**: numpy + ``vitreous`` only, so it runs on a
laptop with no ML stack installed.

Inputs (every one of them optional except the statistics — a missing input
degrades its block and the bundle records why, it never fabricates one):

``--stats``        ``hebbian_stats.npz`` written by ``HebbianStats.save_npz``.
``--activations``  ``.npz`` with ``unit_activations`` ``[N, U]`` (or
                   ``concept_activations`` ``[N, C]``) and, ideally,
                   ``image_ids`` ``[N]`` — without ids nothing can be joined to
                   the attribute tables by image and rows must already align.
``--isic-task2``   directory of ISIC 2018 Task 2 masks → the only vocabulary
                   that may name a concept.
``--metadata-csv`` ``HAM10000_metadata.csv`` → the confound probe (body site /
                   age / sex can warn about a concept, never name one).
``--eval``         ``.npz`` with ``y_true`` ``[N]`` and ``y_prob`` ``[N, K]``
                   from a **held-out** split → the clinical block.

Examples::

    # everything a full HAM10000 run can supply
    python scripts/build_interpretability_bundle.py \\
        --stats exports/ham10000/hebbian_stats.npz \\
        --dataset ham10000 \\
        --isic-task2 data/isic2018/ISIC2018_Task2_Training_GroundTruth_v3 \\
        --metadata-csv data/ham10000/HAM10000_metadata.csv \\
        --activations exports/ham10000/probe_activations.npz \\
        --eval exports/ham10000/heldout.npz \\
        --n-concepts 16 --threshold 0.2 \\
        --out exports/ham10000/interpretability.json

    # statistics only: the graph, plus a bundle that says why the rest is absent
    python scripts/build_interpretability_bundle.py \\
        --stats exports/ham10000/hebbian_stats.npz --dataset ham10000 \\
        --out exports/ham10000/interpretability.json

    # no data at all — synthesize inputs and write a bundle (smoke test)
    python scripts/build_interpretability_bundle.py --demo --out /tmp/bundle.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Run from a source checkout without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core" / "src"))

import numpy as np

from vitreous.dermoscopy import (
    DERMOSCOPIC_ATTRIBUTES,
    AttributeTable,
    build_metadata_attributes,
    load_isic2018_task2_attributes,
)
from vitreous.hebbian import HebbianStats
from vitreous.interpret import build_interpretability_bundle

ACTIVATION_KEYS = ("unit_activations", "concept_activations", "activations")
IMAGE_ID_KEYS = ("image_ids", "ids", "image_id")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _load_class_names(args: argparse.Namespace) -> Tuple[Optional[List[str]], Any, Dict[str, Any]]:
    """Class names + taxonomy from --class-names / --manifest / --dataset."""
    meta: Dict[str, Any] = {}
    names: Optional[List[str]] = None
    taxonomy = None
    if args.dataset:
        from vitreous.data import get_dataset

        spec = get_dataset(args.dataset).spec
        names = list(spec.class_names)
        taxonomy = spec.taxonomy
        meta["dataset"] = spec.name
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        names = list(manifest.get("class_names") or names or [])
        meta["dataset"] = manifest.get("dataset", meta.get("dataset"))
        meta["backbone"] = manifest.get("backbone")
        meta["manifest"] = str(args.manifest)
    if args.class_names:
        names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    return names, taxonomy, meta


def _load_activations(
    path: Path,
) -> Tuple[str, np.ndarray, Optional[List[str]]]:
    """Read ``[N, U]``/``[N, C]`` activations (+ optional image ids) from a npz."""
    with np.load(path, allow_pickle=False) as z:
        key = next((k for k in ACTIVATION_KEYS if k in z), None)
        if key is None:
            raise SystemExit(
                f"{path}: expected one of {list(ACTIVATION_KEYS)}; found {list(z)}"
            )
        acts = np.asarray(z[key], dtype=np.float64)
        ids = None
        id_key = next((k for k in IMAGE_ID_KEYS if k in z), None)
        if id_key is not None:
            ids = [str(v) for v in np.asarray(z[id_key]).ravel().tolist()]
    if acts.ndim != 2:
        raise SystemExit(f"{path}: {key} must be [N, *], got {acts.shape}")
    if ids is not None and len(ids) != acts.shape[0]:
        raise SystemExit(
            f"{path}: {len(ids)} image ids for {acts.shape[0]} activation rows"
        )
    kind = "concept" if key == "concept_activations" else "unit"
    return kind, acts, ids


def _load_eval(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        for key in ("y_true", "y_prob"):
            if key not in z:
                raise SystemExit(f"{path}: missing '{key}' (found {list(z)})")
        return np.asarray(z["y_true"]), np.asarray(z["y_prob"], dtype=np.float64)


def _read_metadata_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# row alignment — the join every real run needs and few of them keep
# --------------------------------------------------------------------------- #


def _align(
    acts: np.ndarray,
    act_ids: Optional[Sequence[str]],
    tables: Dict[str, AttributeTable],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Reduce activations + attribute tables to their common images, in order.

    With image ids on every side this is an inner join (and it prints what it
    dropped). Without them the rows are assumed to already correspond, which is
    checked rather than trusted.
    """
    if not tables:
        return acts, {}
    if act_ids is None:
        for name, table in tables.items():
            if table.n_images != acts.shape[0]:
                raise SystemExit(
                    f"the activation file carries no image_ids, so its {acts.shape[0]} "
                    f"rows must line up 1:1 with the {name} table's {table.n_images} "
                    f"rows — they do not. Save image_ids alongside the activations."
                )
        return acts, {name: np.asarray(t.matrix) for name, t in tables.items()}

    common = list(act_ids)
    for table in tables.values():
        known = set(table.image_ids)
        common = [i for i in common if i in known]
    if not common:
        raise SystemExit(
            "no image id is shared by the activations and the attribute tables; "
            "check that both use the ISIC image id (e.g. 'ISIC_0000000')"
        )
    dropped = len(act_ids) - len(common)
    if dropped:
        print(f"  aligned on image id: kept {len(common)}, dropped {dropped}")
    act_index = {img: i for i, img in enumerate(act_ids)}
    rows = np.asarray([act_index[i] for i in common], dtype=np.int64)
    out = {}
    for name, table in tables.items():
        index = {img: i for i, img in enumerate(table.image_ids)}
        out[name] = np.asarray(table.matrix)[
            np.asarray([index[i] for i in common], dtype=np.int64)
        ]
    return acts[rows], out


# --------------------------------------------------------------------------- #
# --demo: synthesize every input so the script is runnable with no data
# --------------------------------------------------------------------------- #


def _demo_inputs(seed: int = 0):
    """Planted stats + probe + split: one named, one shortcut, one unnamed concept.

    Four disjoint co-activation blocks, one per class. On the probe set, block 0
    fires with ``pigment_network``, block 1 with ``location: foot`` (the
    confound), block 2 with nothing, block 3 with ``globules``.
    """
    blocks, per_block = 4, 8
    units = blocks * per_block
    names = ["Melanoma", "Melanocytic nevi", "Basal cell carcinoma", "Benign keratosis"]
    taxonomy = {
        "Melanoma": True,
        "Melanocytic nevi": False,
        "Basal cell carcinoma": True,
        "Benign keratosis": False,
    }
    rng = np.random.default_rng(seed)

    n_train = 400
    labels = np.repeat(np.arange(blocks), n_train // blocks)
    acts = np.zeros((n_train, units))
    for s, b in enumerate(labels):
        acts[s, b * per_block : (b + 1) * per_block] = 1.0 + 0.05 * rng.standard_normal(
            per_block
        )
    a_hat = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
    stats = HebbianStats(
        coact=a_hat.T @ a_hat / n_train,
        mean_act=a_hat.mean(axis=0),
        class_act=np.stack([a_hat[labels == c].mean(axis=0) for c in range(blocks)]),
        class_count=np.array([(labels == c).sum() for c in range(blocks)], dtype=float),
        unit_index=np.arange(units),
        layer="neurons",
        num_updates=n_train // 16,
    )

    n_probe = 240
    image_ids = [f"ISIC_{i:07d}" for i in range(n_probe)]
    derm = (rng.random((n_probe, len(DERMOSCOPIC_ATTRIBUTES))) < 0.4).astype(np.uint8)
    on_foot = rng.random(n_probe) < 0.35
    meta_rows = [
        {
            "image_id": image_ids[i],
            "sex": "male" if i % 2 else "female",
            "age": str(40 + (i % 5) * 10),
            "localization": "foot" if on_foot[i] else "back",
        }
        for i in range(n_probe)
    ]
    unit_acts = rng.standard_normal((n_probe, units)) * 0.4
    unit_acts[:, 0:per_block] += 2.5 * derm[:, [0]]                    # pigment network
    unit_acts[:, per_block : 2 * per_block] += 2.5 * on_foot[:, None]  # the shortcut
    unit_acts[:, 3 * per_block :] += 2.5 * derm[:, [4]]                # globules

    derm_table = AttributeTable(
        image_ids=image_ids,
        attribute_names=list(DERMOSCOPIC_ATTRIBUTES),
        matrix=derm,
        provenance={"dataset": "SYNTHETIC — --demo, not ISIC 2018 Task 2"},
    )
    meta_table = build_metadata_attributes(meta_rows)

    n_eval = 500
    y = rng.integers(0, blocks, n_eval)
    logits = rng.standard_normal((n_eval, blocks)) * 0.8
    logits[np.arange(n_eval), y] += 1.8
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_prob = e / e.sum(axis=1, keepdims=True)

    return {
        "stats": stats,
        "class_names": names,
        "taxonomy": taxonomy,
        "unit_activations": unit_acts,
        "image_ids": image_ids,
        "dermoscopic": derm_table,
        "metadata": meta_table,
        "y_true": y,
        "y_prob": y_prob,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _report(bundle: Dict[str, Any]) -> None:
    """Print what was measured — and, as loudly, what was not."""
    graph = bundle["graph"]
    print(
        f"graph: {graph['num_units']} units · {graph['num_concepts']} concepts "
        f"· {graph['num_dead_units']} dead"
    )
    grounding = bundle["grounding"]
    if grounding["available"]:
        print(
            f"grounding: {grounding['n_named']}/{grounding['n_concepts']} concepts "
            f"named from {grounding['n_images']} images"
        )
        for record in grounding["concepts"]:
            print(f"  {record['concept_id']}: {record['display']}")
    else:
        print(f"grounding: unavailable — {grounding['reason']}")

    warnings = bundle["confound_warnings"]
    probe = bundle["confound_probe"]
    if probe["available"]:
        print(f"confound probe: {len(warnings)} concept(s) flagged")
        for warning in warnings:
            print(f"  WARNING {warning['concept_id']}: {warning['message']}")
    else:
        print(f"confound probe: unavailable — {probe['reason']}")

    clinical = bundle["clinical"]
    if clinical["available"]:
        print(f"clinical: {clinical['summary']}")
    else:
        print(f"clinical: unavailable — {clinical['reason']}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stats", default=None,
                   help="hebbian_stats.npz (HebbianStats.save_npz)")
    p.add_argument("--manifest", default=None,
                   help="bundle manifest.json (class names)")
    p.add_argument("--dataset", default=None,
                   help="registered vitreous dataset (class names + taxonomy)")
    p.add_argument("--class-names", default=None,
                   help="comma-separated class names (overrides)")
    p.add_argument("--activations", default=None,
                   help="npz: unit_activations|concept_activations [+ image_ids]")
    p.add_argument("--isic-task2", default=None,
                   help="ISIC 2018 Task 2 mask directory")
    p.add_argument("--metadata-csv", default=None,
                   help="HAM10000_metadata.csv (confound probe only)")
    p.add_argument("--eval", default=None,
                   help="npz with held-out y_true [N] and y_prob [N, K]")
    p.add_argument("--k", type=int, default=8, help="co-activation edges kept per unit")
    p.add_argument("--n-concepts", type=int, default=16)
    p.add_argument("--min-units", type=int, default=2)
    p.add_argument("--activity-threshold", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-support", type=int, default=5,
                   help="images needed before a concept may be named")
    p.add_argument("--min-effect", type=float, default=0.25,
                   help="effect_lo bar a dermoscopic attribute must clear to name")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="the decision threshold actually shipped")
    p.add_argument("--target-sensitivity", type=float, default=0.95)
    p.add_argument("--focus-class", default="Melanoma")
    p.add_argument("--demo", action="store_true", help="synthesize every input (no data needed)")
    p.add_argument("--indent", type=int, default=2, help="JSON indent; 0 writes compact")
    p.add_argument("--out", required=True, help="write the bundle JSON here")
    args = p.parse_args()

    provenance: Dict[str, Any] = {"generator": "scripts/build_interpretability_bundle.py"}
    kwargs: Dict[str, Any] = {}

    if args.demo:
        demo = _demo_inputs(seed=args.seed)
        stats = demo.pop("stats")
        class_names = demo.pop("class_names")
        demo.pop("image_ids")
        kwargs.update(demo)
        provenance.update(
            {
                "dataset": "SYNTHETIC — --demo",
                "warning": "every number in this bundle comes from synthetic data; "
                "it demonstrates the shape, not a model",
            }
        )
        taxonomy = kwargs.pop("taxonomy")
    else:
        if not args.stats:
            p.error("--stats is required (or use --demo)")
        stats = HebbianStats.load_npz(args.stats)
        class_names, taxonomy, meta = _load_class_names(args)
        provenance.update({k: v for k, v in meta.items() if v is not None})
        provenance["stats"] = str(args.stats)
        if class_names is None:
            p.error("need --class-names, --manifest or --dataset to know the class names")
        if len(class_names) != stats.num_classes:
            p.error(
                f"class name count ({len(class_names)}) != classes in the statistics "
                f"({stats.num_classes})"
            )
        print(f"stats: {stats.num_units} units · {stats.num_classes} classes "
              f"· {stats.num_updates} updates · layer '{stats.layer}'")

        acts_kind, acts, act_ids = (None, None, None)
        if args.activations:
            acts_kind, acts, act_ids = _load_activations(Path(args.activations))
            print(f"activations: {acts.shape[0]} images × {acts.shape[1]} {acts_kind}s"
                  + ("" if act_ids else " (no image_ids — rows must already align)"))

        tables: Dict[str, AttributeTable] = {}
        if args.isic_task2:
            derm_table = load_isic2018_task2_attributes(
                args.isic_task2, image_ids=act_ids
            )
            tables["dermoscopic"] = derm_table
            print(f"ISIC 2018 Task 2: {derm_table.n_images} images · support "
                  f"{dict(zip(derm_table.attribute_names, derm_table.support().tolist()))}")
            provenance["isic2018_task2_root"] = str(args.isic_task2)
        if args.metadata_csv:
            meta_table = build_metadata_attributes(_read_metadata_rows(Path(args.metadata_csv)))
            tables["metadata"] = meta_table
            print(f"metadata: {meta_table.n_images} rows (confound probe only)")
            provenance["metadata_csv"] = str(args.metadata_csv)

        if acts is not None:
            acts, matrices = _align(acts, act_ids, tables)
            if "dermoscopic" in matrices:
                kwargs["dermoscopic"] = (
                    matrices["dermoscopic"],
                    list(tables["dermoscopic"].attribute_names),
                )
            if "metadata" in matrices:
                kwargs["metadata"] = (
                    matrices["metadata"],
                    list(tables["metadata"].attribute_names),
                )
            kwargs[
                "concept_activations" if acts_kind == "concept" else "unit_activations"
            ] = acts
        elif tables:
            print("  attribute tables loaded but no --activations: nothing to ground")

        if args.eval:
            y_true, y_prob = _load_eval(Path(args.eval))
            kwargs["y_true"] = y_true
            kwargs["y_prob"] = y_prob
            provenance["eval"] = str(args.eval)
            provenance.setdefault("split", "held-out (caller-declared)")

    bundle = build_interpretability_bundle(
        stats,
        class_names=class_names,
        taxonomy=taxonomy,
        threshold=args.threshold,
        target_sensitivity=args.target_sensitivity,
        focus_class=args.focus_class,
        k=args.k,
        n_concepts=args.n_concepts,
        seed=args.seed,
        min_units=args.min_units,
        activity_threshold=args.activity_threshold,
        min_support=args.min_support,
        min_effect=args.min_effect,
        provenance=provenance,
        **kwargs,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(bundle, separators=(",", ":"), allow_nan=False)
        if args.indent <= 0
        else json.dumps(bundle, indent=args.indent, allow_nan=False)
    )
    out.write_text(text)

    _report(bundle)
    print(f"\ninterpretability bundle → {out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
