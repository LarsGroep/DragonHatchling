"""Interpretability bundle — the three measured views, composed into one asset.

``vitreous.hebbian`` measures *what the network's units do together*,
``vitreous.dermoscopy`` measures *what a concept is associated with in the
crop*, and ``vitreous.clinical`` measures *how good the classifier actually is*.
Each is honest on its own and each is useless on its own: a graph with no names
is unreadable, a name with no error bar is a guess, and a concept story attached
to an unmeasured classifier is decoration. This module is the single place they
are joined, and the shape it emits (``interpretability_bundle.json``) is what a
HAM10000 workbench consumes.

What each block is measured from
--------------------------------
``graph``
    :func:`vitreous.hebbian.build_hebbian_graph_asset` over a frozen
    :class:`~vitreous.hebbian.HebbianStats`: units, their top-k co-activation
    edges, Louvain communities, Ward concepts, and per-concept class affinity.
    Measured from recorded forward activations only. **Concepts here carry no
    label** and this module does not add one — see the join rule below.
``grounding``
    :func:`vitreous.dermoscopy.ground_concepts` over a ``[N_images,
    N_concepts]`` concept-activation matrix and the ISIC 2018 Task 2 attribute
    table. Measured as Cohen's *d* (with a sample-size-honest lower bound)
    between images that have a dermoscopic attribute and images that do not. A
    concept is named only when a *dermoscopic* attribute clears both the
    effect-size and the minimum-support bar; otherwise it keeps its id and the
    block records why.
``confound_warnings`` / ``confound_probe``
    :func:`vitreous.dermoscopy.probe_metadata_confound` over the same
    activations and the patient-metadata table (body site / age bucket / sex).
    Measured with exactly the same machinery as the dermoscopic attributes and
    then deliberately *not* used as a name: what comes back is a warning. This
    is lay judgment question #2 of ``docs/UX-VISION.md`` ("is that the right
    place to look?") at the concept tier, so it is surfaced **at the top level**
    of the bundle rather than buried inside a per-concept record.
``clinical``
    :func:`vitreous.clinical.clinical_report` over held-out ``y_true`` /
    ``y_prob``: sensitivity/specificity/PPV/NPV with Wilson intervals at the
    shipped decision threshold, the threshold sweep, ROC AUC, per-class and
    melanoma recall, balanced accuracy — and the majority-class baseline that
    accuracy must never be quoted without. The malignant class group comes from
    the dataset's :class:`~vitreous.data.Taxonomy` when one is supplied.

Degradation rule (the reason a block is missing is itself data)
---------------------------------------------------------------
Every block except ``graph`` is optional, and each is independently degradable:
a caller holding only Hebbian statistics gets the graph and nothing else. An
absent block is **never silently dropped and never fabricated** — it is present
with ``{"available": false, "reason_code": ..., "reason": ...}`` and its
measured counters set to ``null``. That is what lets a reader tell "no metadata
was supplied, so no shortcut was looked for" (``available: false``) from "the
shortcut detector ran and flagged nothing" (``available: true, n_flagged: 0``).
Shape mismatches are a different thing entirely and raise :class:`ValueError` —
degrading on those would hide a bug behind an honest-looking null.

The join rule (grounding must not rewrite the graph)
----------------------------------------------------
Graph concepts are unnamed by construction and this module keeps them that way:
grounding is emitted as a **separate block keyed by concept id**, joinable on
``c:<id>`` (:func:`vitreous.hebbian.concept_node_id`), the very id the graph
asset's ``id_convention`` already declares. No label is written back into
``graph.asset``; an unnamed concept therefore stays unnamed everywhere, and a
frontend that renders only the graph cannot accidentally show an invented name.

Serialization
-------------
Every value in the returned bundle is a plain ``bool``/``int``/``float``/``str``
/``list``/``dict``/``None``: numpy scalars are converted and non-finite floats
(``NaN``/``±inf``) become ``null`` with their dotted path recorded in
``provenance.non_finite_coerced``, so a mismeasurement shows up as a named hole
rather than as invalid JSON. Nothing in the bundle is time-stamped: identical
inputs and seed produce a byte-identical document, which is what makes a rebuilt
bundle diffable against the one that shipped.

**Import discipline (M0 rule):** numpy only. ``import vitreous.interpret`` must
not import torch — the whole point is that a bundle can be rebuilt, re-grounded
and re-scored from saved statistics on a laptop with no ML stack installed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .clinical import (
    Z_95,
    clinical_report,
    melanoma_index,
    resolve_class_index,
)
from .dermoscopy import (
    DERMOSCOPIC_ATTRIBUTES,
    DERMOSCOPIC_DISPLAY_NAMES,
    METADATA_ATTRIBUTES,
    METADATA_DISPLAY_NAMES,
    AttributeTable,
    ground_concepts,
    probe_metadata_confound,
)
from .graph import DEFAULT_K, DEFAULT_SEED
from .hebbian import (
    DEFAULT_ACTIVITY_THRESHOLD,
    DEFAULT_MIN_UNITS,
    DEFAULT_N_CONCEPTS,
    DEFAULT_TOP_CLASSES,
    HebbianStats,
    build_hebbian_graph_asset,
    concept_node_id,
)
from .malignancy import malignant_indices

__all__ = [
    "INTERPRET_SCHEMA_VERSION",
    "BUNDLE_KIND",
    "BLOCKS",
    "REASON_CODES",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TARGET_SENSITIVITY",
    "DEFAULT_FOCUS_CLASS",
    "DEFAULT_MIN_SUPPORT",
    "DEFAULT_MIN_EFFECT",
    "DEFAULT_TOP_K",
    "DEFAULT_LABEL_PARTS",
    "DEFAULT_CONFOUND_MARGIN",
    "concept_activations_from_units",
    "build_interpretability_bundle",
]

#: Version of the dict shape emitted by :func:`build_interpretability_bundle`.
INTERPRET_SCHEMA_VERSION = 1

#: ``kind`` discriminator, so a loader can tell this from a graph/pack asset.
BUNDLE_KIND = "interpretability_bundle"

#: The blocks that carry an ``available`` / ``reason_code`` / ``reason`` triple.
BLOCKS: Tuple[str, ...] = ("graph", "grounding", "confound_probe", "clinical")

#: Machine-readable reasons a block can be absent. ``ok`` means it ran.
REASON_CODES: Tuple[str, ...] = (
    "ok",
    "no_concepts",
    "missing_concept_activations",
    "missing_dermoscopic_attributes",
    "missing_metadata_attributes",
    "missing_labels",
    "missing_probabilities",
)

# Defaults chosen to match what the product actually ships / the sibling
# modules already default to, so a bundle built with no knobs touched reads the
# same numbers the workbench does.
DEFAULT_THRESHOLD = 0.2            # LensExplorer's "high-sensitivity default"
DEFAULT_TARGET_SENSITIVITY = 0.95
DEFAULT_FOCUS_CLASS = "Melanoma"   # HAM10000 spec name; "mel" also resolves
DEFAULT_MIN_SUPPORT = 5
DEFAULT_MIN_EFFECT = 0.25
DEFAULT_TOP_K = 4
DEFAULT_LABEL_PARTS = 2
DEFAULT_CONFOUND_MARGIN = 0.1
DEFAULT_SWEEP_POINTS = 101


# --------------------------------------------------------------------------- #
# input coercion
# --------------------------------------------------------------------------- #


def _resolve_stats(stats: Any) -> HebbianStats:
    """Accept :class:`HebbianStats` or its :meth:`~HebbianStats.to_dict` form."""
    if isinstance(stats, HebbianStats):
        return stats
    if isinstance(stats, Mapping):
        return HebbianStats.from_dict(dict(stats))
    raise TypeError(
        f"stats must be a HebbianStats (or its dict form), got "
        f"{type(stats).__name__}"
    )


def _resolve_class_names(
    class_names: Optional[Sequence[str]], num_classes: int
) -> List[str]:
    """Validate ``class_names`` against the statistics' class count.

    ``None`` yields positional placeholders ``class_0..class_{K-1}`` — indices
    rendered as strings, not semantics. Mirrors ``hebbian._resolve_class_names``
    so the bundle and the graph asset can never disagree about class order.
    """
    if class_names is None:
        return [f"class_{i}" for i in range(int(num_classes))]
    names = [str(c) for c in class_names]
    if len(names) != int(num_classes):
        raise ValueError(
            f"class_names has {len(names)} entries but the Hebbian statistics "
            f"declare {int(num_classes)} classes"
        )
    return names


def _concept_units(concept: Any) -> List[int]:
    """Member unit indices of a concept dict (graph asset) or HebbianConcept."""
    units = concept["units"] if isinstance(concept, Mapping) else concept.units
    return [int(u) for u in units]


def concept_activations_from_units(
    unit_activations: Any, concepts: Sequence[Any]
) -> np.ndarray:
    """``[N, U]`` per-image unit activations → ``[N, C]`` concept activations.

    A concept's activation on an image is the **mean activation of its member
    units** — the definition the legacy pipeline used
    (``hatchvision.explain.concepts.concept_scores``) and the one the grounding
    effect sizes assume. Nothing is learned here; this is a projection of a
    measurement onto the concept membership the Hebbian clustering already
    produced.

    ``unit_activations`` must be indexed by *tracked* unit (the same 0..U-1
    space as :attr:`HebbianStats.unit_index`), not by original channel.
    ``concepts`` may be the graph asset's ``concepts`` list or a list of
    :class:`~vitreous.hebbian.HebbianConcept`.
    """
    acts = np.asarray(unit_activations, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"unit_activations must be [N, U], got {acts.shape}")
    if not concepts:
        return np.zeros((acts.shape[0], 0), dtype=np.float64)
    cols = []
    for c in concepts:
        units = _concept_units(c)
        if not units:
            cols.append(np.zeros(acts.shape[0], dtype=np.float64))
            continue
        if max(units) >= acts.shape[1]:
            raise ValueError(
                f"concept references unit {max(units)} but unit_activations has "
                f"only {acts.shape[1]} columns — the activations must cover the "
                f"same tracked units the statistics were recorded on"
            )
        cols.append(acts[:, units].mean(axis=1))
    return np.stack(cols, axis=1)


def _is_matrix_names_pair(obj: Any) -> bool:
    """Is ``obj`` a ``(matrix, names)`` pair rather than a bare matrix?"""
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        return False
    second = obj[1]
    return (
        isinstance(second, (tuple, list))
        and len(second) > 0
        and all(isinstance(n, str) for n in second)
    )


def _resolve_attributes(
    attributes: Any,
    kind: str,
    default_names: Sequence[str],
    default_display: Mapping[str, str],
) -> Optional[Dict[str, Any]]:
    """Normalize an attribute input to ``{matrix, names, display, provenance}``.

    Accepts an :class:`~vitreous.dermoscopy.AttributeTable`, a ``(matrix,
    names)`` pair, or a bare ``[N, A]`` matrix whose column count matches the
    canonical vocabulary (in which case the canonical names are assumed and the
    assumption is recorded). ``None`` passes through as ``None``.
    """
    if attributes is None:
        return None
    if isinstance(attributes, AttributeTable):
        return {
            "matrix": np.asarray(attributes.matrix),
            "names": list(attributes.attribute_names),
            "display": dict(attributes.display_names),
            "image_ids": list(attributes.image_ids),
            "provenance": dict(attributes.provenance),
        }
    names: Optional[List[str]] = None
    if _is_matrix_names_pair(attributes):
        matrix_like, names_like = attributes
        matrix = np.asarray(matrix_like)
        names = [str(n) for n in names_like]
    elif isinstance(attributes, tuple):
        raise ValueError(
            f"{kind} attributes given as a {len(attributes)}-tuple; expected an "
            f"AttributeTable, a (matrix, names) pair, or a [N, A] matrix"
        )
    else:
        matrix = np.asarray(attributes)
    if matrix.ndim != 2:
        raise ValueError(f"{kind} attribute matrix must be [N, A], got {matrix.shape}")
    if names is None:
        if matrix.shape[1] != len(default_names):
            raise ValueError(
                f"{kind} attributes were given as a bare [N, {matrix.shape[1]}] "
                f"matrix, which does not match the canonical {kind} vocabulary "
                f"({len(default_names)} columns). Pass an AttributeTable or a "
                f"(matrix, names) pair so the columns are named."
            )
        names = list(default_names)
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"{kind} attribute matrix has {matrix.shape[1]} columns but "
            f"{len(names)} attribute names"
        )
    return {
        "matrix": matrix,
        "names": names,
        "display": {n: dict(default_display).get(n, n) for n in names},
        "image_ids": None,
        "provenance": {
            "source": "caller-supplied matrix",
            "assumed_vocabulary": names == list(default_names),
        },
    }


def _taxonomy_malignant_map(taxonomy: Any) -> Optional[Mapping[str, Any]]:
    """Pull the ``{class_name: bool}`` malignant map out of a taxonomy input."""
    if taxonomy is None:
        return None
    if hasattr(taxonomy, "malignant"):
        return dict(taxonomy.malignant)
    if isinstance(taxonomy, Mapping):
        if "malignant" in taxonomy and isinstance(taxonomy["malignant"], Mapping):
            return dict(taxonomy["malignant"])
        return dict(taxonomy)
    raise TypeError(
        f"taxonomy must be a vitreous.data.Taxonomy or a mapping, got "
        f"{type(taxonomy).__name__}"
    )


def _taxonomy_json(taxonomy: Any) -> Optional[Dict[str, Any]]:
    if taxonomy is None:
        return None
    if hasattr(taxonomy, "to_json"):
        return dict(taxonomy.to_json())
    return dict(taxonomy) if isinstance(taxonomy, Mapping) else None


# --------------------------------------------------------------------------- #
# block envelopes
# --------------------------------------------------------------------------- #


def _absent(reason_code: str, reason: str, **fields: Any) -> Dict[str, Any]:
    """An unavailable block: says *why*, and reports nothing as measured.

    Counters are ``None`` rather than ``0`` on purpose — "not measured" and
    "measured as zero" are different claims and the bundle must distinguish
    them.
    """
    if reason_code not in REASON_CODES:  # pragma: no cover - guards typos
        raise ValueError(f"unknown reason_code {reason_code!r}")
    return {"available": False, "reason_code": reason_code, "reason": reason, **fields}


def _present(**fields: Any) -> Dict[str, Any]:
    return {"available": True, "reason_code": "ok", "reason": None, **fields}


# --------------------------------------------------------------------------- #
# JSON hygiene
# --------------------------------------------------------------------------- #


def _json_safe(obj: Any, path: str, coerced: List[str]) -> Any:
    """Recursively convert to JSON-native types; log non-finite floats.

    numpy scalars/arrays become Python scalars/lists, tuples become lists, and
    ``NaN``/``±inf`` become ``None`` with ``path`` appended to ``coerced`` — a
    hole a reader can see, rather than a token ``json.dumps`` emits but no
    strict JSON parser accepts.
    """
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if not math.isfinite(value):
            coerced.append(path)
            return None
        return value
    if isinstance(obj, np.ndarray):
        return [_json_safe(v, f"{path}[{i}]", coerced) for i, v in enumerate(obj.tolist())]
    if isinstance(obj, Mapping):
        return {
            str(k): _json_safe(v, f"{path}.{k}", coerced) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, f"{path}[{i}]", coerced) for i, v in enumerate(obj)]
    raise TypeError(f"cannot serialize {type(obj).__name__} at {path}")


# --------------------------------------------------------------------------- #
# the bundle
# --------------------------------------------------------------------------- #


def build_interpretability_bundle(
    stats: Any,
    *,
    class_names: Optional[Sequence[str]],
    y_true: Optional[Sequence[int]] = None,
    y_prob: Optional[Sequence[Sequence[float]]] = None,
    concept_activations: Optional[Any] = None,
    unit_activations: Optional[Any] = None,
    dermoscopic: Optional[Any] = None,
    metadata: Optional[Any] = None,
    taxonomy: Optional[Any] = None,
    threshold: float = DEFAULT_THRESHOLD,
    target_sensitivity: Optional[float] = DEFAULT_TARGET_SENSITIVITY,
    focus_class: Optional[Union[int, str]] = DEFAULT_FOCUS_CLASS,
    positive_classes: Optional[Sequence[Union[int, str]]] = None,
    k: int = DEFAULT_K,
    n_concepts: int = DEFAULT_N_CONCEPTS,
    seed: int = DEFAULT_SEED,
    min_units: int = DEFAULT_MIN_UNITS,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    top_classes: int = DEFAULT_TOP_CLASSES,
    top_k: int = DEFAULT_TOP_K,
    min_support: int = DEFAULT_MIN_SUPPORT,
    min_effect: float = DEFAULT_MIN_EFFECT,
    label_parts: int = DEFAULT_LABEL_PARTS,
    confound_margin: float = DEFAULT_CONFOUND_MARGIN,
    sweep_points: int = DEFAULT_SWEEP_POINTS,
    z: float = Z_95,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose the Hebbian graph, concept grounding and clinical metrics.

    Only ``stats`` and ``class_names`` are required. Everything else adds a
    block; anything left out leaves that block present-but-unavailable with the
    reason recorded (see the module docstring's degradation rule).

    Parameters
    ----------
    stats:
        :class:`~vitreous.hebbian.HebbianStats` (or its dict form) — the frozen
        co-activation measurement the graph is built from.
    class_names:
        Class order for the whole bundle; must match the statistics' class
        count. ``None`` falls back to positional ``class_0..`` placeholders.
    y_true, y_prob:
        Held-out ``[N]`` labels and ``[N, K]`` probabilities. **Both** are
        needed for the clinical block; the split must be held out, since every
        number in that block is otherwise a training-set fiction.
    concept_activations:
        ``[N_images, N_concepts]`` — how strongly each concept fired on each
        grounding image, with rows aligned to ``dermoscopic``/``metadata``
        rows. Column ``c`` must correspond to concept ``c`` of the *graph built
        here* (concept count comes from the clustering, not from
        ``n_concepts``); pass ``unit_activations`` instead to have the
        projection done for you.
    unit_activations:
        ``[N_images, U]`` per-image tracked-unit activations. Converted with
        :func:`concept_activations_from_units` when ``concept_activations`` is
        not given — this is usually the easier thing for a run to save, since
        it does not depend on the clustering parameters.
    dermoscopic:
        ISIC 2018 Task 2 attributes as an
        :class:`~vitreous.dermoscopy.AttributeTable`, a ``(matrix, names)``
        pair, or a bare ``[N, 5]`` matrix. The only vocabulary that can name a
        concept.
    metadata:
        Patient metadata (body site / age / sex) in the same forms. Can only
        ever produce a confound warning — there is no code path from here to a
        label.
    taxonomy:
        A :class:`~vitreous.data.Taxonomy` (or its mapping form). Its malignant
        classes become the positive group of the clinical binary readout, via
        :func:`vitreous.malignancy.malignant_indices`.
    threshold, target_sensitivity, focus_class:
        The operating point actually in use, the sensitivity floor to solve
        for, and the class whose recall is lifted out on its own. A
        ``focus_class`` this dataset does not have is recorded as a note and
        falls back to the melanoma class when there is one.
    positive_classes:
        Explicit override for the binary positive group (names or indices);
        takes precedence over ``taxonomy``.
    provenance:
        Free-form record (dataset, run, checkpoint, split) merged into
        ``provenance``. Deliberately no timestamp is added: the bundle is a
        pure function of its inputs so two rebuilds are byte-identical.

    Returns
    -------
    dict
        JSON-serializable bundle — see the module docstring for the block
        semantics. Top-level keys: ``schema_version``, ``kind``,
        ``class_names``, ``num_classes``, ``num_concepts``, ``graph``,
        ``grounding``, ``confound_probe``, ``confound_warnings``, ``clinical``,
        ``contract``, ``provenance``.

    Raises
    ------
    ValueError
        On shape/length disagreements between the inputs (a bug, not a missing
        input — those degrade instead).
    """
    hstats = _resolve_stats(stats)
    names = _resolve_class_names(class_names, hstats.num_classes)

    # ---------------------------------------------------------------- graph --
    graph_params = {
        "k": int(k),
        "n_concepts": int(n_concepts),
        "seed": int(seed),
        "min_units": int(min_units),
        "activity_threshold": float(activity_threshold),
        "top_classes": int(top_classes),
    }
    asset = build_hebbian_graph_asset(
        hstats,
        names,
        k=k,
        n_concepts=n_concepts,
        seed=seed,
        min_units=min_units,
        activity_threshold=activity_threshold,
        top_classes=top_classes,
        provenance=dict(provenance or {}),
    )
    concepts = list(asset["concepts"])
    concept_ids = [concept_node_id(c["id"]) for c in concepts]
    graph_block = _present(
        asset=asset,
        num_units=int(asset["num_units"]),
        num_concepts=len(concepts),
        num_dead_units=len(asset["dead_units"]),
        concept_ids=list(concept_ids),
        labels=(
            "none — graph concepts are unnamed by construction; names, when "
            "earned, live in the grounding block and join on concept_id"
        ),
    )

    # --------------------------------------------------------- activations --
    acts: Optional[np.ndarray] = None
    activation_source: Optional[str] = None
    if concept_activations is not None:
        acts = np.asarray(concept_activations, dtype=np.float64)
        activation_source = "concept_activations"
        if acts.ndim != 2:
            raise ValueError(
                f"concept_activations must be [N_images, N_concepts], got {acts.shape}"
            )
    elif unit_activations is not None:
        acts = concept_activations_from_units(unit_activations, concepts)
        activation_source = "unit_activations (projected onto concept membership)"
    if acts is not None and acts.shape[1] != len(concepts):
        raise ValueError(
            f"concept activations have {acts.shape[1]} columns but the Hebbian "
            f"clustering produced {len(concepts)} concepts. The column count "
            f"comes from the clustering (small clusters are dropped), not from "
            f"n_concepts={int(n_concepts)} — rebuild the activations against "
            f"this bundle's concepts, or pass unit_activations instead."
        )

    derm = _resolve_attributes(
        dermoscopic, "dermoscopic", DERMOSCOPIC_ATTRIBUTES, DERMOSCOPIC_DISPLAY_NAMES
    )
    meta = _resolve_attributes(
        metadata, "metadata", METADATA_ATTRIBUTES, METADATA_DISPLAY_NAMES
    )
    for table, kind in ((derm, "dermoscopic"), (meta, "metadata")):
        if table is not None and acts is not None:
            if table["matrix"].shape[0] != acts.shape[0]:
                raise ValueError(
                    f"{kind} attributes have {table['matrix'].shape[0]} rows but "
                    f"the concept activations have {acts.shape[0]} — every "
                    f"attribute row must describe the image of the same "
                    f"activation row"
                )

    grounding_params = {
        "top_k": int(top_k),
        "min_support": int(min_support),
        "min_effect": float(min_effect),
        "label_parts": int(label_parts),
        "z": float(z),
        "effect": "Cohen's d, gated on its sample-size-honest lower bound",
    }

    # ------------------------------------------------------------ grounding --
    if not concepts:
        grounding_block = _absent(
            "no_concepts",
            "the Hebbian clustering produced no concept large enough to ground "
            f"(min_units={int(min_units)})",
            n_concepts=0,
            concepts=[],
        )
    elif acts is None:
        grounding_block = _absent(
            "missing_concept_activations",
            "no concept activations were supplied; naming a concept requires a "
            "[N_images, N_concepts] activation matrix measured over the images "
            "the attributes describe",
            n_concepts=None,
            concepts=[],
        )
    elif derm is None:
        grounding_block = _absent(
            "missing_dermoscopic_attributes",
            "no dermoscopic attribute table was supplied (ISIC 2018 Task 2); "
            "concepts keep their ids because the only vocabulary that may name "
            "one is absent",
            n_concepts=None,
            concepts=[],
        )
    else:
        groundings = ground_concepts(
            acts,
            derm["matrix"],
            derm["names"],
            concept_ids=concept_ids,
            display_names=derm["display"],
            top_k=top_k,
            min_support=min_support,
            min_effect=min_effect,
            label_parts=label_parts,
            z=z,
        )
        records = []
        for g, concept in zip(groundings, concepts):
            record = g.to_json()
            record["concept"] = int(concept["id"])
            record["units"] = list(concept["units"])
            records.append(record)
        n_named = sum(1 for r in records if r["named"])
        grounding_block = _present(
            n_images=int(acts.shape[0]),
            n_concepts=len(records),
            n_named=n_named,
            n_unnamed=len(records) - n_named,
            activation_source=activation_source,
            vocabulary={
                "names": list(derm["names"]),
                "display_names": dict(derm["display"]),
                "provenance": dict(derm["provenance"]),
            },
            params=dict(grounding_params),
            concepts=records,
        )

    # -------------------------------------------------------- confound probe --
    confound_warnings: List[Dict[str, Any]] = []
    if not concepts:
        confound_block = _absent(
            "no_concepts",
            "the Hebbian clustering produced no concept to probe",
            n_concepts_probed=None,
            n_flagged=None,
        )
    elif acts is None:
        confound_block = _absent(
            "missing_concept_activations",
            "no concept activations were supplied; the shortcut detector "
            "compares a concept's metadata effect with its dermoscopic effect "
            "and can measure neither without them",
            n_concepts_probed=None,
            n_flagged=None,
        )
    elif meta is None:
        confound_block = _absent(
            "missing_metadata_attributes",
            "no patient-metadata table was supplied; no shortcut was looked "
            "for, which is not the same as none being present",
            n_concepts_probed=None,
            n_flagged=None,
        )
    else:
        probes = probe_metadata_confound(
            acts,
            meta["matrix"],
            meta["names"],
            dermoscopic=None if derm is None else derm["matrix"],
            dermoscopic_names=None if derm is None else derm["names"],
            concept_ids=concept_ids,
            min_support=min_support,
            min_effect=min_effect,
            margin=confound_margin,
            z=z,
        )
        probe_records = []
        for p, concept in zip(probes, concepts):
            record = p.to_json()
            record["concept"] = int(concept["id"])
            probe_records.append(record)
            if record["flagged"]:
                confound_warnings.append(
                    {
                        "concept": record["concept"],
                        "concept_id": record["concept_id"],
                        "concept_index": record["concept_index"],
                        "severity": "caution",
                        "metadata_attribute": record["metadata_attribute"],
                        "metadata_effect": record["metadata_effect"],
                        "metadata_support": record["metadata_support"],
                        "dermoscopic_attribute": record["dermoscopic_attribute"],
                        "dermoscopic_effect": record["dermoscopic_effect"],
                        "margin": record["margin"],
                        "confound_score": record["confound_score"],
                        "message": record["message"],
                    }
                )
        confound_block = _present(
            n_images=int(acts.shape[0]),
            n_concepts_probed=len(probe_records),
            n_flagged=len(confound_warnings),
            has_dermoscopic_counterweight=derm is not None,
            caveat=(
                None
                if derm is not None
                else "measured without a dermoscopic counterweight: with nothing "
                "to compare against, every metadata association looks decisive — "
                "supply ISIC 2018 Task 2 attributes to make this comparative"
            ),
            vocabulary={
                "names": list(meta["names"]),
                "role": "confound probe only — a metadata attribute can never "
                "become a concept name",
                "provenance": dict(meta["provenance"]),
            },
            params={
                "min_support": int(min_support),
                "min_effect": float(min_effect),
                "margin": float(confound_margin),
                "z": float(z),
            },
            concepts=probe_records,
        )

    # ------------------------------------------------------------- clinical --
    clinical_block = _build_clinical_block(
        y_true=y_true,
        y_prob=y_prob,
        names=names,
        taxonomy=taxonomy,
        positive_classes=positive_classes,
        threshold=threshold,
        target_sensitivity=target_sensitivity,
        focus_class=focus_class,
        sweep_points=sweep_points,
        z=z,
        provenance=provenance,
    )

    blocks = {
        "graph": graph_block,
        "grounding": grounding_block,
        "confound_probe": confound_block,
        "clinical": clinical_block,
    }

    bundle: Dict[str, Any] = {
        "schema_version": INTERPRET_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "class_names": list(names),
        "num_classes": len(names),
        "num_concepts": len(concepts),
        "graph": graph_block,
        "grounding": grounding_block,
        "confound_probe": confound_block,
        "confound_warnings": confound_warnings,
        "clinical": clinical_block,
        "contract": {
            "join_key": "concept_id",
            "join": (
                "grounding.concepts[].concept_id == 'c:' + "
                "graph.asset.concepts[].id == confound_probe.concepts[].concept_id"
            ),
            "unnamed_concepts": (
                "a concept with no qualifying dermoscopic attribute has "
                "label=null and must be rendered as its concept_id; no label is "
                "ever written into graph.asset"
            ),
            "metadata": (
                "patient metadata can only produce a confound warning, never a "
                "name — confound_warnings is top-level so it cannot be missed"
            ),
            "accuracy": (
                "clinical.headline.accuracy carries the majority-class baseline "
                "in the same object; accuracy must never be shown without it"
            ),
            "absent_blocks": (
                "a block with available=false was not measured; its counters are "
                "null, which is different from a measured zero"
            ),
        },
        "provenance": {
            **dict(provenance or {}),
            "generator": "vitreous.interpret.build_interpretability_bundle",
            "schema_version": INTERPRET_SCHEMA_VERSION,
            "providers": [
                {
                    "name": "vitreous.hebbian.build_hebbian_graph_asset",
                    "block": "graph",
                    "ran": True,
                    "params": graph_params,
                },
                {
                    "name": "vitreous.dermoscopy.ground_concepts",
                    "block": "grounding",
                    "ran": bool(grounding_block["available"]),
                    "params": grounding_params,
                },
                {
                    "name": "vitreous.dermoscopy.probe_metadata_confound",
                    "block": "confound_probe",
                    "ran": bool(confound_block["available"]),
                    "params": {
                        "min_support": int(min_support),
                        "min_effect": float(min_effect),
                        "margin": float(confound_margin),
                        "z": float(z),
                    },
                },
                {
                    "name": "vitreous.clinical.clinical_report",
                    "block": "clinical",
                    "ran": bool(clinical_block["available"]),
                    "params": {
                        "threshold": float(threshold),
                        "target_sensitivity": (
                            None if target_sensitivity is None else float(target_sensitivity)
                        ),
                        "sweep_points": int(sweep_points),
                        "z": float(z),
                    },
                },
            ],
            "blocks": {
                name: {
                    "available": bool(block["available"]),
                    "reason_code": block["reason_code"],
                    "reason": block["reason"],
                }
                for name, block in blocks.items()
            },
            "skipped": [
                {
                    "block": name,
                    "reason_code": block["reason_code"],
                    "reason": block["reason"],
                }
                for name, block in blocks.items()
                if not block["available"]
            ],
            "determinism": (
                "no timestamp is recorded: identical inputs and seed produce a "
                "byte-identical bundle"
            ),
            "non_finite_coerced": [],
        },
    }

    coerced: List[str] = []
    safe = _json_safe(bundle, "bundle", coerced)
    safe["provenance"]["non_finite_coerced"] = coerced
    return safe


# --------------------------------------------------------------------------- #
# clinical block
# --------------------------------------------------------------------------- #


def _build_clinical_block(
    *,
    y_true: Optional[Sequence[int]],
    y_prob: Optional[Sequence[Sequence[float]]],
    names: List[str],
    taxonomy: Any,
    positive_classes: Optional[Sequence[Union[int, str]]],
    threshold: float,
    target_sensitivity: Optional[float],
    focus_class: Optional[Union[int, str]],
    sweep_points: int,
    z: float,
    provenance: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """The clinical readout, shaped so it cannot be rendered dishonestly.

    The block carries ``baselines`` at its top, and ``headline.accuracy`` is a
    single object holding the accuracy, its Wilson interval, **and** the
    majority-class rate it must be compared against — so no consumer can lift
    the accuracy out of the bundle without carrying its floor along.
    """
    if y_true is None:
        return _absent(
            "missing_labels",
            "no held-out labels were supplied; every clinical number needs "
            "y_true and y_prob from a split the model did not train on",
            report=None,
            summary=None,
            headline=None,
            baselines=None,
        )
    if y_prob is None:
        return _absent(
            "missing_probabilities",
            "no held-out probabilities were supplied; sensitivity at a "
            "threshold cannot be measured from labels alone",
            report=None,
            summary=None,
            headline=None,
            baselines=None,
        )

    probs = np.asarray(y_prob, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"y_prob must be [N, K], got {probs.shape}")
    if probs.shape[1] != len(names):
        raise ValueError(
            f"y_prob has {probs.shape[1]} columns but class_names has "
            f"{len(names)} entries"
        )

    notes: List[str] = []

    # -- positive (malignant) group -------------------------------------- #
    positive_idx: List[int] = []
    source = "none"
    if positive_classes:
        positive_idx = [resolve_class_index(names, c) for c in positive_classes]
        source = "explicit positive_classes"
    elif taxonomy is not None:
        mal = _taxonomy_malignant_map(taxonomy)
        positive_idx = [int(i) for i in malignant_indices(names, mal or {})]
        source = "taxonomy.malignant"
        if not positive_idx:
            notes.append(
                "the taxonomy flags no class as malignant, so there is no "
                "binary malignancy readout"
            )
            source = "taxonomy.malignant (empty)"
    if not positive_idx:
        fallback = melanoma_index(names)
        if fallback is not None:
            positive_idx = [fallback]
            source = "melanoma class (no taxonomy and no positive_classes given)"
            notes.append(
                "no taxonomy was supplied; the binary readout covers melanoma "
                "alone rather than the full malignant group"
            )

    # -- focus class ------------------------------------------------------ #
    focus_idx: Optional[int] = None
    if focus_class is not None:
        try:
            focus_idx = resolve_class_index(names, focus_class)
        except ValueError:
            notes.append(
                f"focus class {focus_class!r} is not one of this dataset's "
                f"classes; falling back to the melanoma class if there is one"
            )
    if focus_idx is None:
        focus_idx = melanoma_index(names)

    report = clinical_report(
        y_true,
        probs,
        class_names=names,
        positive_classes=positive_idx or None,
        threshold=threshold,
        target_sensitivity=target_sensitivity,
        focus_class=focus_idx,
        sweep_points=sweep_points,
        z=z,
        provenance={
            **dict(provenance or {}),
            "positive_class_source": source,
            "split": (provenance or {}).get("split", "held-out (caller-declared)"),
        },
    )

    base = dict(report["baselines"])
    multi = report["multiclass"]
    accuracy = dict(multi["accuracy"])
    # The pairing that makes the number honest — same object, not a sibling a
    # renderer might drop.
    accuracy.update(
        {
            "majority_class_rate": base["majority_class_rate"],
            "majority_class_name": base["majority_class_name"],
            "uniform_chance": base["uniform_chance"],
            "meaningful_floor": base["meaningful_floor"],
            "imbalanced": base["imbalanced"],
            "margin_over_majority_class": (
                None
                if multi["accuracy"]["value"] is None
                else float(multi["accuracy"]["value"] - base["majority_class_rate"])
            ),
            "note": base["note"],
        }
    )

    binary = report.get("binary") or {}
    at = binary.get("at_threshold") or {}
    headline = {
        "summary": report["summary"],
        "accuracy": accuracy,
        "balanced_accuracy": multi["balanced_accuracy"],
        "balanced_accuracy_classes_counted": multi["balanced_accuracy_classes_counted"],
        "focus_class": report.get("focus_class"),
        "positive_label": binary.get("label"),
        "threshold": float(threshold) if binary else None,
        "sensitivity_at_threshold": at.get("sensitivity"),
        "specificity_at_threshold": at.get("specificity"),
        "ppv_at_threshold": at.get("ppv"),
        "roc_auc": binary.get("roc_auc"),
        "target_sensitivity_operating_point": binary.get(
            "target_sensitivity_operating_point"
        ),
    }

    return _present(
        n=int(report["n"]),
        report=report,
        summary=report["summary"],
        baselines=base,
        headline=headline,
        positive_classes=[names[i] for i in positive_idx],
        positive_class_indices=list(positive_idx),
        positive_class_source=source,
        taxonomy=_taxonomy_json(taxonomy),
        threshold=float(threshold),
        target_sensitivity=(
            None if target_sensitivity is None else float(target_sensitivity)
        ),
        notes=notes,
    )
