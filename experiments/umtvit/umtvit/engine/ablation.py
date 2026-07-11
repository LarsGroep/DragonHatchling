"""Ablation runner (ARCHITECTURE §6.5, §9 row U5).

The science of the experiment: train the same recipe with one axis toggled at a
time and put the §6 metrics side by side. :class:`AblationRunner` deep-copies a
base :class:`~umtvit.config.Config`, applies each variant's dotted-path
overrides (plus optional shared short-budget overrides), trains it with the U4
:class:`~umtvit.engine.trainer.Trainer` from a fixed seed, scores it with
:func:`~umtvit.eval.report.run_evaluation`, and assembles a comparison table
(markdown + dict).

:data:`ABLATIONS` is the canonical axis set the design calls out (§6.5):

- ``no_cross_attention`` — ``model.cross_rounds = 0`` (dual streams straight to
  fusion; isolates the cross-scale bridge's contribution).
- ``full_pair`` — ``model.cross_attention = "full_pair"`` (DSCATNet-style dense
  cross-attention vs the default CLS-bridged variant).
- ``no_som`` — ``loss.lambda_som = 0`` (drops the SOM quantization term).
- ``no_order`` — ``loss.lambda_order = loss.order_monotone = 0`` (removes the
  layer-scale ordering bias, both the frequency and monotone-centroid parts).
- ``smooth_hwz`` — ``loss.smooth_axes = [h, w, z]`` (restores the Z-axis
  smoothness term that fights depth differentiation).
- ``kohonen_ema`` — ``model.som_update = "kohonen_ema"`` (classical Hebbian EMA
  SOM update vs the differentiable gradient update).

Each variant is ``(name, {dotted_key: value, ...})``; overrides address the
config by dotted path (e.g. ``"loss.lambda_som"``). The runner never mutates the
base config in place.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from umtvit.config import Config
from umtvit.engine.trainer import Trainer
from umtvit.eval.report import run_evaluation
from umtvit.models.model import UMTViT
from umtvit.models.som3d import Soft3DSOM

__all__ = ["AblationRunner", "ABLATIONS"]

Variant = Tuple[str, Mapping[str, Any]]

# The canonical ablation axes (ARCHITECTURE §6.5). Overrides are dotted paths
# into the config; every value here is valid against the schema.
ABLATIONS: List[Variant] = [
    ("no_cross_attention", {"model.cross_rounds": 0}),
    ("full_pair", {"model.cross_attention": "full_pair"}),
    ("no_som", {"loss.lambda_som": 0.0}),
    ("no_order", {"loss.lambda_order": 0.0, "loss.order_monotone": 0.0}),
    ("smooth_hwz", {"loss.smooth_axes": ["h", "w", "z"]}),
    ("kohonen_ema", {"model.som_update": "kohonen_ema"}),
]

# Columns of the comparison table: (header, path-into-metrics-dict). ``None``
# leaves mean the metric was skipped (unlabeled) or undefined for that variant.
_COLUMNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("probe_acc", ("linear_probe", "accuracy")),
    ("knn_acc", ("knn", "accuracy")),
    ("quant_err", ("som", "quantization_error")),
    ("topo_err", ("som", "topographic_error")),
    ("dead_frac", ("som", "dead_neuron_fraction")),
    ("trust", ("manifold", "trustworthiness")),
    ("continuity", ("manifold", "continuity")),
    ("z_monotone", ("zaxis", "monotone_decreasing")),
]


def _apply_override(cfg: Config, dotted_key: str, value: Any) -> None:
    """Set ``cfg.<dotted_key> = value`` in place (navigating nested dataclasses)."""
    parts = dotted_key.split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise KeyError(f"unknown config path segment {part!r} in {dotted_key!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise KeyError(f"unknown config field {leaf!r} in {dotted_key!r}")
    setattr(target, leaf, value)


def _dig(metrics: Mapping[str, Any], path: Tuple[str, ...]) -> Any:
    """Follow ``path`` into a nested metrics dict; ``None`` if any hop is missing."""
    node: Any = metrics
    for key in path:
        if not isinstance(node, Mapping) or node.get(key) is None:
            return None
        node = node[key]
    return node


def _fmt_cell(value: Any) -> str:
    """Format one table cell (``n/a`` for None/NaN, bools verbatim, floats 4 dp)."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "n/a"
    return f"{f:.4f}"


class AblationRunner:
    """Train + evaluate a set of config variants and tabulate the §6 metrics.

    Args:
        base_config: The reference :class:`~umtvit.config.Config`. Deep-copied
            per variant; never mutated in place.
        variants: Sequence of ``(name, override_dict)`` where ``override_dict``
            maps dotted config paths to values (e.g.
            ``{"loss.lambda_som": 0.0}``). Defaults to :data:`ABLATIONS`.
        train_data: Training data for the :class:`Trainer` — a DataLoader /
            dataset / factory, **or** a callable ``fn(cfg) -> data`` invoked per
            variant (so image-size-dependent data can track the variant config).
        eval_datasets: Mapping ``split -> eval-mode dataset`` for
            :func:`run_evaluation`, **or** a callable ``fn(cfg) -> mapping``.
        budget_overrides: Optional dotted overrides applied to **every** variant
            before its own overrides (e.g. ``{"train.max_steps": 20,
            "train.epochs": None}``) to shrink the training budget.
        seed: Seed re-applied before each variant's model/SOM init and passed to
            :func:`run_evaluation`, so variants differ only by their overrides.
        eval_kwargs: Extra keyword args forwarded to :func:`run_evaluation`.
    """

    def __init__(
        self,
        base_config: Config,
        variants: Optional[Sequence[Variant]] = None,
        *,
        train_data: Any,
        eval_datasets: Any,
        budget_overrides: Optional[Mapping[str, Any]] = None,
        seed: int = 0,
        eval_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.base_config = base_config
        self.variants: List[Variant] = list(
            variants if variants is not None else ABLATIONS
        )
        self.train_data = train_data
        self.eval_datasets = eval_datasets
        self.budget_overrides = dict(budget_overrides or {})
        self.seed = int(seed)
        self.eval_kwargs = dict(eval_kwargs or {})
        self.results: Dict[str, Any] = {}

    def _build_config(self, override: Mapping[str, Any]) -> Config:
        cfg = copy.deepcopy(self.base_config)
        # A deep-copied validated config has model.image_size already set; clear
        # it so re-validation re-derives it and never trips the equality check if
        # a variant changes dataset.image_size.
        cfg.model.image_size = None
        for key, value in self.budget_overrides.items():
            _apply_override(cfg, key, value)
        for key, value in override.items():
            _apply_override(cfg, key, value)
        return cfg.validate()

    @staticmethod
    def _resolve(source: Any, cfg: Config) -> Any:
        """Call a factory ``fn(cfg)`` or pass a static source through unchanged."""
        if callable(source) and not hasattr(source, "__getitem__"):
            return source(cfg)
        return source

    def run(self) -> Dict[str, Any]:
        """Train + evaluate every variant; return the comparison dict + table.

        Returns:
            ``{"results": {name: metrics}, "columns": [...], "rows": [...],
            "table": markdown}`` — ``results`` holds each variant's full
            :func:`run_evaluation` dict, ``rows`` the flat per-variant cell
            values, and ``table`` the rendered markdown.
        """
        self.results = {}
        for name, override in self.variants:
            cfg = self._build_config(override)
            torch.manual_seed(self.seed)
            model = UMTViT(cfg)
            som = Soft3DSOM.from_config(cfg)
            data = self._resolve(self.train_data, cfg)
            Trainer(cfg, model, som, data).train()
            eval_ds = self._resolve(self.eval_datasets, cfg)
            metrics = run_evaluation(
                model, som, cfg, eval_ds, seed=self.seed, **self.eval_kwargs
            )
            self.results[name] = metrics
        return self.table()

    def table(self) -> Dict[str, Any]:
        """Assemble the comparison table (markdown + structured rows) from results."""
        columns = [header for header, _ in _COLUMNS]
        rows: List[Dict[str, Any]] = []
        for name, _ in self.variants:
            metrics = self.results.get(name, {})
            row: Dict[str, Any] = {"variant": name}
            for header, path in _COLUMNS:
                row[header] = _dig(metrics, path)
            rows.append(row)

        header_line = "| variant | " + " | ".join(columns) + " |"
        sep_line = "|" + "---|" * (len(columns) + 1)
        body = [
            "| "
            + " | ".join([row["variant"]] + [_fmt_cell(row[h]) for h in columns])
            + " |"
            for row in rows
        ]
        table_md = "\n".join(
            ["## Ablation comparison", "", header_line, sep_line, *body]
        )
        return {
            "results": self.results,
            "columns": ["variant", *columns],
            "rows": rows,
            "table": table_md,
        }
