"""Run-configuration schema for UMT-ViT (ARCHITECTURE §4, §5).

This module is the **single source of truth** for a UMT-ViT run. Every model,
data, and training knob is a typed dataclass field here; nothing downstream
reads dataset specifics, hyper-parameters, or paths from anywhere else. A run
is fully described by one YAML file that deserialises into :class:`Config`.

Contract:

- :func:`load_config` reads a YAML file, builds a :class:`Config`, and runs
  :meth:`Config.validate` — so a returned ``Config`` is always structurally
  valid.
- :meth:`Config.to_yaml` serialises back to YAML such that
  ``load_config(path)`` of the written file compares equal to the original
  (round-trip stability; see ``tests/test_config.py``).
- Validation rejects malformed configs with a :class:`ConfigError` whose
  message **names the offending field** (dotted path, e.g.
  ``dataset.splits``), so misconfiguration is diagnosable without a traceback
  read.

Import discipline: this module depends only on the standard library and
PyYAML. It never imports torch/numpy/PIL, so config parsing and validation
stay cheap and side-effect free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

__all__ = [
    "ConfigError",
    "SplitConfig",
    "DatasetConfig",
    "ModelConfig",
    "LossConfig",
    "TrainConfig",
    "Config",
    "load_config",
]

# Allowed enum values. Kept as module constants so error messages can quote
# the exact accepted set and tests can reference one authority.
LOADERS = ("csv", "imagefolder", "shapes")
SOM_UPDATES = ("gradient", "kohonen_ema")
CROSS_ATTENTION_MODES = ("cls_bridged", "full_pair")

# Tolerance for the train/val/test fractions summing to 1.0.
_SPLIT_SUM_TOL = 1e-3


class ConfigError(ValueError):
    """Raised when a config is structurally invalid.

    The message always names the offending field by its dotted path so the
    problem is locatable from the message alone.
    """


def _require(condition: bool, field_path: str, message: str) -> None:
    """Raise :class:`ConfigError` naming ``field_path`` unless ``condition``."""
    if not condition:
        raise ConfigError(f"{field_path}: {message}")


def _pop_unknown(field_path: str, data: Dict[str, Any], dc_type: type) -> None:
    """Reject keys in ``data`` that are not fields of ``dc_type``."""
    known = {f.name for f in fields(dc_type)}
    unknown = set(data) - known
    _require(
        not unknown,
        field_path,
        f"unknown key(s) {sorted(unknown)}; allowed keys are {sorted(known)}",
    )


@dataclass
class SplitConfig:
    """Train/val/test fractions plus the split seed (ARCHITECTURE §4).

    Fractions are over the whole dataset and must sum to ~1.0. ``seed`` makes
    the (later, U1) grouped split deterministic.
    """

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 1

    def validate(self, path: str = "dataset.splits") -> None:
        for name in ("train", "val", "test"):
            value = getattr(self, name)
            _require(
                isinstance(value, (int, float)) and 0.0 <= value <= 1.0,
                f"{path}.{name}",
                f"must be a fraction in [0, 1], got {value!r}",
            )
        total = self.train + self.val + self.test
        _require(
            abs(total - 1.0) <= _SPLIT_SUM_TOL,
            path,
            f"train+val+test must sum to 1.0 (±{_SPLIT_SUM_TOL}), got {total}",
        )
        _require(
            isinstance(self.seed, int),
            f"{path}.seed",
            f"must be an int, got {self.seed!r}",
        )


@dataclass
class DatasetConfig:
    """Declarative dataset description (ARCHITECTURE §4).

    ``loader`` selects the reader; ``label_column``/``group_column`` are
    optional (absent ``label_column`` ⇒ fully unlabeled mode). No class names,
    resolutions, or directory layouts are hardcoded in model/training code —
    they live here.
    """

    name: str = "shapes"
    loader: str = "shapes"
    image_dir: Optional[str] = None
    metadata_csv: Optional[str] = None
    label_column: Optional[str] = None
    group_column: Optional[str] = None
    image_size: int = 64
    channels: int = 3
    splits: SplitConfig = field(default_factory=SplitConfig)
    augmentation: str = "default"

    def validate(self, path: str = "dataset") -> None:
        _require(
            isinstance(self.name, str) and self.name != "",
            f"{path}.name",
            "must be a non-empty string",
        )
        _require(
            self.loader in LOADERS,
            f"{path}.loader",
            f"unknown loader {self.loader!r}; must be one of {list(LOADERS)}",
        )
        _require(
            isinstance(self.image_size, int) and self.image_size > 0,
            f"{path}.image_size",
            f"must be a positive int, got {self.image_size!r}",
        )
        _require(
            isinstance(self.channels, int) and self.channels > 0,
            f"{path}.channels",
            f"must be a positive int, got {self.channels!r}",
        )
        _require(
            isinstance(self.augmentation, str) and self.augmentation != "",
            f"{path}.augmentation",
            "must be a non-empty policy name",
        )
        # The csv loader needs a metadata file to read paths/labels from.
        if self.loader == "csv":
            _require(
                bool(self.metadata_csv),
                f"{path}.metadata_csv",
                "is required when loader == 'csv'",
            )
        self.splits.validate(f"{path}.splits")


@dataclass
class ModelConfig:
    """Backbone + volume + SOM geometry (ARCHITECTURE §3, §7).

    The latent volume is ``volume_h × volume_w × depth × volume_channels``
    (H'×W'×L×C): the Z-axis length is the number of encoder layers ``depth``.
    ``som_grid`` is the 3-D SOM neuron lattice.
    """

    image_size: int = 128
    fine_patch: int = 8
    coarse_patch: int = 16
    dim: int = 256
    depth: int = 8
    heads: int = 8
    volume_h: int = 16
    volume_w: int = 16
    volume_channels: int = 64
    som_grid: Tuple[int, int, int] = (8, 8, 8)
    som_update: str = "gradient"
    cross_attention: str = "cls_bridged"

    def validate(self, path: str = "model") -> None:
        for name in (
            "image_size",
            "fine_patch",
            "coarse_patch",
            "dim",
            "depth",
            "heads",
            "volume_h",
            "volume_w",
            "volume_channels",
        ):
            value = getattr(self, name)
            _require(
                isinstance(value, int) and value > 0,
                f"{path}.{name}",
                f"must be a positive int, got {value!r}",
            )
        _require(
            self.image_size % self.fine_patch == 0,
            f"{path}.fine_patch",
            f"must divide image_size ({self.image_size}), got {self.fine_patch}",
        )
        _require(
            self.image_size % self.coarse_patch == 0,
            f"{path}.coarse_patch",
            f"must divide image_size ({self.image_size}), got {self.coarse_patch}",
        )
        _require(
            self.dim % self.heads == 0,
            f"{path}.heads",
            f"must divide dim ({self.dim}), got {self.heads}",
        )
        _require(
            isinstance(self.som_grid, tuple) and len(self.som_grid) == 3,
            f"{path}.som_grid",
            f"must be a 3-tuple, got {self.som_grid!r}",
        )
        for axis, value in zip(("x", "y", "z"), self.som_grid):
            _require(
                isinstance(value, int) and value > 0,
                f"{path}.som_grid",
                f"axis {axis} must be a positive int, got {value!r}",
            )
        _require(
            self.som_update in SOM_UPDATES,
            f"{path}.som_update",
            f"unknown value {self.som_update!r}; must be one of {list(SOM_UPDATES)}",
        )
        _require(
            self.cross_attention in CROSS_ATTENTION_MODES,
            f"{path}.cross_attention",
            f"unknown value {self.cross_attention!r}; "
            f"must be one of {list(CROSS_ATTENTION_MODES)}",
        )


@dataclass
class LossConfig:
    """Objective weights and temperatures (ARCHITECTURE §3.6-§3.8).

    Defaults are the DECISION-LOG standing defaults. ``geodesic`` is
    ablation-gated and weighted 0 by default. Each ``lambda_*`` is a
    non-negative weight; setting one to 0 disables that term.
    """

    lambda_ntxent: float = 1.0
    lambda_som: float = 0.5
    lambda_smooth: float = 0.1
    lambda_order: float = 0.1
    lambda_geodesic: float = 0.0
    ntxent_temperature: float = 0.2
    som_temperature: float = 1.0

    def validate(self, path: str = "loss") -> None:
        for name in (
            "lambda_ntxent",
            "lambda_som",
            "lambda_smooth",
            "lambda_order",
            "lambda_geodesic",
        ):
            value = getattr(self, name)
            _require(
                isinstance(value, (int, float)) and value >= 0.0,
                f"{path}.{name}",
                f"must be a non-negative weight, got {value!r}",
            )
        for name in ("ntxent_temperature", "som_temperature"):
            value = getattr(self, name)
            _require(
                isinstance(value, (int, float)) and value > 0.0,
                f"{path}.{name}",
                f"must be a positive temperature, got {value!r}",
            )


@dataclass
class TrainConfig:
    """Optimisation + runtime schedule (ARCHITECTURE §7, DECISION-LOG).

    Exactly one of ``epochs``/``max_steps`` drives run length: set the other
    to ``None``. ``amp`` and ``grad_checkpoint`` are the memory knobs that let
    the same code fit a Kaggle T4.
    """

    batch_size: int = 128
    lr: float = 3e-4
    warmup_steps: int = 500
    epochs: Optional[int] = 100
    max_steps: Optional[int] = None
    amp: bool = True
    grad_checkpoint: bool = True
    seed: int = 0
    checkpoint_dir: str = "checkpoints"

    def validate(self, path: str = "train") -> None:
        _require(
            isinstance(self.batch_size, int) and self.batch_size > 0,
            f"{path}.batch_size",
            f"must be a positive int, got {self.batch_size!r}",
        )
        _require(
            isinstance(self.lr, (int, float)) and self.lr > 0.0,
            f"{path}.lr",
            f"must be a positive learning rate, got {self.lr!r}",
        )
        _require(
            isinstance(self.warmup_steps, int) and self.warmup_steps >= 0,
            f"{path}.warmup_steps",
            f"must be a non-negative int, got {self.warmup_steps!r}",
        )
        for name in ("epochs", "max_steps"):
            value = getattr(self, name)
            _require(
                value is None or (isinstance(value, int) and value > 0),
                f"{path}.{name}",
                f"must be a positive int or null, got {value!r}",
            )
        _require(
            (self.epochs is None) != (self.max_steps is None),
            f"{path}.epochs",
            "exactly one of epochs / max_steps must be set (the other null)",
        )
        for name in ("amp", "grad_checkpoint"):
            value = getattr(self, name)
            _require(
                isinstance(value, bool),
                f"{path}.{name}",
                f"must be a bool, got {value!r}",
            )
        _require(
            isinstance(self.seed, int),
            f"{path}.seed",
            f"must be an int, got {self.seed!r}",
        )
        _require(
            isinstance(self.checkpoint_dir, str) and self.checkpoint_dir != "",
            f"{path}.checkpoint_dir",
            "must be a non-empty path",
        )


@dataclass
class Config:
    """A complete UMT-ViT run configuration (ARCHITECTURE §4-§7)."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> "Config":
        """Validate all sections in place and return ``self`` for chaining."""
        self.dataset.validate()
        self.model.validate()
        self.loss.validate()
        self.train.validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict view suitable for YAML serialisation.

        ``som_grid`` is emitted as a list (YAML has no tuple type); reloading
        via :func:`load_config` restores the tuple, keeping round-trips equal.
        """
        return asdict(self)

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """Serialise to YAML. If ``path`` is given, also write it there."""
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build (and validate) a :class:`Config` from a nested dict."""
        _require(
            isinstance(data, dict),
            "<root>",
            f"config must be a mapping, got {type(data).__name__}",
        )
        _pop_unknown("<root>", data, cls)

        dataset_raw = _as_section("dataset", data.get("dataset", {}), DatasetConfig)
        splits_raw = dataset_raw.pop("splits", {})
        _require(
            isinstance(splits_raw, dict),
            "dataset.splits",
            f"must be a mapping, got {type(splits_raw).__name__}",
        )
        _pop_unknown("dataset.splits", splits_raw, SplitConfig)
        dataset = DatasetConfig(splits=SplitConfig(**splits_raw), **dataset_raw)

        model_raw = _as_section("model", data.get("model", {}), ModelConfig)
        if "som_grid" in model_raw and isinstance(model_raw["som_grid"], list):
            model_raw["som_grid"] = tuple(model_raw["som_grid"])
        model = ModelConfig(**model_raw)

        loss = LossConfig(**_as_section("loss", data.get("loss", {}), LossConfig))
        train = TrainConfig(**_as_section("train", data.get("train", {}), TrainConfig))

        return cls(dataset=dataset, model=model, loss=loss, train=train).validate()


def _as_section(path: str, raw: Any, dc_type: type) -> Dict[str, Any]:
    """Validate that ``raw`` is a mapping with only known keys; return a copy."""
    _require(
        isinstance(raw, dict),
        path,
        f"must be a mapping, got {type(raw).__name__}",
    )
    section = dict(raw)
    if is_dataclass(dc_type):
        _pop_unknown(path, section, dc_type)
    return section


def load_config(path: Union[str, Path]) -> Config:
    """Load, parse, and validate a run config from a YAML file.

    Raises :class:`ConfigError` (naming the offending field) if the file is
    structurally invalid, or :class:`FileNotFoundError` if it is missing.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ConfigError("<root>: config file is empty")
    return Config.from_dict(data)
