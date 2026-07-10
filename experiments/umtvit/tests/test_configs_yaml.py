"""The shipped ``configs/*.yaml`` files must load and validate.

Guards against a config file drifting out of sync with the schema. All three
v1 datasets (shapes, HAM10000, EuroSAT) load and validate here **without the
data being present** — validation is filesystem-free, which is what lets the
same configs be checked in CI and on Kaggle alike.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from umtvit.config import Config, load_config

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

_ALL_CONFIGS = ("shapes", "ham10000", "eurosat")


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_config_loads_and_validates(name: str):
    config = load_config(_CONFIGS_DIR / f"{name}.yaml")
    assert isinstance(config, Config)
    # image_size is unified: the model derives it from the dataset, and the
    # patch sizes must divide it.
    assert config.model.image_size == config.dataset.image_size
    assert config.model.image_size % config.model.fine_patch == 0
    assert config.model.image_size % config.model.coarse_patch == 0
    assert len(config.model.som_grid) == 3


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_config_round_trips(name: str, tmp_path: Path):
    original = load_config(_CONFIGS_DIR / f"{name}.yaml")
    path = tmp_path / f"{name}_out.yaml"
    original.to_yaml(path)
    assert load_config(path) == original


def test_shapes_is_ci_smoke():
    config = load_config(_CONFIGS_DIR / "shapes.yaml")
    assert config.dataset.loader == "shapes"
    assert config.dataset.n_per_class is not None


def test_gpu_scale_presets_match_standing_defaults():
    # DECISION-LOG standing defaults for the GPU-scale runs.
    for name in ("ham10000", "eurosat"):
        config = load_config(_CONFIGS_DIR / f"{name}.yaml")
        assert config.model.dim == 256
        assert config.model.depth == 8
        assert config.model.som_grid == (8, 8, 8)


def test_ham10000_uses_grouped_dermoscopy_csv():
    config = load_config(_CONFIGS_DIR / "ham10000.yaml")
    assert config.dataset.loader == "csv"
    assert isinstance(config.dataset.image_dir, list)  # two Kaggle image parts
    assert config.dataset.group_column == "lesion_id"  # leakage-free splits
    assert config.dataset.augmentation == "dermoscopy_default"
