"""Config schema tests (ARCHITECTURE §4): round-trip + validation rejections.

Every rejection test asserts the raised :class:`ConfigError` message *names the
offending field* (dotted path), which is the diagnosability contract in
``config.py``. All CPU-only, no downloads; file output goes to ``tmp_path``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from umtvit.config import Config, ConfigError, load_config


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #
def test_default_config_is_valid():
    # Config() defaults must themselves pass validation.
    assert Config().validate() is not None


def test_to_yaml_round_trip_equal(tmp_path: Path):
    original = Config().validate()
    path = tmp_path / "run.yaml"
    original.to_yaml(path)
    reloaded = load_config(path)
    assert reloaded == original


def test_round_trip_restores_som_grid_tuple(tmp_path: Path):
    original = Config().validate()
    path = tmp_path / "run.yaml"
    original.to_yaml(path)
    reloaded = load_config(path)
    # YAML has no tuple type; load_config must restore it so equality holds.
    assert isinstance(reloaded.model.som_grid, tuple)
    assert reloaded.model.som_grid == original.model.som_grid


def test_from_dict_to_dict_round_trip():
    original = Config().validate()
    rebuilt = Config.from_dict(copy.deepcopy(original.to_dict()))
    assert rebuilt == original


# --------------------------------------------------------------------------- #
# Validation rejections — each must name the offending field
# --------------------------------------------------------------------------- #
def _mutated(**section_updates) -> dict:
    """Return a default config dict with the given nested updates applied."""
    data = Config().to_dict()
    for section, updates in section_updates.items():
        data[section].update(updates)
    return data


def test_reject_unknown_loader():
    data = _mutated(dataset={"loader": "parquet"})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "dataset.loader" in str(exc.value)


def test_reject_non_positive_image_size():
    data = _mutated(dataset={"image_size": 0})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "dataset.image_size" in str(exc.value)


def test_reject_splits_not_summing_to_one():
    data = Config().to_dict()
    data["dataset"]["splits"]["train"] = 0.5  # 0.5 + 0.1 + 0.1 = 0.7
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "dataset.splits" in str(exc.value)


def test_reject_fine_patch_not_dividing_image_size():
    data = _mutated(model={"fine_patch": 7})  # 128 % 7 != 0
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "model.fine_patch" in str(exc.value)


def test_reject_coarse_patch_not_dividing_image_size():
    data = _mutated(model={"coarse_patch": 13})  # 128 % 13 != 0
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "model.coarse_patch" in str(exc.value)


def test_reject_unknown_cross_attention():
    data = _mutated(model={"cross_attention": "quadratic"})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "model.cross_attention" in str(exc.value)


def test_reject_unknown_som_update():
    data = _mutated(model={"som_update": "hebbian_magic"})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "model.som_update" in str(exc.value)


def test_reject_som_grid_wrong_length():
    data = _mutated(model={"som_grid": [3, 3]})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "model.som_grid" in str(exc.value)


def test_reject_som_grid_non_positive():
    data = _mutated(model={"som_grid": [3, 0, 3]})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "model.som_grid" in str(exc.value)


def test_reject_dim_not_divisible_by_heads():
    data = _mutated(model={"heads": 7})  # 256 % 7 != 0
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "model.heads" in str(exc.value)


def test_reject_unknown_top_level_key():
    data = Config().to_dict()
    data["bogus_section"] = {}
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_reject_unknown_section_key():
    data = Config().to_dict()
    data["model"]["mystery_knob"] = 3
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "model" in str(exc.value)


def test_load_missing_file_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_empty_file_raises_config_error(tmp_path: Path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(empty)


# --------------------------------------------------------------------------- #
# U1 schema hardening — image_size unification
# --------------------------------------------------------------------------- #
def test_model_image_size_derived_from_dataset():
    # model.image_size omitted (null) ⇒ derived from dataset.image_size.
    data = Config().to_dict()
    data["dataset"]["image_size"] = 96
    data["model"]["image_size"] = None
    config = Config.from_dict(data)
    assert config.model.image_size == 96


def test_model_image_size_may_match_dataset_explicitly():
    data = Config().to_dict()
    data["dataset"]["image_size"] = 96
    data["model"]["image_size"] = 96  # explicit but consistent — allowed
    assert Config.from_dict(data).model.image_size == 96


def test_reject_model_image_size_mismatch():
    data = Config().to_dict()
    data["dataset"]["image_size"] = 96
    data["model"]["image_size"] = 128  # disagrees with the dataset
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "model.image_size" in str(exc.value)


# --------------------------------------------------------------------------- #
# U1 schema hardening — new dataset fields
# --------------------------------------------------------------------------- #
def test_new_dataset_field_defaults():
    dataset = Config().dataset
    assert dataset.path_column == "image_id"
    assert dataset.path_suffix == ".jpg"
    assert dataset.n_per_class is None


def test_reject_empty_path_column():
    data = _mutated(dataset={"path_column": ""})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "dataset.path_column" in str(exc.value)


def test_reject_non_positive_n_per_class():
    data = _mutated(dataset={"n_per_class": 0})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "dataset.n_per_class" in str(exc.value)


def test_image_dir_accepts_list():
    data = _mutated(
        dataset={"loader": "csv", "metadata_csv": "m.csv", "image_dir": ["a", "b"]}
    )
    config = Config.from_dict(data)
    assert config.dataset.image_dir == ["a", "b"]


def test_reject_image_dir_bad_list():
    data = _mutated(dataset={"loader": "csv", "metadata_csv": "m.csv", "image_dir": ["a", 3]})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "dataset.image_dir" in str(exc.value)


def test_reject_imagefolder_without_image_dir():
    data = _mutated(dataset={"loader": "imagefolder", "image_dir": None})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "dataset.image_dir" in str(exc.value)


# --------------------------------------------------------------------------- #
# U1 schema hardening — loss schedule fields
# --------------------------------------------------------------------------- #
def test_new_loss_schedule_defaults():
    loss = Config().loss
    assert loss.sigma_start == 2.0
    assert loss.sigma_end == 0.5
    assert loss.order_fmax == 0.5


def test_reject_non_positive_order_fmax():
    data = _mutated(loss={"order_fmax": 0.0})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "loss.order_fmax" in str(exc.value)


def test_reject_sigma_end_above_sigma_start():
    data = _mutated(loss={"sigma_start": 0.5, "sigma_end": 2.0})
    with pytest.raises(ConfigError) as exc:
        Config.from_dict(data)
    assert "loss.sigma_end" in str(exc.value)
