"""The shipped ``configs/*.yaml`` files must load and validate.

Guards against a config file drifting out of sync with the schema.
"""

from __future__ import annotations

from pathlib import Path

from umtvit.config import Config, load_config

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_shapes_yaml_loads_and_validates():
    config = load_config(_CONFIGS_DIR / "shapes.yaml")
    assert isinstance(config, Config)
    # CI-scale sanity: small, patch sizes divide the image, tiny latent volume.
    assert config.dataset.loader == "shapes"
    assert config.model.image_size % config.model.fine_patch == 0
    assert config.model.image_size % config.model.coarse_patch == 0
    assert len(config.model.som_grid) == 3


def test_shapes_yaml_round_trips(tmp_path: Path):
    original = load_config(_CONFIGS_DIR / "shapes.yaml")
    path = tmp_path / "shapes_out.yaml"
    original.to_yaml(path)
    assert load_config(path) == original
