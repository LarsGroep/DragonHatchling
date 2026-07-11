"""U5 evaluation-suite tests (ARCHITECTURE §6, §9 row U5).

CPU-only, tiny model, ~50 training steps: the whole file trains one shared
model once (module-scoped fixture) and reads every §6 metric off it. Covers the
milestone's acceptance surface:

- end-to-end ``run_evaluation``: linear probe ≫ 1.5×chance, k-NN > chance, every
  SOM/manifold key present and finite, trustworthiness *and* continuity in
  ``[0, 1]``, Z-axis returns depth-length centroid lists + a verdict string;
- unlabeled mode returns ``None`` probes without crashing (label-free metrics
  still run) and renders a report;
- ``render_report`` emits the results table, the verdict, and the SSL caveat.

Shapes are highly separable, so the probe/k-NN read the frozen features on the
(deterministic, seeded) train split — a stable end-to-end signal check rather
than a held-out generalisation benchmark.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from umtvit.config import Config, DatasetConfig, LossConfig, ModelConfig, TrainConfig
from umtvit.data.dataset import UniversalDataset
from umtvit.engine import Trainer
from umtvit.eval import (
    extract_features,
    knn_accuracy,
    linear_probe,
    render_report,
    run_evaluation,
)
from umtvit.eval.features import FrozenFeatures
from umtvit.models import Soft3DSOM, UMTViT

_DEPTH = 2
_N_PER_CLASS = 20
_EVAL_KW = dict(probe_steps=200, manifold_max_n=200, som_sample_voxels=256, zaxis_images=6)


def _model_cfg(**over) -> ModelConfig:
    base = dict(
        fine_patch=8,
        coarse_patch=16,
        dim=32,
        depth=_DEPTH,
        heads=4,
        volume_h=4,
        volume_w=4,
        volume_channels=8,
        proj_dim=16,
        som_grid=(3, 3, 3),
    )
    base.update(over)
    return ModelConfig(**base)


def _labeled_config() -> Config:
    return Config(
        dataset=DatasetConfig(
            loader="shapes",
            n_per_class=_N_PER_CLASS,
            image_size=32,
            channels=3,
            augmentation="natural_default",
            label_column="shape",
        ),
        model=_model_cfg(),
        loss=LossConfig(),
        train=TrainConfig(
            epochs=None,
            max_steps=50,
            batch_size=12,
            lr=2e-3,
            warmup_steps=6,
            amp=False,
            grad_checkpoint=False,
            som_sample_voxels=64,
            seed=0,
        ),
    ).validate()


class _UnlabeledImages:
    """Tiny in-memory eval dataset yielding ``(image, -1)`` (no label_column)."""

    def __init__(self, n: int = 24, size: int = 32):
        g = torch.Generator().manual_seed(3)
        self.x = torch.rand(n, 3, size, size, generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], -1


@pytest.fixture(scope="module")
def trained():
    """Train the shared tiny model once; return (model, som, cfg, datasets)."""
    cfg = _labeled_config()
    two_view = UniversalDataset(cfg, split="train", mode="two_view")
    torch.manual_seed(0)
    model = UMTViT(cfg)
    som = Soft3DSOM.from_config(cfg)
    Trainer(cfg, model, som, DataLoader(two_view, batch_size=12, shuffle=True)).train()
    model.eval()
    train_eval = UniversalDataset(cfg, split="train", mode="eval")
    datasets = {"train": train_eval, "test": train_eval}
    return model, som, cfg, datasets


@pytest.fixture(scope="module")
def evaluation(trained):
    model, som, cfg, datasets = trained
    return run_evaluation(model, som, cfg, datasets, seed=0, **_EVAL_KW)


# --------------------------------------------------------------------------- #
# Frozen features
# --------------------------------------------------------------------------- #
def test_features_shape_and_labels(trained):
    model, _som, cfg, datasets = trained
    feats = extract_features(model, datasets["train"])
    assert feats.labeled
    n = len(datasets["train"])
    assert tuple(feats.pooled.shape) == (n, cfg.model.depth * cfg.model.volume_channels)
    assert feats.pixels.shape == (n, cfg.dataset.channels * cfg.dataset.image_size ** 2)
    assert feats.num_classes == 3


# --------------------------------------------------------------------------- #
# End-to-end run_evaluation: signal + finiteness
# --------------------------------------------------------------------------- #
def test_probe_beats_chance(evaluation):
    probe = evaluation["linear_probe"]
    assert probe is not None
    assert probe["accuracy"] > 1.5 * probe["chance"], probe
    assert set(probe["per_class_accuracy"]) == {0, 1, 2}


def test_knn_beats_chance(evaluation):
    knn = evaluation["knn"]
    assert knn is not None
    assert knn["accuracy"] > knn["chance"], knn
    assert knn["k"] == 5


def test_som_metrics_present_and_finite(evaluation):
    som = evaluation["som"]
    for key in ("quantization_error", "topographic_error", "dead_neuron_fraction"):
        assert key in som and math.isfinite(som[key]), (key, som)
    assert 0.0 <= som["topographic_error"] <= 1.0
    assert 0.0 <= som["dead_neuron_fraction"] <= 1.0


def test_manifold_trust_and_continuity_in_unit_interval(evaluation):
    manifold = evaluation["manifold"]
    for key in ("trustworthiness", "continuity"):
        value = manifold[key]
        assert math.isfinite(value), (key, manifold)
        assert 0.0 <= value <= 1.0, (key, value)
    assert manifold["k"] == 7


def test_zaxis_returns_depth_length_lists_and_verdict(evaluation, trained):
    _model, _som, cfg, _datasets = trained
    zaxis = evaluation["zaxis"]
    assert len(zaxis["per_channel_centroids"]) == cfg.model.depth
    assert len(zaxis["channel_mean_centroids"]) == cfg.model.depth
    assert all(math.isfinite(c) for c in zaxis["per_channel_centroids"])
    assert isinstance(zaxis["monotone_decreasing"], bool)
    assert isinstance(zaxis["verdict"], str) and zaxis["verdict"]


def test_all_top_level_keys_present(evaluation):
    for key in ("linear_probe", "knn", "som", "manifold", "zaxis", "meta"):
        assert key in evaluation


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def test_render_report_has_sections_and_caveat(evaluation):
    md = render_report(evaluation)
    assert "# UMT-ViT Evaluation Report" in md
    assert "Linear probe accuracy" in md
    assert "## 2. SOM quality" in md
    assert evaluation["zaxis"]["verdict"] in md
    assert "DSCATNet" in md  # the honest SSL-yardstick caveat


# --------------------------------------------------------------------------- #
# Unlabeled mode: None probes, no crash
# --------------------------------------------------------------------------- #
def test_unlabeled_mode_returns_none_probes(trained):
    model, som, cfg, _datasets = trained
    unlabeled = _UnlabeledImages()
    datasets = {"train": unlabeled, "test": unlabeled}
    res = run_evaluation(model, som, cfg, datasets, seed=0, **_EVAL_KW)

    assert res["linear_probe"] is None
    assert res["knn"] is None
    # Label-free metrics still run and stay finite.
    assert math.isfinite(res["som"]["quantization_error"])
    assert math.isfinite(res["manifold"]["trustworthiness"])
    assert len(res["zaxis"]["per_channel_centroids"]) == cfg.model.depth
    md = render_report(res)
    assert "Fully-unlabeled" in md


def test_probes_none_on_unlabeled_features_directly():
    feats = FrozenFeatures(
        pooled=torch.randn(10, 8), labels=None, pixels=torch.randn(10, 4), num_classes=0
    )
    assert linear_probe(feats, feats) is None
    assert knn_accuracy(feats, feats) is None
