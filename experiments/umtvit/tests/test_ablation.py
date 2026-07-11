"""U5 ablation-runner tests (ARCHITECTURE §6.5, §9 row U5).

CPU-only, tiny model, a handful of steps per variant. Covers:

- the ``cross_rounds == 0`` (``no_cross_attention``) structural baseline: it
  forwards cleanly with an empty cross-attention stack and its representation
  *differs* from ``cross_rounds == 1`` under an otherwise identical init;
- ``AblationRunner`` over two tiny variants produces a comparison table naming
  both variants with every metric column present and finite;
- the canonical :data:`ABLATIONS` axis set is well-formed (all six named axes,
  each an override valid against the schema).
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from umtvit.config import Config, DatasetConfig, LossConfig, ModelConfig, TrainConfig
from umtvit.data.dataset import UniversalDataset
from umtvit.engine import ABLATIONS, AblationRunner
from umtvit.engine.ablation import _apply_override
from umtvit.models import Soft3DSOM, UMTViT


def _model_cfg(**over) -> ModelConfig:
    base = dict(
        fine_patch=8,
        coarse_patch=16,
        dim=32,
        depth=2,
        heads=4,
        volume_h=4,
        volume_w=4,
        volume_channels=8,
        proj_dim=16,
        som_grid=(3, 3, 3),
    )
    base.update(over)
    return ModelConfig(**base)


def _base_config() -> Config:
    return Config(
        dataset=DatasetConfig(
            loader="shapes",
            n_per_class=12,
            image_size=32,
            channels=3,
            augmentation="natural_default",
            label_column="shape",
        ),
        model=_model_cfg(),
        loss=LossConfig(),
        train=TrainConfig(
            epochs=None,
            max_steps=8,
            batch_size=12,
            lr=2e-3,
            warmup_steps=2,
            amp=False,
            grad_checkpoint=False,
            som_sample_voxels=64,
            seed=0,
        ),
    ).validate()


# --------------------------------------------------------------------------- #
# cross_rounds == 0 forwards and differs from cross_rounds == 1
# --------------------------------------------------------------------------- #
def test_cross_rounds_zero_forwards_and_differs():
    x = torch.rand(2, 3, 32, 32)

    def _pooled(cross_rounds: int) -> torch.Tensor:
        cfg = Config(
            dataset=DatasetConfig(image_size=32, channels=3),
            model=_model_cfg(cross_rounds=cross_rounds),
            loss=LossConfig(),
            train=TrainConfig(
                epochs=1, max_steps=None, batch_size=2, amp=False,
                grad_checkpoint=False, som_sample_voxels=16,
            ),
        ).validate()
        torch.manual_seed(5)
        model = UMTViT(cfg)
        model.eval()
        with torch.no_grad():
            return model(x)["pooled"], model

    p0, m0 = _pooled(0)
    p1, _m1 = _pooled(1)

    assert len(m0.backbone.cross) == 0  # cross stack skipped entirely
    assert p0.shape == p1.shape
    assert torch.isfinite(p0).all()
    assert not torch.allclose(p0, p1), "cross_rounds=0 must change the representation"


# --------------------------------------------------------------------------- #
# AblationRunner over two tiny variants -> comparison table
# --------------------------------------------------------------------------- #
def test_ablation_runner_two_variants_table():
    cfg = _base_config()
    two_view = UniversalDataset(cfg, split="train", mode="two_view")
    train_eval = UniversalDataset(cfg, split="train", mode="eval")

    def train_data(_c):
        return DataLoader(two_view, batch_size=12, shuffle=True)

    variants = [
        ("no_cross_attention", {"model.cross_rounds": 0}),
        ("no_som", {"loss.lambda_som": 0.0}),
    ]
    runner = AblationRunner(
        cfg,
        variants,
        train_data=train_data,
        eval_datasets={"train": train_eval, "test": train_eval},
        seed=0,
        eval_kwargs=dict(probe_steps=100, manifold_max_n=120, som_sample_voxels=128, zaxis_images=4),
    )
    out = runner.run()

    assert set(out["results"]) == {"no_cross_attention", "no_som"}
    assert [r["variant"] for r in out["rows"]] == ["no_cross_attention", "no_som"]

    metric_cols = [c for c in out["columns"] if c != "variant"]
    for col in ("probe_acc", "knn_acc", "quant_err", "topo_err", "dead_frac",
                "trust", "continuity", "z_monotone"):
        assert col in metric_cols

    # Markdown names both variants and every column header.
    for name in ("no_cross_attention", "no_som"):
        assert name in out["table"]
    for col in metric_cols:
        assert col in out["table"]

    # Every numeric metric that ran is finite (booleans/None allowed too).
    for row in out["rows"]:
        for col in ("quant_err", "topo_err", "dead_frac", "trust", "continuity"):
            assert row[col] is not None and math.isfinite(float(row[col])), (row["variant"], col)
        assert row["probe_acc"] is not None  # labeled shapes ⇒ probe ran


# --------------------------------------------------------------------------- #
# Canonical ABLATIONS constant
# --------------------------------------------------------------------------- #
def test_canonical_ablations_are_wellformed():
    names = [name for name, _ in ABLATIONS]
    assert names == [
        "no_cross_attention",
        "full_pair",
        "no_som",
        "no_order",
        "smooth_hwz",
        "kohonen_ema",
    ]
    # Each override applies cleanly and yields a schema-valid config.
    for _name, override in ABLATIONS:
        cfg = _base_config()
        cfg.model.image_size = None
        for key, value in override.items():
            _apply_override(cfg, key, value)
        cfg.validate()  # must not raise
