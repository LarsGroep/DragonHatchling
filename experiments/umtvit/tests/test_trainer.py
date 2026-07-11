"""U4 trainer tests (ARCHITECTURE §7, §9 row U4).

CPU-only, AMP off, tiny configs; the whole file stays well under the suite
budget. Covers the acceptance tests for the milestone:

- a 200-step shapes smoke run whose total loss falls (first-20 vs last-20 mean)
  with no NaNs;
- the geodesic gate: weight 0 ⇒ never computed (monkeypatched call counter) and
  a constant-zero history term; weight > 0 ⇒ computed;
- resume bit-consistency: train N epochs straight vs train N/2 + save + load +
  train N/2 gives identical final loss and parameters;
- checkpoint round-trip restores optimiser state;
- the ``on_epoch_end`` callback fires once per epoch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from umtvit.config import (
    Config,
    DatasetConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
)
from umtvit.data.dataset import UniversalDataset
from umtvit.engine import trainer as trainer_module
from umtvit.engine import Trainer
from umtvit.models import Soft3DSOM, UMTViT


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
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


def _train_cfg(**over) -> TrainConfig:
    base = dict(
        batch_size=4,
        lr=1e-3,
        warmup_steps=3,
        epochs=4,
        max_steps=None,
        amp=False,
        grad_checkpoint=False,
        som_sample_voxels=64,
        seed=0,
    )
    base.update(over)
    return TrainConfig(**base)


def _config(model=None, loss=None, train=None, dataset=None) -> Config:
    return Config(
        dataset=dataset or DatasetConfig(image_size=32, channels=3),
        model=model or _model_cfg(),
        loss=loss or LossConfig(),
        train=train or _train_cfg(),
    ).validate()


def _build(config: Config, seed: int = 42):
    torch.manual_seed(seed)
    model = UMTViT(config)
    som = Soft3DSOM.from_config(config)
    return model, som


class _DeterministicTwoView:
    """Fixed, deterministic (view_a, view_b, label) pairs for exact-resume tests."""

    def __init__(self, n: int = 8, size: int = 32):
        g = torch.Generator().manual_seed(1)
        self.x = torch.rand(n, 3, size, size, generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.x[i] + 0.01, 0


def _det_loader(n: int = 8, batch: int = 4) -> DataLoader:
    return DataLoader(_DeterministicTwoView(n), batch_size=batch, shuffle=False)


# --------------------------------------------------------------------------- #
# 200-step shapes smoke run
# --------------------------------------------------------------------------- #
def test_shapes_smoke_loss_decreases_no_nan():
    config = _config(
        dataset=DatasetConfig(
            loader="shapes",
            n_per_class=8,
            image_size=32,
            channels=3,
            augmentation="natural_default",
        ),
        train=_train_cfg(
            epochs=None, max_steps=200, batch_size=16, lr=2e-3, warmup_steps=10
        ),
    )
    dataset = UniversalDataset(config, split="train", mode="two_view")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model, som = _build(config)
    t = Trainer(config, model, som, loader)
    history = t.train()

    assert len(history["total"]) == 200
    assert not any(math.isnan(v) for v in history["total"]), "NaN in total loss"
    for key in ("ntxent", "som", "smooth", "order", "order_monotone"):
        assert not any(math.isnan(v) for v in history[key]), f"NaN in {key}"
    first = float(np.mean(history["total"][:20]))
    last = float(np.mean(history["total"][-20:]))
    assert last < first, f"loss did not fall (first20={first:.3f}, last20={last:.3f})"


# --------------------------------------------------------------------------- #
# Geodesic gate
# --------------------------------------------------------------------------- #
def test_geodesic_off_never_computed(monkeypatch):
    calls = {"n": 0}

    def _counting_geodesic(v_sub, za, zb, k=6):
        calls["n"] += 1
        return v_sub.new_zeros(())

    monkeypatch.setattr(trainer_module, "geodesic_loss", _counting_geodesic)

    config = _config(
        loss=LossConfig(lambda_geodesic=0.0),
        train=_train_cfg(epochs=None, max_steps=6, batch_size=4),
    )
    model, som = _build(config)
    history = Trainer(config, model, som, _det_loader()).train()

    assert calls["n"] == 0, "geodesic computed despite weight 0"
    assert all(v == 0.0 for v in history["geodesic"]), "geodesic history not zero"


def test_geodesic_on_is_computed(monkeypatch):
    calls = {"n": 0}

    def _counting_geodesic(v_sub, za, zb, k=6):
        calls["n"] += 1
        return v_sub.new_zeros(())

    monkeypatch.setattr(trainer_module, "geodesic_loss", _counting_geodesic)

    config = _config(
        loss=LossConfig(lambda_geodesic=0.1),
        train=_train_cfg(epochs=None, max_steps=6, batch_size=4),
    )
    model, som = _build(config)
    Trainer(config, model, som, _det_loader()).train()

    assert calls["n"] == 6, f"geodesic not computed every step (got {calls['n']})"


# --------------------------------------------------------------------------- #
# Resume bit-consistency
# --------------------------------------------------------------------------- #
def test_resume_is_bit_consistent(tmp_path):
    ckpt = str(tmp_path / "mid.pt")

    # straight: 4 epochs
    cfg_a = _config(train=_train_cfg(epochs=4))
    model_a, som_a = _build(cfg_a, seed=7)
    hist_a = Trainer(cfg_a, model_a, som_a, _det_loader()).train()

    # split: 2 epochs, save, resume, 2 more
    cfg_b = _config(train=_train_cfg(epochs=4))
    model_b, som_b = _build(cfg_b, seed=7)
    t_b = Trainer(cfg_b, model_b, som_b, _det_loader())
    t_b.train(until_epoch=2)
    t_b.save_checkpoint(ckpt)

    cfg_c = _config(train=_train_cfg(epochs=4))
    model_c, som_c = _build(cfg_c, seed=999)  # different init on purpose
    hist_c = Trainer(cfg_c, model_c, som_c, _det_loader()).train(resume_from=ckpt)

    assert len(hist_a["total"]) == len(hist_c["total"])
    assert hist_a["total"][-1] == hist_c["total"][-1], "final loss diverged"
    max_param_diff = max(
        float((pa.detach() - pc.detach()).abs().max())
        for pa, pc in zip(model_a.parameters(), model_c.parameters())
    )
    assert max_param_diff == 0.0, f"parameters diverged (max {max_param_diff})"
    assert float((som_a.weights.detach() - som_c.weights.detach()).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# Checkpoint round-trip restores optimiser state
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip_restores_optimizer_state(tmp_path):
    ckpt = str(tmp_path / "opt.pt")
    config = _config(train=_train_cfg(epochs=2))
    model, som = _build(config)
    t = Trainer(config, model, som, _det_loader())
    t.train()
    t.save_checkpoint(ckpt)

    # A fresh trainer with a *fresh* (empty) optimiser state, then load.
    model2, som2 = _build(config, seed=123)
    t2 = Trainer(config, model2, som2, _det_loader())
    assert t2.opt.state_dict()["state"] == {}  # no momentum buffers yet
    t2.load_checkpoint(ckpt)

    src = t.opt.state_dict()["state"]
    dst = t2.opt.state_dict()["state"]
    assert dst.keys() == src.keys() and len(dst) > 0
    for pid in src:
        assert torch.equal(src[pid]["exp_avg"], dst[pid]["exp_avg"])
        assert torch.equal(src[pid]["exp_avg_sq"], dst[pid]["exp_avg_sq"])
    assert t2.epoch == t.epoch and t2.step == t.step


# --------------------------------------------------------------------------- #
# Callback fires each epoch
# --------------------------------------------------------------------------- #
def test_callback_fires_each_epoch():
    fired = []
    config = _config(train=_train_cfg(epochs=3))
    model, som = _build(config)

    def _cb(epoch, metrics, trainer):
        fired.append(epoch)
        assert metrics["epoch"] == epoch
        assert "quantization_error" in metrics
        assert trainer is t

    t = Trainer(config, model, som, _det_loader(), on_epoch_end=_cb)
    t.train()
    assert fired == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Gradient checkpointing path is exercised
# --------------------------------------------------------------------------- #
def test_grad_checkpoint_flag_toggles_encoder():
    config = _config(train=_train_cfg(epochs=None, max_steps=4, grad_checkpoint=True))
    model, som = _build(config)
    t = Trainer(config, model, som, _det_loader())
    assert model.backbone.encoder.use_checkpoint is True
    history = t.train()
    assert not any(math.isnan(v) for v in history["total"])
