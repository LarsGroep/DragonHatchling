"""The two things a run must persist for post-hoc interpretability.

A finished run used to keep a scalar ``val_acc`` and a bare probe tensor. Both
of those are dead ends:

- **No per-image probabilities** ⇒ no sensitivity, specificity, PPV, AUC or
  confidence interval can ever be computed, at any threshold, without
  retraining. For a cancer-diagnosis tool the threshold readout *is* the
  product, so this was the binding constraint.
- **No image ids on the probe batch** ⇒ concept activations cannot be joined to
  external annotations such as the ISIC 2018 Task 2 dermoscopic masks, which
  forces grounding back onto whatever the loader carries in-band. For HAM10000
  that was patient metadata, which must never name a visual concept.

``Trainer.predict`` and ``DatasetLoader.probe_image_ids`` close both holes.
These tests pin them, and pin that adding them changed nothing else.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hatchvision import HebbianFeatureMemory, TrainConfig, Trainer, create_model
from hatchvision.data import DatasetSpec
from hatchvision.data.base import DatasetLoader

SPEC = DatasetSpec(
    name="synthetic",
    num_classes=4,
    class_names=("a", "b", "c", "d"),
    image_size=32,
    in_channels=3,
)


def _loader(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, SPEC.num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _trainer(memory=None):
    model = create_model("simple_cnn", SPEC)
    return Trainer(model, TrainConfig(epochs=1, log_every=0), hebbian_memory=memory)


# --------------------------------------------------------------------------- #
# Trainer.predict
# --------------------------------------------------------------------------- #


def test_predict_returns_full_probability_matrix():
    loader = _loader()
    y_true, y_prob = _trainer().predict(loader)

    assert isinstance(y_true, np.ndarray) and isinstance(y_prob, np.ndarray)
    assert y_true.shape == (24,)
    assert y_prob.shape == (24, SPEC.num_classes)
    # Genuine softmax: rows sum to 1, all entries in [0, 1].
    assert np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-5)
    assert (y_prob >= 0).all() and (y_prob <= 1).all()
    assert np.isfinite(y_prob).all()


def test_predict_preserves_loader_order():
    """Rows must align with labels, or every downstream metric is garbage."""
    loader = _loader()
    y_true, _ = _trainer().predict(loader)
    expected = np.concatenate([y.numpy() for _, y in loader])
    assert np.array_equal(y_true, expected)


def test_predict_agrees_with_evaluate():
    """argmax of the retained matrix must reproduce evaluate()'s accuracy."""
    loader = _loader()
    trainer = _trainer()
    _, acc = trainer.evaluate(loader)
    y_true, y_prob = trainer.predict(loader)
    assert (y_prob.argmax(1) == y_true).mean() == pytest.approx(acc, abs=1e-6)


def test_predict_does_not_contaminate_hebbian_memory():
    """Scoring is not training: co-activation statistics must not move."""
    model = create_model("simple_cnn", SPEC)
    memory = HebbianFeatureMemory(model, num_classes=SPEC.num_classes, max_units=32)
    trainer = Trainer(model, TrainConfig(epochs=1, log_every=0), hebbian_memory=memory)
    loader = _loader()
    trainer.train_epoch(loader)  # give the memory something to hold

    before = {k: v.coact.clone() for k, v in memory.stats.items()}
    trainer.predict(loader)
    for layer, stats in memory.stats.items():
        assert torch.equal(stats.coact, before[layer]), layer


def test_predict_rejects_an_empty_loader():
    empty = DataLoader(TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="no batches"):
        _trainer().predict(empty)


def test_predict_output_feeds_clinical_report():
    """The whole point: the retained matrix is what the metrics module needs."""
    clinical = pytest.importorskip("vitreous.clinical")
    y_true, y_prob = _trainer().predict(_loader())
    report = clinical.clinical_report(
        y_true, y_prob, class_names=list(SPEC.class_names), threshold=0.2
    )
    summary = clinical.format_honest_summary(report)
    assert isinstance(summary, str) and summary
    # Accuracy may never be quoted without its floor.
    assert "majority" in summary.lower()


# --------------------------------------------------------------------------- #
# DatasetLoader.probe_image_ids
# --------------------------------------------------------------------------- #


class _PathSplit(torch.utils.data.Dataset):
    """Mimics the (path, label) split objects the real loaders build."""

    def __init__(self, paths):
        self.samples = [(p, 0) for p in paths]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return torch.zeros(3, 32, 32), self.samples[i][1]


class _PathLoader(DatasetLoader):
    spec = SPEC

    def __init__(self, paths):
        self._paths = paths

    def train_dataset(self):
        return _PathSplit(self._paths)

    def val_dataset(self):
        return _PathSplit(self._paths)


class _IdlessSplit(torch.utils.data.Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, i):
        return torch.zeros(3, 32, 32), 0


class _IdlessLoader(DatasetLoader):
    spec = SPEC

    def train_dataset(self):
        return _IdlessSplit()

    def val_dataset(self):
        return _IdlessSplit()


def test_probe_image_ids_recovers_isic_ids_and_aligns_with_probe_batch():
    paths = [f"/data/HAM10000_images_part_1/ISIC_002430{i}.jpg" for i in range(6)]
    loader = _PathLoader(paths)

    ids = loader.probe_image_ids(4)
    assert ids == ["ISIC_0024300", "ISIC_0024301", "ISIC_0024302", "ISIC_0024303"]
    # 1:1 with the probe batch — the alignment the join depends on.
    assert len(ids) == loader.probe_batch(4).shape[0]


def test_probe_image_ids_is_capped_by_dataset_size():
    loader = _PathLoader([f"/d/ISIC_{i}.jpg" for i in range(2)])
    assert loader.probe_image_ids(99) == ["ISIC_0", "ISIC_1"]


def test_probe_image_ids_returns_none_rather_than_inventing_ids():
    """"No ids" must stay distinguishable from "ids that are just indices"."""
    assert _IdlessLoader().probe_image_ids(3) is None
