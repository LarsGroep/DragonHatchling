"""Explain pack: unit→class influence, class fingerprints, JSON export."""

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hatchvision import (
    HebbianFeatureMemory,
    TrainConfig,
    Trainer,
    create_model,
)
from hatchvision.data import DatasetSpec
from hatchvision.explain import class_fingerprints, unit_class_influence
from hatchvision.export import build_explain_pack, export_explain_pack

SPEC = DatasetSpec(
    name="synthetic",
    num_classes=4,
    class_names=("a", "b", "c", "d"),
    image_size=32,
    in_channels=3,
)


def _trained(backbone, **kwargs):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(24, 3, 32, 32, generator=g)
    y = torch.randint(0, SPEC.num_classes, (24,), generator=g)
    model = create_model(backbone, SPEC, **kwargs)
    memory = HebbianFeatureMemory(model, num_classes=SPEC.num_classes, max_units=48)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    Trainer(model, TrainConfig(epochs=1, log_every=0), memory).fit(loader)
    return model, memory, x


def _tracked_acts(model, memory, layer, x):
    acts = {}

    def cap(_m, _i, out):
        acts["a"] = torch.relu(HebbianFeatureMemory._pool(out.detach().float()))

    model.eval()
    handle = model.hebbian_layers()[layer].register_forward_hook(cap)
    with torch.no_grad():
        model(x)
    handle.remove()
    a = acts["a"]
    idx = memory.stats[layer].unit_index
    return a if idx is None else a[:, idx]


def test_influence_shapes_and_exactness_hybrid():
    """The hybrid readout is linear in the neurons → exact Shapley values."""
    model, memory, x = _trained(
        "hybrid", encoder="resnet18", pretrained=False, neuron_dim=256
    )
    layer = memory.layer_names[-1]
    inf = unit_class_influence(model, memory, layer, x[:16], batch_size=8)
    n_units = memory.stats[layer].dim
    assert inf.weights.shape == (SPEC.num_classes, n_units)
    assert inf.baseline.shape == (n_units,)
    assert inf.expected_logits.shape == (SPEC.num_classes,)
    assert inf.method == "exact-linear"
    assert torch.isfinite(inf.weights).all()

    # exactness: the tracked units' phi must match the true logit change when
    # those units are the only thing that varies. Verify against the model's
    # effective linear map W_head @ readback restricted to tracked columns.
    model.eval()
    W = model.head[1].weight.detach()                     # [K, feat]
    R = model.backbone.readback.weight.detach()           # [feat, neurons]
    eff = W @ R                                           # [K, neurons]
    idx = memory.stats[layer].unit_index
    if idx is not None:
        eff = eff[:, idx]
    assert torch.allclose(inf.weights, eff, atol=1e-4)


def test_influence_first_order_consistency_cnn():
    """phi = weights·(act−baseline) reproduces logit deviations on a CNN."""
    model, memory, x = _trained("simple_cnn")
    layer = memory.layer_names[-1]
    inf = unit_class_influence(model, memory, layer, x[:16], batch_size=8)
    model.eval()
    with torch.no_grad():
        logits = model(x[16:24])
    a = _tracked_acts(model, memory, layer, x[16:24])
    approx = inf.expected_logits + (a - inf.baseline) @ inf.weights.t()
    err = (approx - logits).abs().max()
    spread = logits.max() - logits.min()
    assert err < max(0.05 * spread, 1e-3), f"first-order error {err} vs spread {spread}"


def test_class_fingerprints_normalized():
    _, memory, _ = _trained("simple_cnn")
    layer = memory.layer_names[-1]
    fp = class_fingerprints(memory, layer)
    assert fp.shape == (SPEC.num_classes, memory.stats[layer].dim)
    assert (fp >= 0).all() and fp.max() <= 1.0 + 1e-6
    # every class that appeared has a peak unit at exactly 1.0
    seen = memory.stats[layer].class_count > 0
    assert torch.allclose(
        fp[seen].max(dim=1).values, torch.ones(int(seen.sum()))
    )


def test_explain_pack_roundtrip(tmp_path):
    model, memory, x = _trained("simple_cnn")
    layer = memory.layer_names[-1]
    path = export_explain_pack(
        memory, layer, SPEC.class_names, tmp_path / "explain.json",
        model=model, background=x[:16],
    )
    doc = json.loads(path.read_text())
    assert doc["format"] == "hatchvision-explain"
    assert doc["layer"] == layer
    assert doc["node_prefix"] == f"u:{layer}:"
    assert doc["num_classes"] == SPEC.num_classes
    n_units = memory.stats[layer].dim
    assert len(doc["fingerprints"]["matrix"]) == SPEC.num_classes
    assert len(doc["fingerprints"]["matrix"][0]) == n_units
    shap = doc["shap"]
    assert shap["method"] in ("exact-linear", "expected-gradients")
    assert len(shap["weights"]) == SPEC.num_classes
    assert len(shap["weights"][0]) == n_units
    assert len(shap["baseline"]) == n_units
    assert len(shap["expected_logits"]) == SPEC.num_classes


def test_explain_pack_without_model():
    """Fingerprints-only pack (what rebuild_graph.py produces post hoc)."""
    _, memory, _ = _trained("simple_cnn")
    layer = memory.layer_names[-1]
    doc = build_explain_pack(memory, layer, SPEC.class_names)
    assert "shap" not in doc
    assert len(doc["fingerprints"]["matrix"]) == SPEC.num_classes


def test_influence_leaves_memory_and_mode_untouched():
    model, memory, x = _trained("simple_cnn")
    layer = memory.layer_names[-1]
    before = memory.stats[layer].coact.clone()
    updates = memory.stats[layer].updates
    model.train()
    unit_class_influence(model, memory, layer, x[:8], batch_size=8)
    assert model.training, "train/eval mode must be restored"
    assert torch.equal(before, memory.stats[layer].coact)
    assert memory.stats[layer].updates == updates
