"""Smoke tests: every piece works end-to-end on tiny synthetic data."""

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
from hatchvision.data import DatasetSpec, available_loaders
from hatchvision.explain import GradCAM, cluster_concepts, find_exemplars
from hatchvision.export import build_ivgraph
from hatchvision.models import available_backbones

SPEC = DatasetSpec(
    name="synthetic",
    num_classes=4,
    class_names=("a", "b", "c", "d"),
    image_size=32,
    in_channels=3,
)


def synthetic_loader(n=32, batch_size=8):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, 3, 32, 32, generator=g)
    y = torch.randint(0, SPEC.num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_registries_populated():
    assert {"simple_cnn", "resnet18", "bdh"} <= set(available_backbones())
    assert {"cifar10", "imagefolder", "ham10000", "isic"} <= set(available_loaders())


@pytest.mark.parametrize("backbone", ["simple_cnn", "bdh", "resnet18"])
def test_backbones_forward(backbone):
    model = create_model(backbone, SPEC)
    x = torch.rand(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, SPEC.num_classes)
    assert model.hebbian_layers(), "every backbone exposes hebbian layers"


def test_bdh_neuron_activations_are_sparse_positive():
    model = create_model("bdh", SPEC)
    acts = {}
    layer_name, layer = next(iter(model.hebbian_layers().items()))
    h = layer.register_forward_hook(lambda m, i, o: acts.setdefault("y", o.detach()))
    model(torch.rand(2, 3, 32, 32))
    h.remove()
    y = acts["y"]
    assert (y >= 0).all(), "BDH neuron activations must be non-negative"
    assert (y == 0).float().mean() > 0.1, "BDH activations should be sparse"


def test_hebbian_memory_does_not_affect_training():
    torch.manual_seed(0)
    model_a = create_model("simple_cnn", SPEC)
    torch.manual_seed(0)
    model_b = create_model("simple_cnn", SPEC)
    loader = synthetic_loader()

    torch.manual_seed(1)
    Trainer(model_a, TrainConfig(epochs=1, log_every=0)).fit(loader)

    torch.manual_seed(1)
    memory = HebbianFeatureMemory(model_b, num_classes=SPEC.num_classes)
    Trainer(model_b, TrainConfig(epochs=1, log_every=0), memory).fit(loader)

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb), "Hebbian memory must not change optimization"
    assert memory.stats, "memory recorded statistics"


def test_hebbian_statistics_and_edges():
    model = create_model("simple_cnn", SPEC)
    memory = HebbianFeatureMemory(model, num_classes=SPEC.num_classes, max_units=32)
    Trainer(model, TrainConfig(epochs=1, log_every=0), memory).fit(synthetic_loader())
    layer = memory.layer_names[-1]
    corr = memory.correlation(layer)
    assert corr.shape[0] <= 32
    assert torch.isfinite(corr).all()
    edges = memory.top_edges(layer, k=10)
    assert edges and all(0.0 <= w <= 1.0 + 1e-5 for _, _, w in edges)
    affinity = memory.class_affinity(layer)
    assert affinity.shape[0] == SPEC.num_classes


def test_gradcam_shapes_and_range():
    model = create_model("simple_cnn", SPEC)
    with GradCAM(model) as cam:
        maps = cam(torch.rand(2, 3, 32, 32))
    assert maps.shape == (2, 32, 32)
    assert maps.min() >= 0 and maps.max() <= 1.0 + 1e-5


def test_concepts_and_ivgraph_export(tmp_path):
    model = create_model("simple_cnn", SPEC)
    memory = HebbianFeatureMemory(model, num_classes=SPEC.num_classes, max_units=32)
    Trainer(model, TrainConfig(epochs=1, log_every=0), memory).fit(synthetic_loader())
    layer = memory.layer_names[-1]
    concepts = cluster_concepts(memory, layer, SPEC.class_names, n_concepts=4)
    assert concepts
    probe = torch.rand(16, 3, 32, 32)
    find_exemplars(concepts, memory, model, probe, top_k=3)
    assert all(len(c.exemplars) == 3 for c in concepts)

    doc = build_ivgraph(memory, concepts, layer, SPEC.class_names)
    assert doc["format"] == "ivgraph"
    node_ids = {n["id"] for n in doc["nodes"]}
    for e in doc["edges"]:
        assert e["source"] in node_ids and e["target"] in node_ids
    # round-trips through JSON
    json.loads(json.dumps(doc))


def test_ivgraph_carries_neuron_coactivation_network():
    """The exported graph must contain the unit-level co-activation edges the
    web app's 'Neurons' view draws — including intra-concept links that the
    global top-K ranking would otherwise drop."""
    model = create_model("simple_cnn", SPEC)
    memory = HebbianFeatureMemory(model, num_classes=SPEC.num_classes, max_units=32)
    Trainer(model, TrainConfig(epochs=1, log_every=0), memory).fit(synthetic_loader())
    layer = memory.layer_names[-1]
    concepts = cluster_concepts(memory, layer, SPEC.class_names, n_concepts=4)

    doc = build_ivgraph(memory, concepts, layer, SPEC.class_names)
    coact = [e for e in doc["edges"] if e["kind"] == "coactivation"]
    assert coact, "no co-activation edges — neuron network would be edgeless"

    # de-duplicated undirected pairs
    pairs = {frozenset((e["source"], e["target"])) for e in coact}
    assert len(pairs) == len(coact)

    # at least one edge lives strictly inside a concept's own units
    def unit_id(u):
        return f"u:{layer}:{u}"

    intra = False
    for c in concepts:
        uset = {unit_id(u) for u in c.units}
        if any(e["source"] in uset and e["target"] in uset for e in coact):
            intra = True
            break
    assert intra, "expected at least one intra-concept co-activation edge"
