"""Tests for the universal-tool components: CUB loader, hybrid backbone,
attribute grounding, and the ONNX inference bundle."""

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
    build_loader,
    create_model,
)
from hatchvision.data import DatasetSpec, available_loaders
from hatchvision.explain import (
    cluster_concepts,
    find_exemplars,
    ground_concepts,
    ground_concepts_from_class_attributes,
)
from hatchvision.export import build_ivgraph, export_ivgraph, export_onnx_bundle

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


# ------------------------------------------------------------------ CUB-200


@pytest.fixture()
def mini_cub(tmp_path):
    """Synthetic 3-class, 12-image CUB_200_2011 directory."""
    from PIL import Image

    base = tmp_path / "CUB_200_2011"
    (base / "attributes").mkdir(parents=True)
    classes = ["001.Alpha_Bird", "002.Beta_Bird", "003.Gamma_Bird"]
    (base / "classes.txt").write_text(
        "".join(f"{i+1} {c}\n" for i, c in enumerate(classes))
    )
    images, labels, split = [], [], []
    for img_id in range(1, 13):
        cls = (img_id - 1) % 3
        rel = f"{classes[cls]}/img_{img_id}.jpg"
        p = base / "images" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color=(img_id * 20 % 255, 80, 120)).save(p)
        images.append(f"{img_id} {rel}\n")
        labels.append(f"{img_id} {cls + 1}\n")
        split.append(f"{img_id} {1 if img_id <= 6 else 0}\n")  # 6 train / 6 val
    (base / "images.txt").write_text("".join(images))
    (base / "image_class_labels.txt").write_text("".join(labels))
    (base / "train_test_split.txt").write_text("".join(split))

    attrs = ["has_wing_color::yellow", "has_bill_shape::hooked"]
    (base / "attributes" / "attributes.txt").write_text(
        "".join(f"{i+1} {a}\n" for i, a in enumerate(attrs))
    )
    lines = []
    for img_id in range(1, 13):
        for attr_id in (1, 2):
            present = 1 if (img_id + attr_id) % 2 == 0 else 0
            lines.append(f"{img_id} {attr_id} {present} 3 10.0\n")
    (base / "attributes" / "image_attribute_labels.txt").write_text("".join(lines))
    (base / "attributes" / "class_attribute_labels_continuous.txt").write_text(
        "80.0 10.0\n10.0 90.0\n50.0 50.0\n"
    )
    return tmp_path


def test_cub_loader(mini_cub):
    assert "cub200" in available_loaders()
    data = build_loader("cub200", root=str(mini_cub), image_size=32)
    assert data.spec.num_classes == 3
    assert data.spec.class_names[0] == "Alpha Bird"
    train, val = data.dataloaders(batch_size=4, num_workers=0)
    x, y = next(iter(train))
    assert x.shape == (4, 3, 32, 32)
    assert len(val.dataset) == 6

    names = data.attribute_names()
    assert names == ["wing color: yellow", "bill shape: hooked"]
    mat = data.val_attribute_matrix()
    assert mat.shape == (6, 2)
    assert set(mat.unique().tolist()) <= {0.0, 1.0}
    cls_mat = data.class_attribute_matrix()
    assert cls_mat.shape == (3, 2)

    probe = data.probe_batch(4)
    pattrs = data.probe_attributes(4)
    assert probe.shape[0] == pattrs.shape[0] == 4


# ------------------------------------------------------------------- hybrid


def test_hybrid_backbone_trains_and_is_sparse():
    model = create_model(
        "hybrid", SPEC, encoder="resnet18", pretrained=False, neuron_dim=128
    )
    frozen = [p for p in model.backbone.encoder.parameters()]
    assert all(not p.requires_grad for p in frozen), "encoder starts frozen"

    layers = model.hebbian_layers()
    assert list(layers) == ["neurons"]
    acts = {}
    h = layers["neurons"].register_forward_hook(
        lambda m, i, o: acts.setdefault("y", o.detach())
    )
    logits = model(torch.rand(2, 3, 32, 32))
    h.remove()
    assert logits.shape == (2, SPEC.num_classes)
    y = acts["y"]
    assert (y >= 0).all() and (y == 0).float().mean() > 0.1

    memory = HebbianFeatureMemory(model, num_classes=SPEC.num_classes, max_units=64)
    Trainer(model, TrainConfig(epochs=1, log_every=0), memory).fit(synthetic_loader())
    assert memory.stats["neurons"].updates > 0


# ---------------------------------------------------------------- grounding


def _trained_with_memory():
    model = create_model("simple_cnn", SPEC)
    memory = HebbianFeatureMemory(model, num_classes=SPEC.num_classes, max_units=32)
    Trainer(model, TrainConfig(epochs=1, log_every=0), memory).fit(synthetic_loader())
    return model, memory


def test_ground_concepts_labels_and_graph():
    model, memory = _trained_with_memory()
    layer = memory.layer_names[-1]
    concepts = cluster_concepts(memory, layer, SPEC.class_names, n_concepts=4)

    n_probe = 24
    probe = torch.rand(n_probe, 3, 32, 32)
    g = torch.Generator().manual_seed(1)
    attr_matrix = (torch.rand(n_probe, 3, generator=g) > 0.5).float()
    names = ["wing color: yellow", "bill shape: hooked", "belly: white"]
    ground_concepts(
        concepts, memory, model, probe, attr_matrix, names,
        min_support=2, min_effect=0.0,
    )
    assert any(c.attributes for c in concepts)
    grounded = next(c for c in concepts if c.attributes)
    assert grounded.label.split(" · ")[0] in names

    doc = build_ivgraph(memory, concepts, layer, SPEC.class_names)
    kinds = {e["kind"] for e in doc["edges"]}
    assert "attribute" in kinds
    attr_nodes = [n for n in doc["nodes"] if n["type"] == "attribute"]
    assert attr_nodes
    node_ids = {n["id"] for n in doc["nodes"]}
    assert all(e["source"] in node_ids and e["target"] in node_ids for e in doc["edges"])


def test_ground_concepts_from_class_attributes():
    model, memory = _trained_with_memory()
    layer = memory.layer_names[-1]
    concepts = cluster_concepts(memory, layer, SPEC.class_names, n_concepts=4)
    cls_attrs = torch.tensor(
        [[90.0, 5.0], [5.0, 90.0], [50.0, 50.0], [20.0, 20.0]]
    )
    ground_concepts_from_class_attributes(
        concepts, cls_attrs, ["red beak", "long tail"], SPEC.class_names
    )
    assert any(c.attributes for c in concepts)


def test_ground_concepts_shape_mismatch_raises():
    model, memory = _trained_with_memory()
    layer = memory.layer_names[-1]
    concepts = cluster_concepts(memory, layer, SPEC.class_names, n_concepts=2)
    with pytest.raises(ValueError):
        ground_concepts(
            concepts, memory, model,
            torch.rand(8, 3, 32, 32), torch.zeros(6, 2), ["a", "b"],
        )


# -------------------------------------------------------------- ONNX bundle


def test_onnx_bundle_matches_pytorch(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")

    model, memory = _trained_with_memory()
    layer = memory.layer_names[-1]
    concepts = cluster_concepts(memory, layer, SPEC.class_names, n_concepts=4)
    export_ivgraph(memory, concepts, layer, SPEC.class_names, tmp_path / "graph.json")
    manifest_path = export_onnx_bundle(model, memory, SPEC, tmp_path)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["class_names"] == list(SPEC.class_names)
    assert manifest["image_size"] == SPEC.image_size
    outs = manifest["activation_outputs"]
    assert outs and outs[-1]["layer"] == layer
    assert outs[-1]["node_prefix"] == f"u:{layer}:"

    # exporting must not leave hooks or eval mode behind
    model_hooks = [
        m._forward_hooks for m in model.modules() if m._forward_hooks
    ]
    # only the Hebbian memory's own hooks may remain
    assert len(model_hooks) <= len(memory.layer_names)

    sess = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"))
    x = torch.rand(2, 3, 32, 32)
    ort_outs = sess.run(None, {"images": x.numpy()})
    names = [o.name for o in sess.get_outputs()]
    assert names[0] == "logits"

    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert torch.allclose(logits, torch.tensor(ort_outs[0]), atol=1e-4)

    act = ort_outs[names.index(outs[-1]["output"])]
    assert act.shape == (2, outs[-1]["units"])
    assert (act >= 0).all()
