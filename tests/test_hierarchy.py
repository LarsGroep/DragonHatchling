"""Concept hierarchy, Hebbian heads, few-shot enrollment, hierarchy export."""

import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hatchvision import (
    HebbianFeatureMemory,
    TrainConfig,
    Trainer,
    build_loader,
    create_model,
)
from hatchvision.explain import probe_activations
from hatchvision.export import build_hierarchy_pack, export_hierarchy_pack
from hatchvision.hebbian import (
    ConceptBottleneckHead,
    ConceptNode,
    HebbianPrototypeHead,
    TreeRoutedHead,
    build_concept_tree,
    node_scores,
)

# --------------------------------------------------------- synthetic memory


def _block_memory(n_units=64, n_samples=600, seed=0):
    """Memory with nested block structure via from_state.

    Two super-blocks (units [0,32) and [32,64)), each split into two
    sub-blocks of 16.  A sample fires its sub-block strongly and the sibling
    sub-block weakly, so correlation is high within sub-blocks, moderate
    within super-blocks, and near zero across them — the dendrogram should
    recover super-blocks at depth 1 and sub-blocks at depth 2.
    """
    g = torch.Generator().manual_seed(seed)
    sub_blocks = [list(range(i * 16, (i + 1) * 16)) for i in range(4)]
    acts = torch.zeros(n_samples, n_units)
    labels = torch.zeros(n_samples, dtype=torch.long)
    for s in range(n_samples):
        sb = s % 4
        labels[s] = sb
        sibling = sb ^ 1                      # other sub-block of the super-block
        acts[s, sub_blocks[sb]] = 0.6 + 0.4 * torch.rand(16, generator=g)
        acts[s, sub_blocks[sibling]] = 0.15 + 0.1 * torch.rand(16, generator=g)
        acts[s] += 0.01 * torch.rand(n_units, generator=g)

    a_hat = acts / (acts.norm(dim=1, keepdim=True) + 1e-8)
    class_act = torch.zeros(4, n_units)
    class_count = torch.zeros(4)
    class_act.index_add_(0, labels, a_hat)
    class_count.index_add_(0, labels, torch.ones(n_samples))
    state = {
        "num_classes": 4,
        "momentum": 0.05,
        "layers": {
            "layer": {
                "coact": a_hat.t() @ a_hat / n_samples,
                "mean_act": a_hat.mean(dim=0),
                "class_act": class_act,
                "class_count": class_count,
                "updates": n_samples,
                "unit_index": None,
            }
        },
    }
    return HebbianFeatureMemory.from_state(state), acts, labels


CLASS_NAMES = ("a", "b", "c", "d")


def test_tree_recovers_block_structure():
    memory, _, _ = _block_memory()
    tree = build_concept_tree(memory, "layer", CLASS_NAMES, max_depth=2)

    assert len(tree.units) == 64
    assert len(tree.children) == 2
    # depth 1 = super-blocks
    super_blocks = [set(range(32)), set(range(32, 64))]
    for child in tree.children:
        assert set(child.units) in super_blocks
        # depth 2 = sub-blocks
        assert len(child.children) == 2
        subs = {frozenset(gc.units) for gc in child.children}
        expected = {
            frozenset(u for u in child.units if u < min(child.units) + 16),
            frozenset(u for u in child.units if u >= min(child.units) + 16),
        }
        assert subs == expected


def test_tree_children_partition_parent_and_coherence_rises():
    memory, _, _ = _block_memory()
    tree = build_concept_tree(memory, "layer", CLASS_NAMES, max_depth=3)
    for node in tree.walk():
        if node.children:
            union = sorted(u for c in node.children for u in c.units)
            assert union == sorted(node.units), "children must partition parent"
            sizes = sum(len(c.units) for c in node.children)
            assert sizes == len(node.units), "children must be disjoint"
            mean_child = sum(c.coherence for c in node.children) / len(node.children)
            assert mean_child >= node.coherence - 1e-6, (
                "splitting must not reduce mean coherence"
            )
    # on the block-structured memory coherence strictly increases at the root
    assert all(c.coherence > tree.coherence for c in tree.children)


def test_tree_respects_stopping_rules():
    memory, _, _ = _block_memory()
    t1 = build_concept_tree(memory, "layer", CLASS_NAMES, max_depth=1)
    assert max(n.depth for n in t1.walk()) <= 1
    big = build_concept_tree(memory, "layer", CLASS_NAMES, max_depth=6, min_units=20)
    for node in big.walk():
        assert len(node.units) >= 20 or node is big


def test_node_scores_normalized_and_shaped():
    memory, acts, _ = _block_memory()
    tree = build_concept_tree(memory, "layer", CLASS_NAMES, max_depth=2)
    scores = node_scores(tree, {"layer": acts[:10]})
    assert set(scores) == {n.node_id for n in tree.walk()}
    for v in scores.values():
        assert v.shape == (10,)
        assert torch.isfinite(v).all()
    # sample 0 fires sub-block 0 → the child containing units 0..31 wins
    left = next(c for c in tree.children if 0 in c.units)
    right = next(c for c in tree.children if 0 not in c.units)
    assert scores[left.node_id][0] > scores[right.node_id][0]


def test_tree_roundtrip_and_heads_on_synthetic():
    memory, acts, labels = _block_memory()
    tree = build_concept_tree(memory, "layer", CLASS_NAMES, max_depth=2)

    # JSON round-trip
    doc = json.loads(json.dumps(tree.to_dict()))
    tree2 = ConceptNode.from_dict(doc)
    assert [n.node_id for n in tree2.walk()] == [n.node_id for n in tree.walk()]
    assert [n.units for n in tree2.walk()] == [n.units for n in tree.walk()]

    # each class fires exactly one sub-block → every head should be perfect
    proto = HebbianPrototypeHead.from_memory(memory, "layer", CLASS_NAMES)
    assert (proto.predict(acts) == labels).float().mean() > 0.99

    routed = TreeRoutedHead(tree, CLASS_NAMES)
    assert (routed.predict(acts, mode="hard") == labels).float().mean() > 0.99
    assert (routed.predict(acts, mode="soft") == labels).float().mean() > 0.99
    path = routed.decision_path(acts[0])
    assert path[0] == tree.node_id and len(path) == 3

    cb = ConceptBottleneckHead.from_memory(memory, "layer", CLASS_NAMES, n_concepts=4)
    assert (cb.predict(acts) == labels).float().mean() > 0.99


def test_prototype_enroll_new_class_synthetic():
    memory, acts, labels = _block_memory()
    proto = HebbianPrototypeHead.from_memory(memory, "layer", CLASS_NAMES)
    before = (proto.predict(acts) == labels).float().mean()

    # a brand-new firing pattern: only the tail units the blocks never used
    g = torch.Generator().manual_seed(1)
    new = torch.zeros(8, 64)
    new[:, 60:64] = 5.0 + torch.rand(8, 4, generator=g)
    idx = proto.enroll("e", new[:4])
    assert proto.class_names[idx] == "e"
    assert (proto.predict(new[4:]) == idx).all(), "enrolled class must be recognized"
    after = (proto.predict(acts) == labels).float().mean()
    assert before - after <= 0.02, "enrollment must not disturb existing classes"


# ------------------------------------------------------------- shapes e2e


@pytest.fixture(scope="module")
def shapes_run(tmp_path_factory):
    """Tiny shapes training run: 64px, simple_cnn, 2 epochs, ~840 images."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import make_shapes_dataset as msd

    root = tmp_path_factory.mktemp("shapes")
    rng = random.Random(0)
    for split, count in (("train", 120), ("val", 20)):
        for cls in msd.CLASSES:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                msd.make_image(cls, 64, rng).save(d / f"{cls}_{i:04d}.png")

    torch.manual_seed(0)
    loader = build_loader("imagefolder", root=str(root), image_size=64)
    train_loader, _ = loader.dataloaders(batch_size=32, num_workers=0)
    model = create_model("simple_cnn", loader.spec)
    memory = HebbianFeatureMemory(model, num_classes=loader.spec.num_classes, max_units=96)
    Trainer(model, TrainConfig(epochs=2, log_every=0), memory).fit(train_loader)

    val_x, val_y = [], []
    for xb, yb in DataLoader(loader.val_dataset(), batch_size=32, num_workers=0):
        val_x.append(xb)
        val_y.append(yb)
    return {
        "model": model,
        "memory": memory,
        "spec": loader.spec,
        "layer": memory.layer_names[-1],
        "val_x": torch.cat(val_x),
        "val_y": torch.cat(val_y),
    }


def test_shapes_heads_beat_2x_chance(shapes_run):
    r = shapes_run
    acts = probe_activations(r["model"], r["val_x"], memory=r["memory"])[r["layer"]]
    chance = 1.0 / r["spec"].num_classes
    names = list(r["spec"].class_names)

    proto = HebbianPrototypeHead.from_memory(r["memory"], r["layer"], names)
    proto_acc = (proto.predict(acts) == r["val_y"]).float().mean()
    assert proto_acc > 2 * chance, f"prototype {proto_acc:.3f} <= 2x chance"

    tree = build_concept_tree(r["memory"], r["layer"], names, max_depth=3)
    routed = TreeRoutedHead(tree, names)
    soft_acc = (routed.predict(acts, mode="soft") == r["val_y"]).float().mean()
    assert soft_acc > 2 * chance, f"soft tree {soft_acc:.3f} <= 2x chance"


def test_shapes_enrollment_no_interference(shapes_run):
    r = shapes_run
    acts = probe_activations(r["model"], r["val_x"], memory=r["memory"])[r["layer"]]
    names = list(r["spec"].class_names)

    # refreshed prototypes (final-model footing) — required for enrollment
    proto = HebbianPrototypeHead.from_activations(
        r["layer"], names, acts, r["val_y"]
    )
    before = (proto.predict(acts) == r["val_y"]).float().mean()

    # synthetic extra class: images unlike any shape (dark diagonal stripes)
    n = 12
    stripe = torch.zeros(n, 3, 64, 64)
    for i in range(n):
        for d in range(0, 64, 8):
            idx = torch.arange(64)
            stripe[i, :, idx, (idx + d) % 64] = 1.0
    mean, std = r["spec"].normalization()
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    stripe = (stripe - m) / s
    stripe_acts = probe_activations(r["model"], stripe, memory=r["memory"])[r["layer"]]

    idx = proto.enroll("stripes", stripe_acts[:5])
    assert proto.class_names[idx] == "stripes"
    # disjoint eval images of the new class are recognized
    assert (proto.predict(stripe_acts[5:]) == idx).float().mean() > 0.5
    after = (proto.predict(acts) == r["val_y"]).float().mean()
    assert before - after <= 0.05, (
        f"enrollment degraded existing classes {before:.3f} -> {after:.3f}"
    )


# ------------------------------------------------------------- export path


def test_hierarchy_pack_roundtrip(tmp_path, shapes_run):
    r = shapes_run
    names = list(r["spec"].class_names)
    tree = build_concept_tree(r["memory"], r["layer"], names, max_depth=2)
    path = export_hierarchy_pack(
        r["memory"], r["layer"], names, tree, tmp_path / "hierarchy.json"
    )
    doc = json.loads(path.read_text())
    assert doc["format"] == "hatchvision-hierarchy"
    assert doc["layer"] == r["layer"]
    assert doc["node_prefix"] == f"u:{r['layer']}:"
    assert doc["class_names"] == names
    assert len(doc["unit_ids"]) == r["memory"].stats[r["layer"]].dim
    assert set(doc["prototypes"]) == set(names)
    n_units = r["memory"].stats[r["layer"]].dim
    for vec in doc["prototypes"].values():
        assert len(vec) == n_units
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-2, "prototypes must be L2-normalized"
    # the embedded tree parses back into an identical ConceptNode tree
    tree2 = ConceptNode.from_dict(doc["tree"])
    assert [n.units for n in tree2.walk()] == [n.units for n in tree.walk()]
    assert "temperature" in doc["config"]


def test_hierarchy_pack_with_patches(tmp_path, shapes_run):
    from hatchvision.explain import attach_patches, node_patch_uris

    r = shapes_run
    names = list(r["spec"].class_names)
    tree = build_concept_tree(r["memory"], r["layer"], names, max_depth=1)
    mean, std = r["spec"].normalization()
    patches = node_patch_uris(
        tree, r["model"], r["memory"], r["val_x"][:12], mean, std,
        exemplars=2, grid=4,
    )
    attach_patches(tree, patches)
    assert all(len(v) == 2 for v in patches.values())
    assert all(u.startswith("data:image/png;base64,") for v in patches.values() for u in v)

    doc = build_hierarchy_pack(r["memory"], r["layer"], names, tree)
    node_patches = doc["tree"].get("patches", [])
    assert node_patches and node_patches[0].startswith("data:image/png;base64,")
