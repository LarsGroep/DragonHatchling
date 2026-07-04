"""ISIC loader: HAM-style CSV, one-hot CSV, folder-per-class, no leakage."""

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hatchvision.data import available_loaders, build_loader
from hatchvision.data.isic import pretty_dx


def _img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 90, 90)).save(path)


def test_isic_registered():
    assert "isic" in available_loaders()


def test_pretty_dx():
    assert pretty_dx("mel") == "Melanoma"
    assert pretty_dx("NV") == "Melanocytic nevus"
    assert pretty_dx("weird_code") == "Weird Code"


def _write_ham(root: Path, n_per_class=6, imgs_per_lesion=2):
    """HAM10000-style: metadata.csv (image_id, dx, lesion_id) + image files.
    Each lesion contributes several images (the leakage trap)."""
    img_dir = root / "HAM10000_images_part_1"
    meta = root / "HAM10000_metadata.csv"
    rows = [["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]]
    dxs = ["mel", "nv", "bkl"]
    k = 0
    lesion_of = {}
    for dx in dxs:
        for lesion in range(n_per_class):
            lid = f"HAM_{dx}_{lesion}"
            for _ in range(imgs_per_lesion):
                iid = f"ISIC_{k:07d}"
                lesion_of[iid] = lid
                _img(img_dir / f"{iid}.jpg")
                rows.append([lid, iid, dx, "histo", "50", "male", "back"])
                k += 1
    with open(meta, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return lesion_of


def test_isic_ham_csv_split_and_no_leakage(tmp_path):
    lesion_of = _write_ham(tmp_path)
    data = build_loader("isic", root=str(tmp_path), image_size=32, val_frac=0.34, seed=0)
    assert data.spec.num_classes == 3
    assert set(data.spec.class_names) == {"Melanoma", "Melanocytic nevus", "Benign keratosis"}

    train, val = data.train_dataset(), data.val_dataset()
    assert len(train) > 0 and len(val) > 0
    assert hasattr(train, "targets") and len(train.targets) == len(train)

    # a sample loads as a (tensor, int) pair of the right shape
    x, y = train[0]
    assert x.shape == (3, 32, 32) and isinstance(y, int)

    # NO lesion appears in both splits (grouped split prevents leakage)
    def lesions(ds):
        return {lesion_of[p.stem] for p, _ in ds.samples}

    assert lesions(train).isdisjoint(lesions(val)), "lesion leaked across splits"

    # every class is represented in train
    assert set(train.targets) == set(range(3))

    # no per-image attributes → concept grounding falls back to class affinity
    assert data.attribute_names() is None
    assert data.val_attribute_matrix() is None


def test_isic_onehot_csv(tmp_path):
    """ISIC-2019-style one-hot ground-truth CSV."""
    img_dir = tmp_path / "images"
    rows = [["image", "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]]
    onehot = {"MEL": 0, "NV": 1, "BCC": 2}
    for code, col in onehot.items():
        for j in range(5):
            iid = f"ISIC_{code}_{j}"
            _img(img_dir / f"{iid}.jpg")
            row = [iid] + ["0"] * 9
            row[1 + col] = "1.0"
            rows.append(row)
    with open(tmp_path / "ISIC_2019_Training_GroundTruth.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    data = build_loader("isic", root=str(tmp_path), image_size=32, seed=0)
    assert data.spec.num_classes == 3
    assert "Melanoma" in data.spec.class_names and "Basal cell carcinoma" in data.spec.class_names
    assert len(data.train_dataset()) + len(data.val_dataset()) == 15


def test_isic_folder_per_class(tmp_path):
    for split in ("Train", "Test"):
        for cls in ("melanoma", "nevus"):
            for j in range(4):
                _img(tmp_path / split / cls / f"{cls}_{j}.png")
    data = build_loader("isic", root=str(tmp_path), image_size=32)
    assert data.spec.num_classes == 2
    assert len(data.train_dataset()) == 8 and len(data.val_dataset()) == 8


def test_isic_end_to_end_bundle(tmp_path):
    """Tiny train + full bundle export on synthetic ISIC data."""
    import torch
    from hatchvision import HebbianFeatureMemory, TrainConfig, Trainer, create_model
    from hatchvision.explain import cluster_concepts, find_exemplars
    from hatchvision.export import build_explain_pack, build_ivgraph

    _write_ham(tmp_path, n_per_class=5, imgs_per_lesion=2)
    data = build_loader("isic", root=str(tmp_path), image_size=32, seed=0)
    tl, vl = data.dataloaders(batch_size=8, num_workers=0)
    model = create_model("simple_cnn", data.spec)
    mem = HebbianFeatureMemory(model, num_classes=data.spec.num_classes, max_units=32)
    Trainer(model, TrainConfig(epochs=1, log_every=0), mem).fit(tl, vl)

    layer = mem.layer_names[-1]
    concepts = cluster_concepts(mem, layer, data.spec.class_names, n_concepts=3)
    probe = data.probe_batch(16)
    find_exemplars(concepts, mem, model, probe)
    # class-grounded (no attributes): labels come from class affinity
    assert all(not c.attributes for c in concepts)

    doc = build_ivgraph(mem, concepts, layer, data.spec.class_names)
    assert doc["format"] == "ivgraph"
    pack = build_explain_pack(mem, layer, data.spec.class_names, model=model, background=probe[:8])
    assert pack["num_classes"] == data.spec.num_classes
    assert pack["shap"]["method"] in ("exact-linear", "expected-gradients")
