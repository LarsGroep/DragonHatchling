"""PackWriter → PackReader end-to-end round-trip (require the [ml] extra).

Skipped cleanly when torch/timm are unavailable; synthetic image, offline,
``pretrained=False``. Builds a complete pack from a fixture image and reads it
back: manifest schema-valid, attention dequantization error < 1/255 per element,
every declared asset present with its declared byte size.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

import jsonschema

from vitreous.data import DatasetSpec
from vitreous.instrument import Instrumenter
from vitreous.models import load_model
from vitreous.packs import PACK_VERSION, PackReader, build_pack, load_pack_schema


@pytest.fixture(scope="module")
def loaded():
    ds = DatasetSpec(
        name="toy",
        display_name="Toy dataset",
        num_classes=5,
        class_names=["a", "b", "c", "d", "e"],
    )
    return load_model("vit_s16", ds, pretrained=False), ds


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture(scope="module")
def pack_dir(loaded, image, tmp_path_factory):
    lm, ds = loaded
    out = tmp_path_factory.mktemp("pack") / "toy_0"
    build_pack(
        lm,
        image,
        {"id": "toy_0", "source": "gallery"},
        ds,
        out,
        ig_steps=4,
        faithfulness_steps=5,
        faithfulness_methods=("chefer",),
    )
    return out


def test_manifest_schema_valid(pack_dir):
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    jsonschema.validate(instance=manifest, schema=load_pack_schema())
    assert manifest["pack_version"] == PACK_VERSION == "1.0.0"


def test_all_declared_assets_exist_with_byte_sizes(pack_dir):
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    for name, entry in manifest["assets"].items():
        path = pack_dir / name
        assert path.exists(), f"missing asset {name}"
        assert path.stat().st_size == entry["bytes"], f"byte-size mismatch for {name}"


def test_expected_assets_present(pack_dir):
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    assets = manifest["assets"]
    for name in (
        "attention.bin",
        "tokens.bin",
        "attr_rollout.bin",
        "attr_chefer.bin",
        "attr_gradcam.bin",
        "attr_ig.bin",
        "attr_ig_pixel.png",
        "attributions.json",
        "faithfulness.json",
    ):
        assert name in assets, f"expected asset {name}"
    # An image asset exists (webp or png fallback).
    assert any(k.startswith("image.") for k in assets)
    # M3/M4 assets are absent and that's fine (schema tolerates it).
    assert "gaussians.bin" not in assets
    assert "graph.json" not in assets
    assert "concepts.json" not in assets


def test_attention_quant_layout(pack_dir):
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    att = manifest["assets"]["attention.bin"]
    assert att["encoding"] == "per_row_uint8"
    assert att["dtype"] == "uint8"
    assert att["shape"] == [12, 6, 197, 197]
    q = att["quant"]
    assert q["scheme"] == "per_row_uint8"
    assert q["row_axis"] == -1
    assert q["scale_count"] == 12 * 6 * 197
    assert q["data_offset"] == 0
    assert q["scale_offset"] == q["data_bytes"] == 12 * 6 * 197 * 197


def test_attention_dequant_error_below_one_255(pack_dir, loaded, image):
    lm, _ = loaded
    original = Instrumenter(lm).capture(image).attention.numpy()
    deq = PackReader(pack_dir).read_attention()
    assert deq.shape == (12, 6, 197, 197)
    assert deq.dtype == np.float32
    err = float(np.abs(original - deq).max())
    assert err < 1.0 / 255.0


def test_reader_roundtrips_arrays(pack_dir):
    rd = PackReader(pack_dir)
    tokens = rd.read_tokens()
    assert tokens.shape == (13, 197, 384)
    assert tokens.dtype == np.float16
    rollout = rd.read_array("attr_rollout.bin")
    assert rollout.shape == (12, 197)
    assert rollout.dtype == np.float32
    gradcam = rd.read_array("attr_gradcam.bin")
    assert gradcam.shape == (14, 14)


def test_reader_reads_json_assets(pack_dir):
    rd = PackReader(pack_dir)
    faith = rd.read_json("faithfulness.json")
    assert "deletion_auc" in faith and "agreement" in faith
    attr_index = rd.read_json("attributions.json")
    assert "chefer" in attr_index


def test_prediction_consistent(pack_dir):
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    pred = manifest["prediction"]
    probs = pred["probabilities"]
    assert len(probs) == manifest["dataset"]["num_classes"]
    assert pred["class_index"] == int(np.argmax(probs))
    assert pred["label"] == manifest["dataset"]["class_names"][pred["class_index"]]
