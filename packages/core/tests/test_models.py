"""Model loader tests (require the [ml] extra: torch + timm).

Skipped cleanly when torch/timm are unavailable. Models are always built with
``pretrained=False`` — no weights are ever downloaded.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from vitreous.data import DatasetSpec
from vitreous.models import (
    LoadedModel,
    ModelSpec,
    get_model_spec,
    list_models,
    load_model,
)


def test_vit_s16_registered():
    assert "vit_s16" in list_models()
    spec = get_model_spec("vit_s16")
    assert isinstance(spec, ModelSpec)
    assert spec.arch == "deit_small_patch16_224"
    assert spec.patch_size == 16


def test_load_model_output_shape():
    ds = DatasetSpec(name="toy", display_name="Toy", num_classes=7)
    loaded = load_model("vit_s16", ds, pretrained=False)
    assert isinstance(loaded, LoadedModel)
    assert loaded.num_classes == 7

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = loaded.module(x)
    assert logits.shape == (2, 7)


def test_load_model_fresh_head_matches_num_classes():
    ds = DatasetSpec(name="toy", display_name="Toy", num_classes=10)
    loaded = load_model("vit_s16", ds)
    # timm exposes the classifier via get_classifier(); out_features == 10.
    head = loaded.module.get_classifier()
    assert head.out_features == 10


def test_load_model_explicit_num_classes_overrides_spec():
    ds = DatasetSpec(name="toy", display_name="Toy", num_classes=10)
    loaded = load_model("vit_s16", ds, num_classes=3)
    assert loaded.num_classes == 3


def test_to_model_info_kwargs_has_patch_size():
    ds = DatasetSpec(name="toy", display_name="Toy", num_classes=4)
    loaded = load_model("vit_s16", ds)
    kw = loaded.to_model_info_kwargs()
    assert kw["patch_size"] == 16
    assert kw["num_layers"] == 12
    assert kw["num_tokens"] == 197
