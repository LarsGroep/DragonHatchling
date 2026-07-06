"""XAI method suite tests (require the [ml] extra: torch + timm).

Skipped cleanly when torch/timm are unavailable; all models built with
``pretrained=False`` and synthetic images — fully offline.

Covers, per method: correct shapes/dtypes, rollout rows derived from true
softmax attention, Chefer/IG class-specificity, determinism with fixed seeds,
and that the M1 hook-purity guarantee still holds after adding the grad-enabled
capture path.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from vitreous.data import DatasetSpec
from vitreous.instrument import Instrumenter
from vitreous.models import load_model
from vitreous.xai import (
    attention_rollout,
    chefer_relevance,
    grad_cam,
    integrated_gradients,
)


@pytest.fixture(scope="module")
def loaded():
    ds = DatasetSpec(name="toy", display_name="Toy", num_classes=5)
    return load_model("vit_s16", ds, pretrained=False)


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224)


def _count_hooks(module):
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in module.modules()
    )


# -- rollout ---------------------------------------------------------------- #


def test_rollout_shape_dtype(loaded, image):
    trace = Instrumenter(loaded).capture(image)
    attr = attention_rollout(trace)
    assert attr.token_scores.shape == (12, 197)
    assert attr.token_scores.dtype == np.float32


def test_rollout_rows_derived_from_true_softmax(loaded, image):
    trace = Instrumenter(loaded).capture(image)
    attr = attention_rollout(trace)
    # Recompute layer-0 rollout directly from the captured softmax attention.
    A = trace.attention.float()
    T = A.shape[-1]
    eye = torch.eye(T)
    a0 = A[0].mean(dim=0)
    a0 = 0.5 * a0 + 0.5 * eye
    a0 = a0 / a0.sum(dim=-1, keepdim=True)
    assert np.allclose(attr.token_scores[0], a0[0].numpy(), atol=1e-5)
    # Every cumulative row is row-stochastic → CLS relevance sums to 1.
    assert np.allclose(attr.token_scores.sum(axis=1), 1.0, atol=1e-4)


# -- chefer ----------------------------------------------------------------- #


def test_chefer_shape_dtype_formulation(loaded, image):
    attr = chefer_relevance(loaded, image, class_idx=0)
    assert attr.token_scores.shape == (12, 197)
    assert attr.token_scores.dtype == np.float32
    assert attr.meta["formulation"] == "grad_weighted_rollout_iccv2021"
    assert attr.meta["final"].shape == (197,)


def test_chefer_class_specific(loaded, image):
    a0 = chefer_relevance(loaded, image, class_idx=0)
    a3 = chefer_relevance(loaded, image, class_idx=3)
    assert not np.allclose(a0.token_scores[-1], a3.token_scores[-1])


def test_chefer_deterministic(loaded, image):
    a = chefer_relevance(loaded, image, class_idx=1)
    b = chefer_relevance(loaded, image, class_idx=1)
    assert np.array_equal(a.token_scores, b.token_scores)


# -- grad-cam --------------------------------------------------------------- #


def test_gradcam_shape_dtype_nonneg(loaded, image):
    attr = grad_cam(loaded, image, class_idx=0)
    assert attr.pixel_map.shape == (14, 14)
    assert attr.pixel_map.dtype == np.float32
    assert np.all(attr.pixel_map >= 0)  # ReLU'd


# -- integrated gradients --------------------------------------------------- #


def test_ig_shapes_dtypes(loaded, image):
    attr = integrated_gradients(loaded, image, class_idx=0, steps=4)
    assert attr.token_scores.shape == (197,)
    assert attr.pixel_map.shape == (224, 224)
    assert attr.token_scores.dtype == np.float32
    assert attr.pixel_map.dtype == np.float32


def test_ig_class_specific(loaded, image):
    a0 = integrated_gradients(loaded, image, class_idx=0, steps=4)
    a3 = integrated_gradients(loaded, image, class_idx=3, steps=4)
    assert not np.allclose(a0.token_scores, a3.token_scores)


def test_ig_deterministic(loaded, image):
    a = integrated_gradients(loaded, image, class_idx=2, steps=4)
    b = integrated_gradients(loaded, image, class_idx=2, steps=4)
    assert np.array_equal(a.token_scores, b.token_scores)
    assert np.array_equal(a.pixel_map, b.pixel_map)


# -- hook purity after grad capture ----------------------------------------- #


def test_grad_capture_leaves_no_hooks(loaded, image):
    module = loaded.module
    Instrumenter(loaded).capture_with_grad(image, target=0)
    assert _count_hooks(module) == 0


def test_hook_purity_holds_after_grad_capture(loaded, image):
    """The monkeypatched grad path must restore the model exactly.

    A plain forward before and after a grad capture must be bit-identical, and
    the observation-only capture must remain bit-identical to a plain forward.
    """
    module = loaded.module
    module.eval()
    with torch.no_grad():
        before = module(image)

    # Exercise the grad-enabled path (installs + removes the monkeypatch).
    chefer_relevance(loaded, image, class_idx=0)

    with torch.no_grad():
        after = module(image)
    assert torch.equal(before, after)

    # And the observation-only capture is still bit-identical to a plain forward.
    trace = Instrumenter(loaded).capture(image)
    assert torch.equal(before, trace.logits)


def test_grad_capture_target_defaults_to_argmax(loaded, image):
    module = loaded.module
    module.eval()
    with torch.no_grad():
        pred = int(module(image)[0].argmax().item())
    trace = Instrumenter(loaded).capture_with_grad(image, target=None)
    assert trace.meta["target"] == pred
    assert trace.attention.shape == (12, 6, 197, 197)
    assert trace.attention_grad.shape == (12, 6, 197, 197)
