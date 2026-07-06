"""Instrumenter tests (require the [ml] extra: torch + timm).

Skipped cleanly when torch/timm are unavailable. Models are built with
``pretrained=False``.

Key guarantees exercised here:
* hook purity — logits bit-identical with/without the Instrumenter attached;
* Trace shapes exactly [12, 6, 197, 197] and [13, 197, 384] for a 224² input;
* hooks fully removed after the context exits (no lingering handles).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from vitreous.data import DatasetSpec
from vitreous.instrument import Instrumenter, Trace
from vitreous.models import load_model


@pytest.fixture(scope="module")
def loaded():
    ds = DatasetSpec(name="toy", display_name="Toy", num_classes=5)
    return load_model("vit_s16", ds, pretrained=False)


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224)


def _count_hooks(module):
    """Total forward + forward_pre hooks registered on a module subtree."""
    total = 0
    for m in module.modules():
        total += len(m._forward_hooks) + len(m._forward_pre_hooks)
    return total


def test_capture_returns_trace(loaded, image):
    trace = Instrumenter(loaded).capture(image)
    assert isinstance(trace, Trace)
    assert trace.attention is not None
    assert trace.tokens is not None
    assert trace.logits is not None
    assert "forward_ms" in trace.timings


def test_trace_shapes_exact(loaded, image):
    trace = Instrumenter(loaded).capture(image)
    assert tuple(trace.attention.shape) == (12, 6, 197, 197)
    assert tuple(trace.tokens.shape) == (13, 197, 384)
    assert tuple(trace.logits.shape) == (1, 5)


def test_attention_is_softmax_distribution(loaded, image):
    trace = Instrumenter(loaded).capture(image)
    # Each attention row sums to 1 (it is a softmax over keys).
    row_sums = trace.attention.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    assert torch.all(trace.attention >= 0)


def test_hook_purity_logits_bit_identical(loaded, image):
    model = loaded.module
    model.eval()
    with torch.no_grad():
        logits_plain = model(image)

    trace = Instrumenter(loaded).capture(image)

    # Bit-identical: hooks observe, never perturb.
    assert torch.equal(logits_plain, trace.logits)


def test_hooks_removed_after_context(loaded, image):
    model = loaded.module
    before = _count_hooks(model)

    with Instrumenter(loaded) as inst:
        during = _count_hooks(model)
        inst.capture(image)
    after = _count_hooks(model)

    assert during > before  # hooks were registered inside the context
    assert after == before  # and fully removed on exit
    assert before == 0


def test_standalone_capture_leaves_no_hooks(loaded, image):
    model = loaded.module
    Instrumenter(loaded).capture(image)
    assert _count_hooks(model) == 0


def test_capture_accepts_unbatched_image(loaded):
    torch.manual_seed(1)
    img = torch.randn(3, 224, 224)  # no batch dim
    trace = Instrumenter(loaded).capture(img)
    assert tuple(trace.attention.shape) == (12, 6, 197, 197)
    assert tuple(trace.tokens.shape) == (13, 197, 384)


def test_non_vit_model_rejected():
    lin = torch.nn.Linear(4, 4)
    with pytest.raises(TypeError):
        Instrumenter(lin)
