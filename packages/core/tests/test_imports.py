"""Import-everything smoke test.

Guarantees the whole package tree imports with only the M0 runtime deps
(pydantic, numpy, jsonschema) — no torch/timm required — and that the public
dataclasses/Protocols instantiate.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "vitreous",
    "vitreous.data",
    "vitreous.models",
    "vitreous.instrument",
    "vitreous.xai",
    "vitreous.xai.eval",
    "vitreous.gaussians",
    "vitreous.graph",
    "vitreous.projections",
    "vitreous.concepts",
    "vitreous.packs",
    "vitreous.packs.manifest",
    "vitreous.packs.writer",
    "vitreous.storage",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_no_torch_required():
    """The M0 surface must import without the ML stack installed."""
    import sys

    # If torch happens to be present that's fine; the guarantee is that our
    # imports above already succeeded without importing it eagerly.
    for name in MODULES:
        mod = importlib.import_module(name)
        assert mod is not None
    # vitreous itself must not have pulled torch in as a side effect of import.
    # (We only assert our modules loaded; we don't forbid torch being available.)
    assert "vitreous" in sys.modules


def test_import_does_not_pull_in_torch():
    """`import vitreous` (+ M1 modules) must not import torch, even if installed.

    Run in a subprocess so a torch already imported by other tests can't mask a
    regression. torch/timm are lazy `[ml]` extras imported only inside
    load_model / Instrumenter.capture.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        import vitreous
        import vitreous.data
        import vitreous.models
        import vitreous.instrument
        assert "torch" not in sys.modules, "torch was imported at import time"
        assert "timm" not in sys.modules, "timm was imported at import time"
        assert "torchvision" not in sys.modules
        # registry populated without torch
        assert "vit_s16" in vitreous.models.list_models()
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_key_symbols_present():
    from vitreous import PackManifest, __version__
    from vitreous.data import DatasetAdapter, register_dataset
    from vitreous.graph import GraphNode, GraphEdge, GraphProvider
    from vitreous.instrument import Trace, Instrumenter
    from vitreous.packs import PackWriter
    from vitreous.storage import StorageAdapter

    assert __version__
    assert PackManifest.__name__ == "PackManifest"
    # Dataclasses instantiate.
    node = GraphNode(id="t0", kind="cls_token", layer=0)
    edge = GraphEdge(source="t0", target="t1", weight=0.5, layer=0)
    assert node.kind == "cls_token"
    assert edge.weight == 0.5
    Trace()  # default-constructs


def test_stubs_raise_not_implemented():
    # build_gaussian_field landed at M3; train_sae is still an M4 stub.
    from vitreous.concepts import train_sae

    with pytest.raises(NotImplementedError):
        train_sae(None)


def test_model_registry_present_without_torch():
    """The model registry is populated at import — no torch needed to inspect it."""
    from vitreous.models import get_model_spec, list_models

    assert "vit_s16" in list_models()
    spec = get_model_spec("vit_s16")
    assert spec.arch == "deit_small_patch16_224"
    assert spec.patch_size == 16
    assert (spec.num_layers, spec.num_heads, spec.embed_dim, spec.num_tokens) == (
        12,
        6,
        384,
        197,
    )
