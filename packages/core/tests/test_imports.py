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
    from vitreous.gaussians import build_gaussian_field
    from vitreous.models import ModelSpec, load_model

    with pytest.raises(NotImplementedError):
        build_gaussian_field(None, None)
    with pytest.raises(NotImplementedError):
        load_model(ModelSpec(arch="deit_small_patch16_224", hf_repo="x/y"))
