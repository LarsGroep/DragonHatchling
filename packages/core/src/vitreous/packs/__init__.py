"""Explanation Pack manifest models, schema access, and the pack writer.

The manifest models (§5) are real and importable at M0; ``PackWriter`` is a
stub whose logic lands at M2 (the pack format freeze).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .manifest import (
    AssetDtype,
    AssetEncoding,
    AssetEntry,
    DatasetInfo,
    ImageMeta,
    ImageSource,
    ModelInfo,
    PackManifest,
    Prediction,
    QuantInfo,
)
from .writer import PACK_VERSION, PackReader, PackWriter, build_pack

# Schema resolution. The single source of truth is ``packages/schema`` in the
# monorepo, but that directory is NOT installed when ``packages/core`` is pip-
# installed on its own (e.g. on Kaggle). So a copy is vendored next to this
# module and shipped as package data; we prefer the vendored copy and fall back
# to the monorepo source for in-repo development. A drift test keeps them equal.
_VENDORED_SCHEMA = Path(__file__).resolve().parent / "pack.schema.json"
_MONOREPO_SCHEMA = (
    Path(__file__).resolve().parents[4] / "schema" / "schema" / "pack.schema.json"
)
SCHEMA_PATH = _VENDORED_SCHEMA if _VENDORED_SCHEMA.exists() else _MONOREPO_SCHEMA
FIXTURE_PATH = (
    Path(__file__).resolve().parents[4] / "schema" / "fixtures" / "manifest.fixture.json"
)


def load_pack_schema() -> Dict[str, Any]:
    """Return the pack JSON Schema (draft 2020-12) as a dict.

    Loads the vendored copy shipped with the package (works after a standalone
    pip install), falling back to the ``packages/schema`` source of truth in a
    monorepo checkout. The Pydantic models mirror it and are kept honest by the
    round-trip + drift tests.
    """
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


__all__ = [
    "AssetDtype",
    "AssetEncoding",
    "AssetEntry",
    "DatasetInfo",
    "ImageMeta",
    "ImageSource",
    "ModelInfo",
    "PackManifest",
    "Prediction",
    "QuantInfo",
    "PackWriter",
    "PackReader",
    "build_pack",
    "PACK_VERSION",
    "SCHEMA_PATH",
    "FIXTURE_PATH",
    "load_pack_schema",
]
