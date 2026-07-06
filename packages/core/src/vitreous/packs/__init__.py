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

# Resolve the JSON Schema / fixture from the sibling ``packages/schema`` dir.
# packages/core/src/vitreous/packs/__init__.py -> parents[4] == repo/packages
_PACKAGES_DIR = Path(__file__).resolve().parents[4]
SCHEMA_PATH = _PACKAGES_DIR / "schema" / "schema" / "pack.schema.json"
FIXTURE_PATH = _PACKAGES_DIR / "schema" / "fixtures" / "manifest.fixture.json"


def load_pack_schema() -> Dict[str, Any]:
    """Return the pack JSON Schema (draft 2020-12) as a dict.

    The schema in ``packages/schema`` is the single source of truth; the
    Pydantic models mirror it and are kept honest by the round-trip test.
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
