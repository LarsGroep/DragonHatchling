"""Pydantic v2 models mirroring ``packages/schema/schema/pack.schema.json``.

This is the Python arm of the single-source-of-truth trio
(JSON Schema -> Pydantic -> TypeScript). The models here MUST stay
structurally identical to the JSON Schema; the round-trip test in
``packages/core/tests/test_manifest_roundtrip.py`` validates the shared
fixture against both the JSON Schema (via ``jsonschema``) and these models,
so drift is caught in CI.

The pack format is frozen at milestone M2 (§5 of ARCHITECTURE.md).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ``model_`` is a Pydantic-protected namespace; the pack manifest legitimately
# has a ``model`` field, so we disable the warning at the module level via each
# model's config below.

AssetDtype = Literal[
    "uint8",
    "int8",
    "uint16",
    "int16",
    "int32",
    "float16",
    "float32",
    "float64",
    "json",
    "webp",
    "png",
]

AssetEncoding = Literal[
    "raw",
    "zstd",
    "gzip",
    "per_row_uint8",
    "json",
    "webp",
    "png",
]

ImageSource = Literal["gallery", "upload"]


class _Strict(BaseModel):
    """Base config: forbid unknown keys so schema drift surfaces as errors."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelInfo(_Strict):
    arch: str = Field(min_length=1)
    hf_repo: str = Field(min_length=1)
    num_layers: Optional[int] = Field(default=None, ge=1)
    num_heads: Optional[int] = Field(default=None, ge=1)
    num_tokens: Optional[int] = Field(default=None, ge=1)
    embed_dim: Optional[int] = Field(default=None, ge=1)


class DatasetInfo(_Strict):
    name: str = Field(min_length=1)
    display_name: Optional[str] = None
    num_classes: int = Field(ge=1)
    class_names: List[str] = Field(min_length=1)


class ImageMeta(_Strict):
    id: str = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    source: ImageSource


class Prediction(_Strict):
    label: str
    class_index: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: List[float] = Field(min_length=1)


class AssetEntry(_Strict):
    dtype: AssetDtype
    shape: List[int] = Field(default_factory=list)
    encoding: AssetEncoding
    bytes: int = Field(ge=0)
    checksum: Optional[str] = None


class PackManifest(_Strict):
    """Top-level Explanation Pack manifest (``manifest.json``)."""

    pack_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    model: ModelInfo
    dataset: DatasetInfo
    image: ImageMeta
    prediction: Prediction
    assets: Dict[str, AssetEntry] = Field(min_length=1)
    timings: Dict[str, float] = Field(default_factory=dict)


__all__ = [
    "AssetDtype",
    "AssetEncoding",
    "ImageSource",
    "ModelInfo",
    "DatasetInfo",
    "ImageMeta",
    "Prediction",
    "AssetEntry",
    "PackManifest",
]
