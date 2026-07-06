"""Manifest round-trip: one fixture, validated two ways in Python.

The TypeScript arm (`packages/schema/tests/pack.typecheck.ts`) checks the same
fixture against the TS types via `tsc`. Together they prove the JSON Schema,
the Pydantic models, and the TS types agree — schema drift is impossible.
"""

from __future__ import annotations

import json

import jsonschema
import pytest

from vitreous.packs import FIXTURE_PATH, PackManifest, load_pack_schema


@pytest.fixture(scope="module")
def fixture_data():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_validates_against_json_schema(fixture_data):
    schema = load_pack_schema()
    # Raises jsonschema.ValidationError on failure.
    jsonschema.validate(instance=fixture_data, schema=schema)


def test_schema_is_valid_draft_2020_12():
    schema = load_pack_schema()
    jsonschema.Draft202012Validator.check_schema(schema)


def test_fixture_parses_with_pydantic(fixture_data):
    manifest = PackManifest.model_validate(fixture_data)
    assert manifest.pack_version == "1.0.0"
    assert manifest.model.arch == "deit_small_patch16_224"
    assert manifest.dataset.num_classes == len(manifest.dataset.class_names)
    assert manifest.image.source in ("gallery", "upload")
    assert 0.0 <= manifest.prediction.confidence <= 1.0
    assert len(manifest.prediction.probabilities) == manifest.dataset.num_classes
    assert "attention.bin" in manifest.assets


def test_pydantic_roundtrip_is_lossless(fixture_data):
    manifest = PackManifest.model_validate(fixture_data)
    dumped = manifest.model_dump(mode="json", exclude_none=True)
    # Re-validate the dumped form against the JSON Schema to close the loop.
    jsonschema.validate(instance=dumped, schema=load_pack_schema())
    # And re-parse to confirm stability.
    assert PackManifest.model_validate(dumped) == manifest


def test_pydantic_rejects_bad_confidence(fixture_data):
    bad = json.loads(json.dumps(fixture_data))
    bad["prediction"]["confidence"] = 1.5
    with pytest.raises(Exception):
        PackManifest.model_validate(bad)


def test_pydantic_rejects_unknown_field(fixture_data):
    bad = json.loads(json.dumps(fixture_data))
    bad["surprise"] = True
    with pytest.raises(Exception):
        PackManifest.model_validate(bad)


def test_jsonschema_rejects_bad_source(fixture_data):
    bad = json.loads(json.dumps(fixture_data))
    bad["image"]["source"] = "webcam"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=load_pack_schema())
