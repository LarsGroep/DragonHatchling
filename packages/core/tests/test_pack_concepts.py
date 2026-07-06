"""build_pack concept integration (§9) — concepts.json is additive.

With a ConceptProvider, build_pack emits concepts.json (per-token top-k feature
ids + activations + dictionary ref) and PackReader reads it back; without one,
concepts.json is absent and the pack is still schema-valid and pack_version
stays 1.0.0. Requires the [ml] extra (torch/timm), skipped cleanly otherwise.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from vitreous.concepts import ConceptPackSpec, KMeansConceptProvider
from vitreous.data import DatasetSpec
from vitreous.instrument import Instrumenter
from vitreous.models import load_model
from vitreous.packs import PACK_VERSION, PackReader, build_pack, load_pack_schema


@pytest.fixture(scope="module")
def loaded():
    ds = DatasetSpec(name="toy", display_name="Toy", num_classes=4,
                     class_names=["a", "b", "c", "d"])
    return load_model("vit_s16", ds, pretrained=False), ds


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture(scope="module")
def concept_spec(loaded, image):
    lm, _ds = loaded
    trace = Instrumenter(lm.module).capture(image)
    toks9 = trace.tokens[9].numpy()  # [197, 384]
    provider = KMeansConceptProvider.fit(toks9, n_clusters=16, seed=0, layer=9)
    return ConceptPackSpec(provider, dictionary_id="toy_vit_s16_L9", layer=9, topk=8)


def test_pack_with_concepts_emits_and_reads(loaded, image, concept_spec, tmp_path):
    lm, ds = loaded
    out = tmp_path / "with_concepts"
    build_pack(lm, image, {"id": "c0", "source": "gallery"}, ds, out,
               ig_steps=3, faithfulness_steps=4, faithfulness_methods=("chefer",),
               concepts=concept_spec)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["pack_version"] == PACK_VERSION == "1.0.0"
    assert "concepts.json" in manifest["assets"]
    # schema still validates with the additive asset present
    import jsonschema
    jsonschema.validate(instance=manifest, schema=load_pack_schema())

    rd = PackReader(out)
    assert rd.has_concepts()
    c = rd.read_concepts()
    assert c["layer"] == 9
    assert c["dictionary_id"] == "toy_vit_s16_L9"
    assert c["provider_kind"] == "kmeans"
    assert c["n_concepts"] == 16
    assert c["num_tokens"] == 197
    assert len(c["feature_ids"]) == 197
    assert len(c["feature_ids"][0]) == 8
    assert len(c["activations"]) == 197
    ids = np.array(c["feature_ids"])
    assert ids.min() >= 0 and ids.max() < 16
    # asset meta advertises the dictionary ref (additive self-description)
    assert manifest["assets"]["concepts.json"]["meta"]["dictionary_id"] == "toy_vit_s16_L9"


def test_pack_without_concepts_omits_asset(loaded, image, tmp_path):
    lm, ds = loaded
    out = tmp_path / "no_concepts"
    build_pack(lm, image, {"id": "n0", "source": "gallery"}, ds, out,
               ig_steps=3, faithfulness_steps=4, faithfulness_methods=("chefer",))
    manifest = json.loads((out / "manifest.json").read_text())
    assert "concepts.json" not in manifest["assets"]
    assert not (out / "concepts.json").exists()
    rd = PackReader(out)
    assert rd.has_concepts() is False
    assert manifest["pack_version"] == "1.0.0"
