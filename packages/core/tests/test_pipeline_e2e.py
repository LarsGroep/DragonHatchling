"""End-to-end local pipeline (§16 M4 dry-run) — offline, no network.

A synthetic folder-per-class dataset → adapter → ViT-S/16 → build_pack →
LocalStorageAdapter.put_pack, proving the whole batch pipeline runs locally and
every declared pack asset lands in storage with a resolvable public URL. This is
the dry-run the M4 acceptance test calls for, wired through the real public API
(minus the actual Supabase network). Requires the [ml] extra.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from PIL import Image

from vitreous.concepts import ConceptPackSpec, KMeansConceptProvider
from vitreous.data import get_dataset, make_synthetic_dataset
from vitreous.instrument import Instrumenter
from vitreous.models import load_model
from vitreous.packs import PackReader, build_pack
from vitreous.storage import LocalStorageAdapter


def test_shapes_dataset_end_to_end_local(tmp_path):
    # 1) synthetic dataset in the folder-per-class layout the adapters accept.
    root = tmp_path / "eurosat_like"
    make_synthetic_dataset(str(root), "folder_per_class", num_classes=3,
                           per_class=4, image_size=224)
    adapter = get_dataset("imagefolder")()
    spec = adapter.spec
    samples = list(adapter.load(str(root)))
    assert samples, "adapter yielded no samples"

    # 2) model + eval transform (pretrained=False — never download in CI).
    lm = load_model("vit_s16", spec, pretrained=False, num_classes=spec.num_classes)
    eval_t = adapter.preprocess()

    # 3) a concept provider from a captured trace (k-means, torch-free centroids).
    s0 = samples[0]
    img0 = Image.open(s0.image).convert("RGB")
    x0 = eval_t(img0).unsqueeze(0)
    trace0 = Instrumenter(lm.module).capture(x0)
    provider = KMeansConceptProvider.fit(trace0.tokens[9].numpy(), n_clusters=12, seed=0, layer=9)
    concept_spec = ConceptPackSpec(provider, dictionary_id="synthetic_L9", layer=9, topk=8)

    # 4) build a pack and upload it via the local storage adapter.
    storage = LocalStorageAdapter(tmp_path / "store")
    image_id = "syn_0"
    pack_dir = tmp_path / "packs" / image_id
    build_pack(lm, x0, {"id": image_id, "source": "gallery"}, spec, pack_dir,
               ig_steps=3, faithfulness_steps=4, faithfulness_methods=("chefer",),
               concepts=concept_spec)

    prefix = f"packs/{spec.name}/{image_id}"
    urls = storage.put_pack(pack_dir, prefix)

    # 5) every declared manifest asset is uploaded and resolvable.
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    for name in manifest["assets"]:
        assert name in urls, f"{name} not uploaded"
        assert storage.exists(f"{prefix}/{name}"), f"{name} missing in storage"
        assert storage.get_url(f"{prefix}/{name}").startswith("file://")
    assert "concepts.json" in manifest["assets"]

    # 6) the uploaded pack reads back correctly from the storage root.
    stored_pack = storage.root / prefix
    rd = PackReader(stored_pack)
    assert rd.manifest.pack_version == "1.0.0"
    assert rd.has_concepts()
    assert rd.read_concepts()["dictionary_id"] == "synthetic_L9"
    assert rd.read_tokens().shape == (13, 197, 384)
