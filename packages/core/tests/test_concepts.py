"""Concept tier (§9) — SAE training, providers, dictionary, quality gate.

Offline, synthetic. The SAE tests require torch (skipped cleanly if absent);
the dictionary / quality-gate / interface tests are torch-free (k-means uses
scikit-learn, which is an M0-env dep).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vitreous.concepts import (
    ConceptDictionary,
    ConceptFeature,
    ConceptPackSpec,
    ExemplarRef,
    KMeansConceptProvider,
    SAEConceptProvider,
    SAEStats,
    build_concept_dictionary,
    layer_token_activations,
    quality_gate,
    train_sae,
)

D = 32  # small activation dim for fast tests


def _clustered(n_per=120, n_clusters=6, d=D, noise=0.08, seed=0):
    """Well-separated clusters, one class per cluster — a 'healthy' concept space."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)) * 3.0
    labels = np.repeat(np.arange(n_clusters), n_per)
    X = centers[labels] + noise * rng.standard_normal((len(labels), d))
    return X.astype(np.float32), labels.astype(np.int64), n_clusters


def _refs(labels):
    return [
        ExemplarRef(image_id=f"img_{i // 197}", token_idx=int(i % 197),
                    activation=0.0, class_label=int(c))
        for i, c in enumerate(labels)
    ]


# --------------------------------------------------------------------------- #
# SAE (torch)
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch")


def test_sae_reduces_reconstruction_loss():
    X, _labels, _ = _clustered(seed=1)
    _sae, stats = train_sae(X, n_features=64, k=8, epochs=80, seed=0)
    assert stats.recon_loss < stats.initial_recon_loss
    assert stats.n_samples == X.shape[0]
    assert 0 <= stats.dead_feature_count <= stats.n_features


def test_topk_activation_exactly_k_nonzeros():
    X, _labels, _ = _clustered(seed=2)
    k = 8
    sae, _stats = train_sae(X, n_features=64, k=k, epochs=10, seed=0)
    feats = sae.encode(torch.as_tensor(X, dtype=torch.float32))
    nnz = (feats != 0).sum(dim=1)
    assert int(nnz.min()) == int(nnz.max()) == k
    # Reported L0 equals k too.
    _sae2, stats = train_sae(X, n_features=64, k=k, epochs=10, seed=0)
    assert stats.l0 == pytest.approx(float(k))


def test_tie_decoder_option():
    X, _labels, _ = _clustered(seed=3)
    sae, _stats = train_sae(X, n_features=48, k=6, epochs=8, seed=0, tie_decoder=True)
    assert sae.tie_decoder is True
    assert sae.decoder_weight().shape == (X.shape[1], 48)


def test_sae_provider_encode_and_top_features():
    X, labels, K = _clustered(seed=4)
    sae, _stats = train_sae(X, n_features=64, k=8, epochs=40, seed=0)
    prov = SAEConceptProvider(sae, layer=9)
    assert prov.kind == "sae"
    assert prov.n_concepts() == 64
    feats = prov.encode(X)
    assert feats.shape == (X.shape[0], 64)
    assert (feats != 0).sum(axis=1).max() <= 8

    trace = SimpleNamespace(tokens=np.random.default_rng(0).standard_normal((13, 197, D)).astype(np.float32))
    ids, acts = prov.top_features_per_token(trace, layer=9, topk=8)
    assert ids.shape == (197, 8) and acts.shape == (197, 8)
    assert ids.min() >= 0 and ids.max() < 64


def test_quality_gate_passes_healthy_sae():
    # Directly construct a healthy dictionary + stats (unit-tests the gate logic).
    feats = []
    for c in range(20):
        # each feature's exemplars are one pure class, and each feature has a
        # distinct strongest exemplar (no duplication).
        ex = [ExemplarRef(f"img_{c}_{j}", j, 1.0 - 0.01 * j, class_label=c % 6) for j in range(8)]
        feats.append(ConceptFeature(id=c, exemplars=ex, firing_rates=[0.0] * 6, coherence=1.0))
    d = ConceptDictionary(model="m", dataset="d", layer=9, provider_kind="sae",
                          n_concepts=20, num_classes=6, features=feats)
    stats = SAEStats(n_features=20, k=8, n_samples=1000, epochs=50,
                     dead_feature_count=2, dead_feature_rate=0.10, l0=8.0,
                     recon_loss=0.01, initial_recon_loss=0.5, duplicate_feature_rate=0.05)
    report = quality_gate(stats, d)
    assert report.use_sae is True
    assert report.reasons == []
    assert report.dead_feature_rate == pytest.approx(0.10)
    assert report.exemplar_coherence == pytest.approx(1.0)


def test_quality_gate_flags_degenerate_sae():
    # Mostly-dead SAE + every feature shares one strongest exemplar + mixed classes.
    feats = []
    for c in range(20):
        ex = [ExemplarRef("SAME_IMG", 0, 1.0, class_label=(j % 6)) for j in range(8)]
        feats.append(ConceptFeature(id=c, exemplars=ex, firing_rates=[0.0] * 6, coherence=1.0 / 6))
    d = ConceptDictionary(model="m", dataset="d", layer=9, provider_kind="sae",
                          n_concepts=20, num_classes=6, features=feats)
    stats = SAEStats(n_features=20, k=8, n_samples=1000, epochs=50,
                     dead_feature_count=18, dead_feature_rate=0.90, l0=8.0,
                     recon_loss=0.4, initial_recon_loss=0.5, duplicate_feature_rate=0.8)
    report = quality_gate(stats, d)
    assert report.use_sae is False
    # all three checks should fire
    assert len(report.reasons) == 3


# --------------------------------------------------------------------------- #
# dictionary building
# --------------------------------------------------------------------------- #


def test_build_dictionary_shape_and_exemplars():
    X, labels, K = _clustered(seed=5)
    prov = KMeansConceptProvider.fit(X, n_clusters=K, seed=0, layer=9)
    d = build_concept_dictionary(prov, X, _refs(labels), num_classes=K,
                                 model="m", dataset="d", layer=9, topk_exemplars=16)
    assert d.n_concepts == K
    assert len(d.features) == K
    assert d.num_classes == K
    for f in d.features:
        assert len(f.firing_rates) == K
        assert len(f.exemplars) <= 16
        for e in f.exemplars:
            assert 0 <= e.token_idx < 197
    # Clusters map cleanly to classes → high coherence.
    coh = np.mean([f.coherence for f in d.features])
    assert coh > 0.9
    # JSON round-trips (concepts.json + dictionary artifacts are JSON).
    import json
    assert json.loads(json.dumps(d.to_json()))["n_concepts"] == K


def test_label_hook_is_pluggable_and_defaults_none():
    X, labels, K = _clustered(seed=6)
    prov = KMeansConceptProvider.fit(X, n_clusters=K, seed=0)
    d0 = build_concept_dictionary(prov, X, _refs(labels), num_classes=K)
    assert all(f.suggested_label is None for f in d0.features)  # no CLIP dep

    seen = []
    def hook(fid, exemplars):
        seen.append(fid)
        return f"concept_{fid}"
    d1 = build_concept_dictionary(prov, X, _refs(labels), num_classes=K, label_hook=hook)
    assert all(f.suggested_label == f"concept_{f.id}" for f in d1.features)
    assert len(seen) == K


# --------------------------------------------------------------------------- #
# provider interface parity + pack spec
# --------------------------------------------------------------------------- #


def test_kmeans_provider_interface_parity():
    X, labels, K = _clustered(seed=7)
    km = KMeansConceptProvider.fit(X, n_clusters=K, seed=0, layer=9)
    sae, _stats = train_sae(X, n_features=K, k=4, epochs=10, seed=0)
    sae_prov = SAEConceptProvider(sae, layer=9)

    trace = SimpleNamespace(tokens=np.random.default_rng(0).standard_normal((13, 197, D)).astype(np.float32))
    for prov in (km, sae_prov):
        assert hasattr(prov, "kind")
        assert isinstance(prov.n_concepts(), int)
        f = prov.encode(X)
        assert f.shape[0] == X.shape[0]
        ids, acts = prov.top_features_per_token(trace, layer=9, topk=5)
        assert ids.shape == (197, 5) and acts.shape == (197, 5)
    # k-means encode rows are a proper distribution (softmax similarity).
    assert km.encode(X).sum(axis=1) == pytest.approx(np.ones(X.shape[0]), abs=1e-4)


def test_concept_pack_spec_build_asset():
    X, labels, K = _clustered(seed=8)
    prov = KMeansConceptProvider.fit(X, n_clusters=32, seed=0, layer=9)
    spec = ConceptPackSpec(prov, dictionary_id="m_d_L9", layer=9, topk=8)
    trace = SimpleNamespace(tokens=np.random.default_rng(1).standard_normal((13, 197, D)).astype(np.float32))
    asset = spec.build_asset(trace)
    assert asset["dictionary_id"] == "m_d_L9"
    assert asset["layer"] == 9
    assert asset["num_tokens"] == 197
    assert asset["n_concepts"] == 32
    assert len(asset["feature_ids"]) == 197
    assert len(asset["feature_ids"][0]) == 8  # topk fits within 32 concepts
    assert len(asset["activations"]) == 197
    assert asset["provider_kind"] == "kmeans"
    import json
    json.dumps(asset)  # must be JSON-serializable


def test_layer_token_activations_selects_layer():
    tokens = np.arange(13 * 197 * D, dtype=np.float32).reshape(13, 197, D)
    trace = SimpleNamespace(tokens=tokens)
    a9 = layer_token_activations(trace, 9)
    assert a9.shape == (197, D)
    assert np.array_equal(a9, tokens[9])
    with pytest.raises(IndexError):
        layer_token_activations(trace, 99)
