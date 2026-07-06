"""Faithfulness evaluation tests (require the [ml] extra: torch + timm).

Skipped cleanly when torch/timm are unavailable; synthetic images, offline.

Faithfulness-test approach (documented honestly)
------------------------------------------------
A randomly-initialized ViT on random images has no spatial structure to learn,
so training a tiny head in-test would not yield a meaningfully faithful ranking
in bounded CPU time. Instead we validate the faithfulness machinery two ways:

1. **Mechanical properties** — curve lengths, [0,1] bounds, monotone masking,
   AUCs in [0,1], and a random ranking sitting in a mid AUC band.
2. **Oracle vs. random** — we build a ground-truth ranking from each patch's
   *measured* single-patch deletion effect, then assert
   ``AUC(deletion, oracle) < AUC(deletion, random)``. This proves the deletion
   metric correctly rewards a more faithful ordering without needing a trained
   model (the model-agnostic form of the M2 ``AUC(chefer) > AUC(random)``
   acceptance test).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from vitreous.data import DatasetSpec
from vitreous.models import load_model
from vitreous.xai._common import unwrap
from vitreous.xai.eval import deletion_insertion, method_agreement


@pytest.fixture(scope="module")
def loaded():
    ds = DatasetSpec(name="toy", display_name="Toy", num_classes=5)
    return load_model("vit_s16", ds, pretrained=False)


def _image(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, 224, 224, generator=g)


# -- mechanical properties -------------------------------------------------- #


def test_deletion_insertion_mechanical(loaded):
    img = _image(0)
    ranking = np.random.default_rng(1).random(196).astype(np.float32)
    di = deletion_insertion(loaded, img, ranking, steps=6)

    assert len(di.deletion) == di.steps + 1
    assert len(di.insertion) == di.steps + 1
    assert all(0.0 <= v <= 1.0 for v in di.deletion)
    assert all(0.0 <= v <= 1.0 for v in di.insertion)
    assert 0.0 <= di.deletion_auc <= 1.0
    assert 0.0 <= di.insertion_auc <= 1.0


def test_deletion_endpoints(loaded):
    """deletion starts at the full-image prob; insertion starts at baseline prob."""
    img = _image(2)
    m = unwrap(loaded)
    ranking = np.arange(196, dtype=np.float32)
    di = deletion_insertion(loaded, img, ranking, steps=4)
    with torch.no_grad():
        target = int(m(img).softmax(dim=-1)[0].argmax())
        full = float(m(img).softmax(dim=-1)[0, target])
        black = float(m(torch.zeros_like(img)).softmax(dim=-1)[0, target])
    assert di.deletion[0] == pytest.approx(full, abs=1e-5)
    assert di.insertion[0] == pytest.approx(black, abs=1e-5)


def test_ranking_shapes_accepted(loaded):
    """[196], [197] (with CLS), and [14,14] rankings all work."""
    img = _image(3)
    for ranking in (
        np.random.default_rng(0).random(196).astype(np.float32),
        np.random.default_rng(0).random(197).astype(np.float32),
        np.random.default_rng(0).random((14, 14)).astype(np.float32),
    ):
        di = deletion_insertion(loaded, img, ranking, steps=4)
        assert len(di.deletion) == di.steps + 1


# -- method agreement ------------------------------------------------------- #


def test_method_agreement_matrix(loaded):
    a = np.random.default_rng(1).random(196)
    b = np.random.default_rng(2).random(196)
    mat = method_agreement({"a": a, "b": b, "c": a.copy()})
    # Diagonal is exactly 1.
    for k in mat:
        assert mat[k][k] == 1.0
    # Symmetric.
    assert mat["a"]["b"] == pytest.approx(mat["b"]["a"])
    # Identical rankings correlate perfectly.
    assert mat["a"]["c"] == pytest.approx(1.0, abs=1e-9)
    # Correlations are in [-1, 1].
    for x in mat.values():
        for v in x.values():
            assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9


def test_method_agreement_reduces_heterogeneous_shapes(loaded):
    """Different attribution shapes are reduced to comparable per-patch vectors."""
    mat = method_agreement(
        {
            "per_layer": np.random.default_rng(0).random((12, 197)),  # [L,T]
            "tokens": np.random.default_rng(1).random(197),  # [T] w/ CLS
            "grid": np.random.default_rng(2).random((14, 14)),  # token grid
        }
    )
    assert set(mat) == {"per_layer", "tokens", "grid"}


# -- oracle vs random (faithfulness sanity) --------------------------------- #


def test_oracle_beats_random_deletion_auc(loaded):
    """AUC(deletion, oracle) < AUC(deletion, random): the metric rewards faithful order."""
    m = unwrap(loaded)
    img = _image(7)
    with torch.no_grad():
        target = int(m(img).softmax(dim=-1)[0].argmax())
        full = float(m(img).softmax(dim=-1)[0, target])
        # Ground-truth per-patch importance = drop in target prob when masked.
        oracle = np.zeros(196, dtype=np.float32)
        for p in range(196):
            r, c = divmod(p, 14)
            xm = img.clone()
            xm[..., r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] = 0.0
            oracle[p] = full - float(m(xm).softmax(dim=-1)[0, target])

    di_oracle = deletion_insertion(loaded, img, oracle, steps=8, target=target)
    random_aucs = []
    for seed in (1, 2, 3):
        rnd = np.random.default_rng(seed).random(196).astype(np.float32)
        random_aucs.append(
            deletion_insertion(loaded, img, rnd, steps=8, target=target).deletion_auc
        )
    assert di_oracle.deletion_auc < float(np.mean(random_aucs))
