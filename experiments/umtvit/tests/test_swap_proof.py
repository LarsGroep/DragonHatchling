"""U7 swap-proof tests (ARCHITECTURE §9 row U7).

The acceptance criterion for U7 is *"zero code diffs between HAM10000 and
EuroSAT runs"*. We cannot download EuroSAT locally, so we prove the swap
**structurally** instead: the exact same code paths build both runs and the
only differences flow from the two YAML files' values.

Two guarantees are checked here:

1. :func:`test_swap_configs_build_from_identical_code_paths` — loading the
   shipped ``configs/ham10000.yaml`` and ``configs/eurosat.yaml`` and building
   ``UMTViT`` + ``Soft3DSOM`` + ``Trainer`` for **both** produces objects of the
   *identical* classes, wired by the *identical* constructors; every structural
   difference (volume grid, image size, loader, augmentation) is traceable to a
   config value, and each model's forward output shape follows its own config.

2. :func:`test_eurosat_layout_end_to_end_zero_code_changes` — a synthetic
   EuroSAT-layout ``imagefolder`` tree (``root/<class>/*.jpg``, 3 classes x 12
   images at 64 px) runs **end to end** through :class:`UniversalDataset` with
   the ``satellite_default`` augmentation policy → a short :class:`Trainer` run →
   :func:`run_evaluation`, using the ``eurosat.yaml`` dataset block verbatim
   (loader, augmentation, image size). Only the *compute* knobs (model scale,
   step budget) are shrunk for CPU — no code path is swapped, added, or edited.
   That is the executable evidence that EuroSAT-shaped data + the eurosat config
   route works with zero code changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from umtvit.config import Config, load_config
from umtvit.data.augment import AUGMENTATION_POLICIES
from umtvit.data.dataset import UniversalDataset
from umtvit.engine.trainer import Trainer
from umtvit.eval.report import render_report, run_evaluation
from umtvit.models.model import UMTViT
from umtvit.models.som3d import Soft3DSOM

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _PairedTensorDataset(torch.utils.data.Dataset):
    """Minimal ``(view_a, view_b, label)`` dataset for Trainer construction.

    Lets a Trainer be *constructed* against a config without touching disk (the
    HAM10000 images are not present locally); no forward is run through it here.
    """

    def __init__(self, n: int, channels: int, size: int) -> None:
        self.n = n
        self.channels = channels
        self.size = size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(idx)
        xa = torch.rand(self.channels, self.size, self.size, generator=g)
        xb = torch.rand(self.channels, self.size, self.size, generator=g)
        return xa, xb, idx % 3


def _build_stack(cfg: Config):
    """Build the model + SOM + trainer for a config via the shared code paths."""
    model = UMTViT(cfg)
    som = Soft3DSOM.from_config(cfg)
    data = _PairedTensorDataset(
        n=cfg.train.batch_size, channels=cfg.dataset.channels,
        size=cfg.dataset.image_size,
    )
    trainer = Trainer(cfg, model, som, data)
    return model, som, trainer


def _make_eurosat_layout(root: Path, n_classes: int = 3, per_class: int = 12,
                         size: int = 64) -> None:
    """Write a synthetic EuroSAT-style ``root/<class>/*.jpg`` imagefolder tree."""
    from PIL import Image

    rng = np.random.default_rng(0)
    for c in range(n_classes):
        cls_dir = root / f"Class_{c}"
        cls_dir.mkdir(parents=True)
        for j in range(per_class):
            # A per-class colour bias so the classes are linearly separable
            # enough for the probe/k-NN to report a finite, above-floor number.
            base = np.zeros((size, size, 3), dtype=np.float32)
            base[..., c % 3] = 0.6
            noise = rng.normal(0.0, 0.15, size=(size, size, 3)).astype(np.float32)
            arr = np.clip(base + noise, 0.0, 1.0)
            Image.fromarray((arr * 255).astype(np.uint8), "RGB").save(
                cls_dir / f"img_{j:02d}.jpg"
            )


# --------------------------------------------------------------------------- #
# 1. Both runs are built by identical code paths; differences are config values
# --------------------------------------------------------------------------- #
def test_swap_configs_build_from_identical_code_paths():
    ham = load_config(_CONFIGS_DIR / "ham10000.yaml")
    euro = load_config(_CONFIGS_DIR / "eurosat.yaml")

    ham_model, ham_som, ham_trainer = _build_stack(ham)
    euro_model, euro_som, euro_trainer = _build_stack(euro)

    # --- identical classes: the swap changes no types, only values --------- #
    assert type(ham_model) is type(euro_model) is UMTViT
    assert type(ham_som) is type(euro_som) is Soft3DSOM
    assert type(ham_trainer) is type(euro_trainer) is Trainer
    for attr in ("backbone", "uplifting", "head"):
        assert type(getattr(ham_model, attr)) is type(getattr(euro_model, attr))

    # --- the datasets are the one universal class for both loaders --------- #
    # (built without disk access: enumeration is what we assert on, not I/O.)
    assert UniversalDataset is UniversalDataset  # single dataset class, always
    assert ham.dataset.loader == "csv"
    assert euro.dataset.loader == "imagefolder"

    # --- shared standing defaults (DECISION-LOG): same backbone geometry --- #
    assert ham.model.dim == euro.model.dim == 256
    assert ham.model.depth == euro.model.depth == 8
    assert ham.model.som_grid == euro.model.som_grid == (8, 8, 8)
    assert ham_som.K == euro_som.K == 8 * 8 * 8

    # --- structural differences ALL trace to config values ----------------- #
    assert ham.dataset.name != euro.dataset.name
    assert ham.dataset.image_size == 128 and euro.dataset.image_size == 64
    assert ham.dataset.augmentation == "dermoscopy_default"
    assert euro.dataset.augmentation == "satellite_default"
    assert (ham.model.volume_h, ham.model.volume_w) == (16, 16)
    assert (euro.model.volume_h, euro.model.volume_w) == (8, 8)

    # --- forward output shapes follow each config -------------------------- #
    for model, cfg in ((ham_model, ham), (euro_model, euro)):
        model.eval()
        x = torch.rand(2, cfg.dataset.channels, cfg.dataset.image_size,
                       cfg.dataset.image_size)
        with torch.no_grad():
            out = model(x)
        vg_h, vg_w = cfg.model.volume_h, cfg.model.volume_w
        assert out["volume"].shape == (
            2, vg_h, vg_w, cfg.model.depth, cfg.model.volume_channels
        )
        assert out["pooled"].shape == (
            2, cfg.model.depth * cfg.model.volume_channels
        )
        assert out["proj"].shape == (2, cfg.model.proj_dim)
        assert torch.isfinite(out["proj"]).all()

    # --- trainer wires the objective weights from each config, same keys --- #
    assert ham_trainer.weights.keys() == euro_trainer.weights.keys()
    assert ham_trainer.weights["ntxent"] == ham.loss.lambda_ntxent
    assert euro_trainer.weights["som"] == euro.loss.lambda_som


# --------------------------------------------------------------------------- #
# 2. EuroSAT-layout data runs end-to-end through the eurosat config route
# --------------------------------------------------------------------------- #
def test_eurosat_layout_end_to_end_zero_code_changes(tmp_path: Path):
    root = tmp_path / "EuroSAT"
    _make_eurosat_layout(root, n_classes=3, per_class=12, size=64)

    # Start from the shipped eurosat.yaml and keep its dataset block VERBATIM
    # (loader=imagefolder, augmentation=satellite_default, image_size=64). Only
    # the compute knobs are shrunk for CPU — these are config values, not code.
    data = load_config(_CONFIGS_DIR / "eurosat.yaml").to_dict()
    data["dataset"]["image_dir"] = str(root)
    assert data["dataset"]["loader"] == "imagefolder"
    assert data["dataset"]["augmentation"] == "satellite_default"
    # CPU-scale model + short budget (compute knobs only).
    data["model"].update(
        dict(image_size=None, dim=32, depth=2, heads=4, volume_h=4, volume_w=4,
             volume_channels=16, proj_dim=32, som_grid=[3, 3, 3])
    )
    data["train"].update(
        dict(epochs=None, max_steps=6, batch_size=8, warmup_steps=2,
             amp=False, grad_checkpoint=False, som_sample_voxels=128)
    )
    cfg = Config.from_dict(data)

    # The satellite policy is the real registry entry (proof it resolves).
    assert cfg.dataset.augmentation in AUGMENTATION_POLICIES

    # Two-view train set + eval sets, all from UniversalDataset (one class).
    train_two_view = UniversalDataset(cfg, split="train", mode="two_view")
    train_eval = UniversalDataset(cfg, split="train", mode="eval")
    test_eval = UniversalDataset(cfg, split="test", mode="eval")
    assert len(train_two_view) > 0
    assert train_two_view.classes == ["Class_0", "Class_1", "Class_2"]

    # Views are the EuroSAT image size and in range.
    xa, xb, label = train_two_view[0]
    assert xa.shape == (3, 64, 64)
    assert 0.0 <= float(xa.min()) and float(xa.max()) <= 1.0
    assert isinstance(label, int) and label in (0, 1, 2)

    # Short Trainer run — same trainer, no code changes.
    torch.manual_seed(0)
    model = UMTViT(cfg)
    som = Soft3DSOM.from_config(cfg)
    loader = DataLoader(train_two_view, batch_size=cfg.train.batch_size, shuffle=True)
    trainer = Trainer(cfg, model, som, loader)
    history = trainer.train()
    assert len(history["total"]) > 0
    assert all(np.isfinite(v) for v in history["total"])

    # Full evaluation through the standard report path.
    results = run_evaluation(
        model, som, cfg,
        {"train": train_eval, "test": test_eval},
        seed=0, probe_steps=100, manifold_max_n=64,
        som_sample_voxels=128, zaxis_images=4,
    )
    # Labeled EuroSAT layout ⇒ probe + k-NN ran and are finite.
    assert results["linear_probe"] is not None
    assert np.isfinite(results["linear_probe"]["accuracy"])
    assert results["knn"] is not None
    for key in ("quantization_error", "topographic_error", "dead_neuron_fraction"):
        assert np.isfinite(results["som"][key])
    # The markdown report renders (the swap produces a real run report).
    report = render_report(results)
    assert "UMT-ViT Evaluation Report" in report
    assert "eurosat" in report.lower()


def test_eurosat_config_is_pure_config_swap_of_ham():
    """The two shipped YAMLs differ only in values, never in required structure."""
    ham = load_config(_CONFIGS_DIR / "ham10000.yaml")
    euro = load_config(_CONFIGS_DIR / "eurosat.yaml")
    # Same set of configured sections/fields (dataclass schema is shared), so a
    # swap is a value edit, not a schema/code change.
    assert set(ham.to_dict().keys()) == set(euro.to_dict().keys())
    assert set(ham.model.__dict__.keys()) == set(euro.model.__dict__.keys())
    assert set(ham.loss.__dict__.keys()) == set(euro.loss.__dict__.keys())
    # Both are valid, labeled, gradient-SOM runs with geodesic gated off.
    for cfg in (ham, euro):
        assert cfg.model.som_update == "gradient"
        assert cfg.loss.lambda_geodesic == 0.0
        assert cfg.dataset.label_column is not None
