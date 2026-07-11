"""U3 tests: spatial uplifting, projection head, and the 3-D SOM (ARCHITECTURE §3.4-§3.5).

All CPU-only and fast: tiny configs, synthetic voxel data, short SOM runs.
Covers

- latent-volume / pooled / proj shapes, from small configs and every shipped
  ``configs/*.yaml`` (constructed without any dataset on disk);
- SOM convergence on 3 well-separated Gaussian blobs: quantization error falls
  over ~200 steps in *both* the gradient and kohonen_ema update modes, and the
  trained topographic error beats a random-map baseline;
- ``data_init`` spreads neurons (lower dead fraction than random init);
- ``revive`` re-seeds exactly the zero-hit neurons;
- σ-schedule resolution (null ⇒ grid-derived, explicit honoured);
- gradient reaches the backbone through the volume via ``L_som``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from umtvit.config import Config, DatasetConfig, ModelConfig, load_config
from umtvit.models import (
    ProjectionHead,
    Soft3DSOM,
    SpatialUplifting,
    UMTViT,
    resolve_sigma,
)

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
_ALL_CONFIGS = ("shapes", "ham10000", "eurosat")


def _make_config(
    image_size: int = 32,
    *,
    fine_patch: int = 8,
    coarse_patch: int = 16,
    dim: int = 32,
    depth: int = 2,
    heads: int = 4,
    volume_grid: int = 4,
    volume_channels: int = 8,
    proj_dim: int = 16,
    som_grid=(3, 3, 3),
    som_update: str = "gradient",
    channels: int = 3,
) -> Config:
    """Build a small, validated Config for CPU tests."""
    return Config(
        dataset=DatasetConfig(image_size=image_size, channels=channels),
        model=ModelConfig(
            fine_patch=fine_patch,
            coarse_patch=coarse_patch,
            dim=dim,
            depth=depth,
            heads=heads,
            volume_h=volume_grid,
            volume_w=volume_grid,
            volume_channels=volume_channels,
            proj_dim=proj_dim,
            som_grid=tuple(som_grid),
            som_update=som_update,
        ),
    ).validate()


def _blobs(n_per: int = 200, dim: int = 8, sep: float = 10.0, std: float = 0.5,
           seed: int = 0) -> torch.Tensor:
    """~``3*n_per`` points in 3 well-separated Gaussian blobs in ``R^dim``."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.zeros(3, dim)
    centers[0, 0] = sep
    centers[1, 1] = sep
    centers[2, 2] = sep
    pts = [c + torch.randn(n_per, dim, generator=g) * std for c in centers]
    return torch.cat(pts, dim=0)


def _sigma_scheduler(grid):
    start, end = resolve_sigma(None, None, grid)

    def sigma_at(t: int, total: int) -> float:
        return start * (end / start) ** (t / max(1, total))

    return sigma_at


# --------------------------------------------------------------------------- #
# Shapes: volume / pooled / proj
# --------------------------------------------------------------------------- #
def test_uplifting_volume_shape():
    config = _make_config()
    backbone_out_layers = [
        torch.randn(2, config.model.volume_h ** 2, config.model.dim)
        for _ in range(config.model.depth)
    ]
    uplift = SpatialUplifting(config)
    volume = uplift(backbone_out_layers)
    m = config.model
    assert volume.shape == (2, m.volume_h, m.volume_w, m.depth, m.volume_channels)
    pooled = uplift.pooled(volume)
    assert pooled.shape == (2, m.depth * m.volume_channels)


def test_projection_head_shape():
    config = _make_config()
    head = ProjectionHead(config)
    pooled = torch.randn(2, config.model.depth * config.model.volume_channels)
    proj = head(pooled)
    assert proj.shape == (2, config.model.proj_dim)


def test_full_model_forward_shapes():
    config = _make_config()
    model = UMTViT(config)
    x = torch.randn(2, config.dataset.channels, config.model.image_size,
                    config.model.image_size)
    out = model(x)
    m = config.model
    assert set(out) == {"volume", "pooled", "proj", "layers"}
    assert out["volume"].shape == (2, m.volume_h, m.volume_w, m.depth, m.volume_channels)
    assert out["pooled"].shape == (2, m.depth * m.volume_channels)
    assert out["proj"].shape == (2, m.proj_dim)
    assert len(out["layers"]) == m.depth


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_volume_shape_from_shipped_config(name: str):
    config = load_config(_CONFIGS_DIR / f"{name}.yaml")
    model = UMTViT(config)
    m = config.model
    x = torch.randn(2, config.dataset.channels, m.image_size, m.image_size)
    out = model(x)
    assert out["volume"].shape == (2, m.volume_h, m.volume_w, m.depth, m.volume_channels)
    assert out["pooled"].shape == (2, m.depth * m.volume_channels)
    assert out["proj"].shape == (2, m.proj_dim)


def test_uplifting_rejects_wrong_layer_count():
    config = _make_config(depth=3)
    uplift = SpatialUplifting(config)
    too_few = [torch.randn(2, config.model.volume_h ** 2, config.model.dim)]
    with pytest.raises(ValueError):
        uplift(too_few)


# --------------------------------------------------------------------------- #
# SOM convergence — both update modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("update", ["gradient", "kohonen_ema"])
def test_som_quantization_error_decreases(update: str):
    torch.manual_seed(0)
    grid = (4, 4, 4)
    data = _blobs(dim=8, seed=1)
    som = Soft3DSOM(grid, feat_dim=8, tau=0.5, update=update)
    sigma_at = _sigma_scheduler(grid)
    steps = 200

    qe0 = som.metrics(data)["quantization_error"]
    if update == "gradient":
        opt = torch.optim.Adam([som.weights], lr=0.1)
        for t in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = som.loss(data, sigma_at(t, steps))
            loss.backward()
            opt.step()
    else:
        for t in range(steps):
            som.loss(data, sigma_at(t, steps))  # in-place EMA update
    qe1 = som.metrics(data)["quantization_error"]

    assert qe1 < qe0, f"{update}: QE did not fall ({qe0:.4f} -> {qe1:.4f})"


def test_som_topographic_error_below_random_baseline():
    torch.manual_seed(0)
    grid = (4, 4, 4)
    data = _blobs(dim=8, seed=2)

    random_som = Soft3DSOM(grid, feat_dim=8, tau=0.5, update="gradient")
    baseline_te = random_som.metrics(data)["topographic_error"]

    som = Soft3DSOM(grid, feat_dim=8, tau=0.5, update="gradient")
    som.data_init(data)  # spread neurons before unfolding
    sigma_at = _sigma_scheduler(grid)
    opt = torch.optim.Adam([som.weights], lr=0.1)
    steps = 200
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        som.loss(data, sigma_at(t, steps)).backward()
        opt.step()

    trained_te = som.metrics(data)["topographic_error"]
    assert trained_te < baseline_te, (
        f"topographic error not below random baseline "
        f"(trained {trained_te:.4f} vs random {baseline_te:.4f})"
    )


# --------------------------------------------------------------------------- #
# data_init and revive
# --------------------------------------------------------------------------- #
def test_data_init_lowers_dead_fraction():
    torch.manual_seed(0)
    grid = (4, 4, 4)
    data = _blobs(dim=8, seed=3)

    random_som = Soft3DSOM(grid, feat_dim=8, tau=0.5)
    random_dead = random_som.metrics(data)["dead_neuron_fraction"]

    data_som = Soft3DSOM(grid, feat_dim=8, tau=0.5)
    data_som.data_init(data)
    init_dead = data_som.metrics(data)["dead_neuron_fraction"]

    assert init_dead < random_dead, (
        f"data_init did not spread neurons "
        f"(dead {init_dead:.3f} vs random {random_dead:.3f})"
    )


def test_revive_reseeds_exactly_zero_hit_neurons():
    torch.manual_seed(0)
    som = Soft3DSOM((3, 3, 3), feat_dim=8, tau=0.5)
    pool = _blobs(n_per=50, dim=8, seed=4)
    before = som.weights.data.clone()

    hit_counts = torch.ones(som.K)
    dead_idx = torch.tensor([0, 5, 11, 26])
    hit_counts[dead_idx] = 0

    revived = som.revive(hit_counts, pool)
    assert revived == dead_idx.numel()

    after = som.weights.data
    changed = torch.tensor([not torch.equal(before[k], after[k]) for k in range(som.K)])
    reseeded = torch.nonzero(changed, as_tuple=False).flatten()
    assert torch.equal(reseeded.sort().values, dead_idx.sort().values), (
        "revive changed a different set of neurons than the zero-hit ones"
    )


def test_revive_noop_when_no_dead():
    som = Soft3DSOM((3, 3, 3), feat_dim=8, tau=0.5)
    pool = _blobs(n_per=10, dim=8, seed=5)
    before = som.weights.data.clone()
    revived = som.revive(torch.ones(som.K), pool)
    assert revived == 0
    assert torch.equal(before, som.weights.data)


# --------------------------------------------------------------------------- #
# sigma derivation
# --------------------------------------------------------------------------- #
def test_resolve_sigma_grid_derived_when_null():
    start, end = resolve_sigma(None, None, (4, 4, 4))
    assert start == pytest.approx(2.0)   # max(grid)/2
    assert end == pytest.approx(0.75)


def test_resolve_sigma_grid_derived_uses_max_axis():
    start, end = resolve_sigma(None, None, (2, 6, 3))
    assert start == pytest.approx(3.0)   # max(grid)/2
    assert end == pytest.approx(0.75)


def test_resolve_sigma_honours_explicit():
    start, end = resolve_sigma(3.0, 1.25, (8, 8, 8))
    assert start == pytest.approx(3.0)
    assert end == pytest.approx(1.25)


def test_resolve_sigma_partial_null():
    start, end = resolve_sigma(None, 0.5, (6, 6, 6))
    assert start == pytest.approx(3.0)
    assert end == pytest.approx(0.5)


def test_config_accepts_null_sigma():
    data = Config().to_dict()
    data["loss"]["sigma_start"] = None
    data["loss"]["sigma_end"] = None
    config = Config.from_dict(data)
    assert config.loss.sigma_start is None
    assert config.loss.sigma_end is None
    start, end = resolve_sigma(
        config.loss.sigma_start, config.loss.sigma_end, config.model.som_grid
    )
    assert start == pytest.approx(max(config.model.som_grid) / 2)
    assert end == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# gradient flows through the volume to the backbone
# --------------------------------------------------------------------------- #
def test_som_loss_backprops_to_backbone():
    config = _make_config(volume_channels=8, som_grid=(3, 3, 3))
    model = UMTViT(config)
    som = Soft3DSOM.from_config(config)
    x = torch.randn(2, config.dataset.channels, config.model.image_size,
                    config.model.image_size)

    volume = model(x)["volume"]
    vox = volume.reshape(-1, volume.shape[-1])
    loss = som.loss(vox, sigma=1.5)
    loss.backward()

    # The uplift projections sit directly on the SOM path; assert real signal.
    uplift_grads = [
        p.grad for n, p in model.named_parameters()
        if n.startswith("uplifting.") and p.grad is not None
    ]
    assert uplift_grads, "no uplifting parameters received a gradient"
    assert max(float(g.abs().max()) for g in uplift_grads) > 0.0

    # And it reaches deeper backbone (encoder) parameters too.
    encoder_grads = [
        float(p.grad.abs().max())
        for n, p in model.named_parameters()
        if n.startswith("backbone.encoder.") and p.grad is not None
    ]
    assert encoder_grads and max(encoder_grads) > 0.0, (
        "SOM loss did not propagate into the backbone encoder"
    )
