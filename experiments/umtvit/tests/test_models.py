"""U2 backbone tests (ARCHITECTURE §3.1-§3.3, §9 row U2).

All CPU-only and fast: tiny channel/dim/depth settings, batch size 2. Covers

- forward shapes at three image sizes (32/64/128) with divisible patches;
- both cross-attention modes forward and disagree with each other;
- the encoder returns exactly ``depth`` layer outputs, each correctly shaped;
- gradient reaches every parameter (non-None) after a scalar backward;
- the default GPU config stays within the parameter budget;
- the backbone builds straight from every shipped ``configs/*.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from umtvit.config import Config, DatasetConfig, ModelConfig, load_config
from umtvit.models import (
    CrossScaleBlock,
    DualScalePatchEmbed,
    FeatureFusion,
    TransformerEncoder,
    UMTViTBackbone,
)

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
_ALL_CONFIGS = ("shapes", "ham10000", "eurosat")


def _make_config(
    image_size: int,
    *,
    fine_patch: int = 8,
    coarse_patch: int = 16,
    dim: int = 32,
    depth: int = 2,
    heads: int = 4,
    volume_grid: int = 4,
    volume_channels: int = 8,
    cross_attention: str = "cls_bridged",
    cross_rounds: int = 1,
    channels: int = 3,
) -> Config:
    """Build a small, validated Config for CPU tests."""
    config = Config(
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
            cross_attention=cross_attention,
            cross_rounds=cross_rounds,
        ),
    )
    return config.validate()


# --- forward shapes at three resolutions ---------------------------------

@pytest.mark.parametrize("image_size", [32, 64, 128])
def test_backbone_forward_shapes(image_size: int):
    config = _make_config(image_size)
    backbone = UMTViTBackbone(config)
    x = torch.randn(2, 3, image_size, image_size)

    out = backbone(x)

    vg = config.model.volume_h
    dim = config.model.dim
    depth = config.model.depth
    assert set(out) == {"layers", "fused"}
    assert isinstance(out["layers"], list)
    assert len(out["layers"]) == depth
    for layer in out["layers"]:
        assert layer.shape == (2, vg * vg, dim)
    assert out["fused"].shape == (2, vg * vg, dim)


# --- dual-scale patch embed ----------------------------------------------

def test_patch_embed_stream_shapes():
    embed = DualScalePatchEmbed(
        image_size=64, fine_patch=8, coarse_patch=16, dim=32, channels=3
    )
    tokens_fine, tokens_coarse = embed(torch.randn(2, 3, 64, 64))
    # 64/8 = 8 -> 64 patch tokens + CLS; 64/16 = 4 -> 16 patch tokens + CLS.
    assert tokens_fine.shape == (2, 8 * 8 + 1, 32)
    assert tokens_coarse.shape == (2, 4 * 4 + 1, 32)


def test_patch_embed_rejects_indivisible_patch():
    with pytest.raises(ValueError):
        DualScalePatchEmbed(image_size=32, fine_patch=7, coarse_patch=16, dim=32)


# --- cross-attention modes -----------------------------------------------

@pytest.mark.parametrize("mode", ["cls_bridged", "full_pair"])
def test_cross_scale_block_preserves_shape(mode: str):
    block = CrossScaleBlock(dim=32, heads=4, mode=mode)
    tf = torch.randn(2, 17, 32)
    tc = torch.randn(2, 5, 32)
    out_f, out_c = block(tf, tc)
    assert out_f.shape == tf.shape
    assert out_c.shape == tc.shape


def test_cross_scale_block_rejects_unknown_mode():
    with pytest.raises(ValueError):
        CrossScaleBlock(dim=32, heads=4, mode="nonsense")


def test_cross_attention_modes_differ():
    """The two modes must produce different backbone outputs.

    Seeding identically before each build gives the two backbones the same
    initial weights (the module structure is identical across modes), so any
    difference in the output isolates the mode logic itself.
    """
    x = torch.randn(2, 3, 64, 64)

    torch.manual_seed(0)
    bridged = UMTViTBackbone(_make_config(64, cross_attention="cls_bridged"))
    torch.manual_seed(0)
    full = UMTViTBackbone(_make_config(64, cross_attention="full_pair"))

    with torch.no_grad():
        out_bridged = bridged(x)
        out_full = full(x)

    assert not torch.allclose(out_bridged["fused"], out_full["fused"])
    assert not torch.allclose(out_bridged["layers"][-1], out_full["layers"][-1])


# --- encoder returns all layers ------------------------------------------

@pytest.mark.parametrize("depth", [1, 3, 6])
def test_encoder_returns_every_layer(depth: int):
    encoder = TransformerEncoder(dim=32, depth=depth, heads=4, mlp_ratio=2.0)
    x = torch.randn(2, 10, 32)
    outputs = encoder(x)
    assert isinstance(outputs, list)
    assert len(outputs) == depth
    for layer in outputs:
        assert layer.shape == (2, 10, 32)


# --- fusion --------------------------------------------------------------

def test_fusion_resamples_onto_volume_grid():
    fusion = FeatureFusion(dim=32, grid_fine=8, grid_coarse=4, volume_grid=6)
    tokens_fine = torch.randn(2, 8 * 8 + 1, 32)
    tokens_coarse = torch.randn(2, 4 * 4 + 1, 32)
    fused = fusion(tokens_fine, tokens_coarse)
    assert fused.shape == (2, 6 * 6, 32)


# --- gradient reaches every parameter ------------------------------------

@pytest.mark.parametrize("mode", ["cls_bridged", "full_pair"])
def test_gradient_flows_to_every_parameter(mode: str):
    """A scalar backward must leave a (non-None) gradient on every parameter.

    Since the U2b fix (cross-scale attn runs *before* per-stream self-attn),
    the self-attn step redistributes the cross-attended CLS into the patch
    tokens before fusion drops the CLS, so the cross-attention parameters now
    receive genuinely non-zero gradients in ``cls_bridged`` mode too — see
    ``test_cross_params_get_nonzero_gradient`` for that stronger check.
    """
    config = _make_config(32, cross_attention=mode)
    backbone = UMTViTBackbone(config)
    out = backbone(torch.randn(2, 3, 32, 32))
    loss = sum(layer.sum() for layer in out["layers"]) + out["fused"].sum()
    loss.backward()
    missing = [name for name, p in backbone.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"


@pytest.mark.parametrize("mode", ["cls_bridged", "full_pair"])
def test_cross_params_get_nonzero_gradient(mode: str):
    """Cross-attention parameters receive NON-ZERO gradients (U2b).

    Before the U2b ordering fix, ``cls_bridged`` with ``cross_rounds=1`` wrote
    the cross-attended result only to the CLS tokens, which fusion drops before
    the objective — so the cross-attention weights received a strictly *zero*
    gradient (non-None but inert). Reordering each round to cross → self-attn
    lets the self-attn spread the updated CLS into the patch tokens, so the
    cross path now contributes to the loss. This asserts at least one
    ``self.cross`` parameter has a non-zero gradient after backward.
    """
    config = _make_config(32, cross_attention=mode, cross_rounds=1)
    backbone = UMTViTBackbone(config)
    out = backbone(torch.randn(2, 3, 32, 32))
    loss = sum(layer.sum() for layer in out["layers"]) + out["fused"].sum()
    loss.backward()

    cross_grads = [
        p.grad for name, p in backbone.named_parameters()
        if name.startswith("cross.") and p.grad is not None
    ]
    assert cross_grads, "no cross-attention parameters found"
    max_abs = max(float(g.abs().max()) for g in cross_grads)
    assert max_abs > 0.0, (
        f"cross-attention parameters received an all-zero gradient in {mode} "
        f"mode (max |grad| = {max_abs}); the cross path is inert"
    )


def test_cls_bridged_cross_path_is_live():
    """The ``cls_bridged`` cross path must influence the backbone output (U2b).

    Zeroing the cross-attention blocks' output projection weights *and* biases
    turns each cross round into an identity on the token streams. If the cross
    path were inert (the pre-U2b bug), the backbone output would be unchanged;
    since cross now precedes the per-stream self-attn, nulling it must shift the
    fused tokens and every layer output. Both backbones share initial weights
    (identical seed + identical structure) so the delta isolates the cross path.
    """
    x = torch.randn(2, 3, 32, 32)

    torch.manual_seed(0)
    live = UMTViTBackbone(_make_config(32, cross_attention="cls_bridged",
                                       cross_rounds=1))
    torch.manual_seed(0)
    nulled = UMTViTBackbone(_make_config(32, cross_attention="cls_bridged",
                                         cross_rounds=1))
    # Null every cross block's output projection -> cross becomes identity.
    with torch.no_grad():
        for block in nulled.cross:
            for attn in (block.fine_from_coarse, block.coarse_from_fine):
                attn.out_proj.weight.zero_()
                attn.out_proj.bias.zero_()

    with torch.no_grad():
        out_live = live(x)
        out_nulled = nulled(x)

    fused_delta = (out_live["fused"] - out_nulled["fused"]).abs().max().item()
    layer_delta = (
        out_live["layers"][-1] - out_nulled["layers"][-1]
    ).abs().max().item()
    assert fused_delta > 1e-6, (
        f"nulling the cross path left the fused tokens unchanged "
        f"(max delta {fused_delta}); the cross-scale bridge is inert"
    )
    assert layer_delta > 1e-6, (
        f"nulling the cross path left the final layer output unchanged "
        f"(max delta {layer_delta})"
    )


# --- parameter budget ----------------------------------------------------

def test_default_gpu_config_within_budget():
    # ham10000 carries the DECISION-LOG GPU standing defaults (dim 256, L 8).
    config = load_config(_CONFIGS_DIR / "ham10000.yaml")
    backbone = UMTViTBackbone(config)
    n_params = sum(p.numel() for p in backbone.parameters())
    assert n_params <= 25_000_000, f"{n_params/1e6:.2f}M params exceeds 25M budget"


# --- config-driven construction from every shipped YAML ------------------

@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_backbone_builds_from_shipped_config(name: str):
    config = load_config(_CONFIGS_DIR / f"{name}.yaml")
    backbone = UMTViTBackbone(config)
    image_size = config.model.image_size
    x = torch.randn(2, config.dataset.channels, image_size, image_size)
    out = backbone(x)
    vg = config.model.volume_h
    assert len(out["layers"]) == config.model.depth
    assert out["fused"].shape == (2, vg * vg, config.model.dim)


def test_backbone_rejects_non_square_volume():
    config = Config(
        dataset=DatasetConfig(image_size=32, channels=3),
        model=ModelConfig(
            fine_patch=8, coarse_patch=16, dim=32, depth=2, heads=4,
            volume_h=4, volume_w=6, volume_channels=8,
        ),
    ).validate()
    with pytest.raises(ValueError):
        UMTViTBackbone(config)
