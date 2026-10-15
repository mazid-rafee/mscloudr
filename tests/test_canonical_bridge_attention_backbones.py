"""Controlled Swin and HDiT-inspired CanonicalBridge regression tests.

No SEN12MS-CR files or GPU required.
"""

import pytest
import torch

from mscloudr.cli.eval_canonical_bridge import (
    _checkpoint_backbone,
    _validate_checkpoint,
)
from mscloudr.cli.train_canonical_bridge import (
    _resume_identity,
    _validate_resume_metadata,
    build_parser,
)
from mscloudr.checkpointing import CHECKPOINT_FORMAT_VERSION
from mscloudr.data import REFERENCE_PROTOCOL
from mscloudr.models import CanonicalBridgeNet, count_canonical_bridge_parameters
from mscloudr.models.canonical_attention import (
    GlobalAttentionBlock,
    SwinStage,
    WindowAttentionBlock,
)


@pytest.mark.parametrize("variant", ["swin", "hdit_global"])
def test_attention_variants_forward_backward(variant):
    torch.manual_seed(42)
    # Reduced dimensions for fast CPU validation; production widths stay
    # (32,64,128,256).
    model = CanonicalBridgeNet(
        widths=(8, 16, 32, 64),
        total_steps=1000,
        backbone_variant=variant,
    )
    x = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    out = model(x, torch.tensor([1000.0]), sar)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert model.opt_stem.weight.grad is not None
    assert torch.isfinite(model.opt_stem.weight.grad).all()


def test_swin_placement_and_shifted_window_alternation():
    model = CanonicalBridgeNet(backbone_variant="swin")
    assert isinstance(model.opt_blocks[2], SwinStage)
    assert isinstance(model.opt_blocks[3], SwinStage)
    assert model.opt_blocks[2].blocks[0].shift_size == 0
    assert model.opt_blocks[2].blocks[1].shift_size == 4
    assert [block.shift_size for block in model.mid_blocks] == [0, 4]
    assert isinstance(model.sar_blocks[2], type(model.sar_blocks[0]))
    assert isinstance(model.dec_blocks[0], type(model.dec_blocks[1]))


def test_hdit_placement_preserves_shallow_cnn():
    model = CanonicalBridgeNet(backbone_variant="hdit_global")
    assert not isinstance(model.opt_blocks[2], SwinStage)
    assert isinstance(model.opt_blocks[3][0], GlobalAttentionBlock)
    assert isinstance(model.opt_blocks[3][1], GlobalAttentionBlock)
    assert all(isinstance(block, GlobalAttentionBlock) for block in model.mid_blocks)


def test_window_shift_mask_with_padded_spatial_shape():
    torch.manual_seed(10)
    block = WindowAttentionBlock(channels=32, window_size=8, shift_size=4)
    x = torch.randn(1, 32, 10, 11, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_cnn_default_is_unchanged():
    model = CanonicalBridgeNet()
    assert model.backbone_variant == "cnn"
    assert count_canonical_bridge_parameters(model) == 7653037
    assert _checkpoint_backbone({}) == "cnn"
    assert _checkpoint_backbone({"backbone_variant": "swin"}) == "swin"
    with pytest.raises(ValueError, match="backbone_variant"):
        _checkpoint_backbone({"backbone_variant": "unknown"})


def test_cli_backbone_selection():
    parser = build_parser()
    assert parser.parse_args(["--run-name", "base"]).backbone == "cnn"
    for variant in ("swin", "hdit_global"):
        args = parser.parse_args(
            ["--run-name", variant, "--backbone", variant]
        )
        assert args.backbone == variant


def test_checkpoint_selection_and_strict_weights_roundtrip():
    for variant in ("swin", "hdit_global"):
        source = CanonicalBridgeNet(
            widths=(8, 16, 32, 64),
            backbone_variant=variant,
        )
        meta = {
            "model_identity": "canonical_bridge_net",
            "split_protocol": REFERENCE_PROTOCOL,
            "schedule_name": "canonical_alpha",
            "bridge_geometry": "straight_linear",
            "conditioning_mode": "mr_inverse",
            "beta_a": 1.0,
            "total_steps": 1000,
            "architecture_widths": [8, 16, 32, 64],
            "trainable_parameters": count_canonical_bridge_parameters(source),
            "backbone_variant": variant,
            "paper_grade": True,
        }
        validated = _validate_checkpoint(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model_state": source.state_dict(),
                "run_metadata": meta,
            },
            allow_smoke_checkpoint=False,
        )
        assert _checkpoint_backbone(validated) == variant
        target = CanonicalBridgeNet(
            widths=tuple(meta["architecture_widths"]),
            backbone_variant=_checkpoint_backbone(validated),
        )
        target.load_state_dict(source.state_dict(), strict=True)


def test_resume_rejects_backbone_change():
    meta = {
        "model_identity": "canonical_bridge_net",
        "trainable_parameters": 7653037,
        "architecture_widths": [32, 64, 128, 256],
        "bridge_geometry": "straight_linear",
        "schedule_name": "canonical_alpha",
        "training_bridge_measure": "uniform_discrete_alpha_grid",
        "beta_a": 1.0,
        "beta_b": 1.0,
        "conditioning_mode": "mr_inverse",
        "conditioning_definition": "mr_inverse",
        "total_steps": 1000,
        "batch_size": 4,
        "num_workers": 4,
        "learning_rate": 5e-5,
        "split_protocol": REFERENCE_PROTOCOL,
        "split_audit": {},
        "reproducibility": {
            "seeds": {"train_seed": 42, "sampler_seed": 42},
            "deterministic_algorithms": False,
        },
    }
    assert _resume_identity(meta)["backbone_variant"] == "cnn"
    changed = {**meta, "backbone_variant": "swin"}
    with pytest.raises(ValueError, match="backbone_variant"):
        _validate_resume_metadata(meta, changed)


@pytest.mark.parametrize("variant", ["cnn", "swin", "hdit_global"])
def test_parameter_count_positive(variant):
    model = CanonicalBridgeNet(
        widths=(8, 16, 32, 64), backbone_variant=variant
    )
    assert count_canonical_bridge_parameters(model) > 0
