"""Controlled tests for encoder-only identity-centered SaS (data-free)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mscloudr.cli.eval_source_guided_dbcr import _checkpoint_config
from mscloudr.cli.train_source_guided_dbcr import _architecture, build_parser
from mscloudr.data import REFERENCE_PROTOCOL
from mscloudr.models.patent_blocks import PatentTimeEmbeddedNAFBlock, PatentTimeEmbeddingConfig
from mscloudr.models.source_guided_dbcr import (
    SOURCE_GUIDED_SCALE_SHIFT_MODES,
    SourceGuidedDBCRConfig,
    SourceGuidedDBCRNet,
)


def tiny_config(mode: str) -> SourceGuidedDBCRConfig:
    return SourceGuidedDBCRConfig(
        widths=(8, 16, 32, 64),
        encoder_blocks=(1, 1, 1, 2),
        decoder_blocks=(1, 1, 1, 1),
        sf_heads=(1, 1, 2, 4),
        time_embedding=PatentTimeEmbeddingConfig(
            positional_dim=8, hidden_dim=8,
        ),
        time_sharing="per_block",
        scale_shift_mode=mode,
    )


def test_experimental_mode_validated_and_patent_default_retained():
    assert SourceGuidedDBCRConfig().scale_shift_mode == "patent"
    assert SOURCE_GUIDED_SCALE_SHIFT_MODES == (
        "patent", "identity_centered", "encoder_only_identity_centered"
    )
    tiny_config("encoder_only_identity_centered").validate()
    with pytest.raises(ValueError, match="scale_shift_mode"):
        replace(tiny_config("patent"), scale_shift_mode="invalid").validate()
    with pytest.raises(ValueError, match="scale_shift_mode"):
        PatentTimeEmbeddedNAFBlock(
            8, scale_shift_mode="encoder_only_identity_centered"
        )


def test_only_optical_and_sar_encoders_use_identity_centered_equation():
    model = SourceGuidedDBCRNet(tiny_config("encoder_only_identity_centered"))
    for branch in (model.opt_encoders, model.sar_encoders):
        for stage in branch:
            for block in stage:
                assert block.scale_shift_mode == "identity_centered"
    for stage in (model.bottleneck, *model.decoders):
        for block in stage:
            assert block.scale_shift_mode == "patent"
    audit = model.architecture_audit()
    assert audit["num_time_conditioned_nafblocks"] == 14
    assert audit["scale_shift_modes_by_component"] == {
        "optical_encoder": "identity_centered",
        "sar_encoder": "identity_centered",
        "bottleneck": "patent",
        "decoder": "patent",
    }
    assert "encoder=" in audit["scale_shift_equation"]


def test_identical_weights_at_initialization_and_no_parameter_count_change():
    models = {}
    for mode in SOURCE_GUIDED_SCALE_SHIFT_MODES:
        torch.manual_seed(42)
        models[mode] = SourceGuidedDBCRNet(tiny_config(mode)).eval()
    baseline_state = models["patent"].state_dict()
    for mode, model in models.items():
        assert model.architecture_audit()["trainable_parameters"] == (
            models["patent"].architecture_audit()["trainable_parameters"]
        )
        for key, value in baseline_state.items():
            assert torch.equal(value, model.state_dict()[key]), (mode, key)

    cloudy = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    timestep = torch.tensor([1000.])
    with torch.no_grad():
        outputs = [model(cloudy, timestep, sar) for model in models.values()]
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_encoder_only_residual_is_active_with_original_bottleneck_decoder():
    torch.manual_seed(5)
    baseline = SourceGuidedDBCRNet(tiny_config("patent")).eval()
    mixed = SourceGuidedDBCRNet(
        tiny_config("encoder_only_identity_centered")
    ).eval()
    mixed.load_state_dict(baseline.state_dict(), strict=True)
    # Only encoder residuals are activated; bottleneck and decoder stay at
    # beta=gamma=0. This isolates the encoder equation.
    with torch.no_grad():
        for net in (baseline, mixed):
            for branch in (net.opt_encoders, net.sar_encoders):
                for stage in branch:
                    for block in stage:
                        block.beta.fill_(0.3)
                        block.gamma.fill_(0.3)

    cloudy = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    timestep = torch.tensor([500.])
    with torch.no_grad():
        control_output = baseline(cloudy, timestep, sar)
        mixed_output = mixed(cloudy, timestep, sar)
    assert torch.isfinite(control_output).all()
    assert torch.isfinite(mixed_output).all()
    assert not torch.allclose(control_output, mixed_output)


def test_cli_and_checkpoint_roundtrip():
    args = build_parser().parse_args([
        "--run-name", "encoder_only_sas_pilot_test",
        "--scale-shift-mode", "encoder_only_identity_centered",
        "--epochs", "3",
        "--train-seed", "42",
        "--sampler-seed", "42",
    ])
    config = _architecture(args)
    assert config.scale_shift_mode == "encoder_only_identity_centered"
    assert config.time_sharing == "per_block"

    metadata = {
        "model_identity": "dbcr_source_guided_v1",
        "split_protocol": REFERENCE_PROTOCOL,
        "schedule_name": "original",
        "conditioning_mode": "raw_t",
        "total_steps": config.total_steps,
        "architecture_config": config.audit_config(),
    }
    restored = _checkpoint_config(metadata)
    assert restored == config
    assert SourceGuidedDBCRNet(restored).architecture_audit()[
        "scale_shift_modes_by_component"
    ]["bottleneck"] == "patent"

    # Older source-guided runs should remain loadable.
    original = tiny_config("patent").audit_config()
    del original["scale_shift_mode"]
    metadata["architecture_config"] = original
    assert _checkpoint_config(metadata).scale_shift_mode == "patent"
