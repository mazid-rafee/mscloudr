"""Controlled SaS equation experiment; synthetic tests (no SEN12MS-CR needed)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mscloudr.cli.eval_source_guided_dbcr import _checkpoint_config
from mscloudr.cli.train_source_guided_dbcr import _architecture, build_parser
from mscloudr.data import REFERENCE_PROTOCOL
from mscloudr.models.patent_blocks import PatentTimeEmbeddedNAFBlock, PatentTimeEmbeddingConfig
from mscloudr.models.source_guided_dbcr import SourceGuidedDBCRConfig, SourceGuidedDBCRNet


def tiny_cfg(mode: str = "patent") -> SourceGuidedDBCRConfig:
    return SourceGuidedDBCRConfig(
        widths=(8, 16, 32, 64),
        encoder_blocks=(1, 1, 1, 2),
        decoder_blocks=(1, 1, 1, 1),
        sf_heads=(1, 1, 2, 4),
        time_embedding=PatentTimeEmbeddingConfig(positional_dim=8, hidden_dim=8),
        scale_shift_mode=mode,
    )


def test_patent_default_is_unchanged_and_invalid_modes_rejected():
    assert SourceGuidedDBCRConfig().scale_shift_mode == "patent"
    with pytest.raises(ValueError, match="scale_shift_mode"):
        replace(tiny_cfg(), scale_shift_mode="other").validate()
    with pytest.raises(ValueError, match="scale_shift_mode"):
        PatentTimeEmbeddedNAFBlock(8, scale_shift_mode="other")


def test_modulation_inputs_follow_exact_equations_and_both_branches():
    torch.manual_seed(42)
    patent = PatentTimeEmbeddedNAFBlock(8, scale_shift_mode="patent").eval()
    experimental = PatentTimeEmbeddedNAFBlock(
        8, scale_shift_mode="identity_centered"
    ).eval()
    experimental.load_state_dict(patent.state_dict(), strict=True)
    x = torch.randn(1, 8, 8, 8)
    zero = torch.zeros(1, 8, 1, 1)
    modulation = (zero, zero, zero, zero)

    results = {}
    def capture(name: str):
        return lambda module, inputs: results.__setitem__(name, inputs[0].detach().clone())

    with torch.no_grad():
        handles = [
            patent.conv1.register_forward_pre_hook(capture("patent_mb")),
            patent.ffn1.register_forward_pre_hook(capture("patent_ffn")),
            experimental.conv1.register_forward_pre_hook(capture("exp_mb")),
            experimental.ffn1.register_forward_pre_hook(capture("exp_ffn")),
        ]
        try:
            y_patent = patent.forward_with_modulation(x, modulation)
            y_exp = experimental.forward_with_modulation(x, modulation)
        finally:
            for handle in handles:
                handle.remove()

        assert torch.equal(y_patent, x)
        assert torch.equal(y_exp, x)  # beta/gamma are zero initially
        assert torch.count_nonzero(results["patent_mb"]) == 0
        assert torch.count_nonzero(results["patent_ffn"]) == 0
        assert torch.allclose(results["exp_mb"], experimental.norm1(x))
        assert torch.allclose(results["exp_ffn"], experimental.norm2(x))

    # With non-zero residual scalars, the *same* learned weights produce
    # different outcomes because only the scale-shift equation changes.
    with torch.no_grad():
        for block in (patent, experimental):
            block.beta.fill_(0.2)
            block.gamma.fill_(0.2)
        output_patent = patent.forward_with_modulation(x, modulation)
        output_exp = experimental.forward_with_modulation(x, modulation)
    assert not torch.allclose(output_patent, output_exp)


def test_full_models_have_identical_seeded_state_and_parameter_counts():
    torch.manual_seed(42)
    baseline = SourceGuidedDBCRNet(tiny_cfg("patent")).eval()
    torch.manual_seed(42)
    variant = SourceGuidedDBCRNet(tiny_cfg("identity_centered")).eval()
    assert baseline.architecture_audit()["trainable_parameters"] == (
        variant.architecture_audit()["trainable_parameters"]
    )
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, variant.state_dict()[name]), name
    assert all(
        block.scale_shift_mode == "identity_centered"
        for branch in (variant.opt_encoders, variant.sar_encoders)
        for stage in branch for block in stage
    )
    assert all(
        block.scale_shift_mode == "identity_centered"
        for stage in (*variant.decoders, variant.bottleneck)
        for block in stage
    )
    assert variant.architecture_audit()["scale_shift_equation"] == (
        "x*(1+scale)+shift"
    )
    cloudy = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        first_baseline = baseline(cloudy, torch.tensor([1000.]), sar)
        first_variant = variant(cloudy, torch.tensor([1000.]), sar)
    assert torch.equal(first_baseline, first_variant)
    with torch.no_grad():
        for model in (baseline, variant):
            for module in model.modules():
                if isinstance(module, PatentTimeEmbeddedNAFBlock):
                    module.beta.fill_(0.1)
                    module.gamma.fill_(0.1)
        baseline_out = baseline(cloudy, torch.tensor([500.]), sar)
        variant_out = variant(cloudy, torch.tensor([500.]), sar)
    assert not torch.allclose(baseline_out, variant_out)


def test_experimental_backward_reaches_time_embedding():
    model = SourceGuidedDBCRNet(tiny_cfg("identity_centered"))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PatentTimeEmbeddedNAFBlock):
                module.beta.fill_(0.1)
                module.gamma.fill_(0.1)
    cloudy = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    loss = model(cloudy, torch.tensor([500.]), sar).abs().mean()
    loss.backward()
    grad = model.opt_encoders[0][0].time_embedding.linear3.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_cli_and_checkpoint_roundtrip_and_older_metadata_compatibility():
    arguments = build_parser().parse_args([
        "--run-name", "identity_sas_test",
        "--epochs", "3",
        "--scale-shift-mode", "identity_centered",
    ])
    cfg = _architecture(arguments)
    assert cfg.scale_shift_mode == "identity_centered"
    assert cfg.time_sharing == "per_block"
    assert cfg.total_steps == 1000
    assert _architecture(build_parser().parse_args([
        "--run-name", "baseline_default_test"
    ])).scale_shift_mode == "patent"

    metadata = {
        "model_identity": "dbcr_source_guided_v1",
        "split_protocol": REFERENCE_PROTOCOL,
        "schedule_name": "original",
        "conditioning_mode": "raw_t",
        "total_steps": cfg.total_steps,
        "architecture_config": cfg.audit_config(),
    }
    assert _checkpoint_config(metadata) == cfg

    old = tiny_cfg().audit_config()
    old.pop("scale_shift_mode")
    metadata["architecture_config"] = old
    assert _checkpoint_config(metadata).scale_shift_mode == "patent"
