"""Synthetic, data-free tests for the read-only internal path diagnostic."""

from __future__ import annotations

import math

import pytest
import torch

from mscloudr.internal_path_diagnostics import (
    InternalPathRecorder, parameter_grad_rms, rms, stage_name
)
from mscloudr.models.patent_blocks import (
    PatentTimeEmbeddedNAFBlock, PatentTimeEmbeddingConfig
)
from mscloudr.models.source_guided_dbcr import (
    SourceGuidedDBCRConfig, SourceGuidedDBCRNet
)


def tiny_model():
    cfg = SourceGuidedDBCRConfig(
        widths=(8, 16, 32, 64),
        encoder_blocks=(1, 1, 1, 2),
        decoder_blocks=(1, 1, 1, 1),
        sf_heads=(1, 1, 2, 4),
        time_embedding=PatentTimeEmbeddingConfig(
            positional_dim=8, hidden_dim=8
        ),
    )
    model = SourceGuidedDBCRNet(cfg).eval()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, PatentTimeEmbeddedNAFBlock):
                layer.beta.fill_(0.1)
                layer.gamma.fill_(0.1)
    return model


def test_stage_grouping_and_rms():
    assert stage_name("opt_encoders.3.27") == "opt_encoders.3"
    assert stage_name("sar_encoders.1.0") == "sar_encoders.1"
    assert stage_name("decoders.2.0") == "decoders.2"
    assert stage_name("bottleneck.0") == "bottleneck"
    x = torch.tensor([[[[3., 4.]]], [[[0., 0.]]]])
    assert rms(x).tolist() == pytest.approx(
        [5.0 / math.sqrt(2.0), 0.0]
    )


def test_hooked_forward_and_backward_have_no_parameter_updates():
    torch.manual_seed(42)
    model = tiny_model()
    recorder = InternalPathRecorder(model)
    cloudy = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    target = torch.randn(1, 13, 16, 16)
    t = torch.tensor([1000.])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        baseline = model(cloudy, t, sar)
    with torch.enable_grad(), recorder.capture():
        pred = model(cloudy, t, sar)
        assert torch.equal(baseline, pred)
        ((pred - target).abs().mean()).backward()
        recorder.record_parameter_grads()
    result = recorder.summary()
    assert result["num_blocks"] == 14
    assert result["num_batches"] == 1
    assert result["stages"]["opt_encoders.3"]["num_blocks"] == 2
    assert result["stages"]["sar_encoders.3"]["num_blocks"] == 2
    assert set(result["blocks"]) == set(recorder.blocks)

    for name, record in result["blocks"].items():
        for key in (
            "scale_conv_abs_mean_mean", "scale_ffn_abs_mean_mean",
            "norm1_rms_mean", "mod_conv_rms_mean", "after_depthwise_rms_mean",
            "after_mb_simplegate_rms_mean", "sca_gate_abs_mean_mean",
            "sca_gate_rms_mean", "after_sca_rms_mean",
            "beta_scaled_sca_rms_mean", "norm2_rms_mean",
            "mod_ffn_rms_mean", "ffn_pre_gate_rms_mean",
            "ffn_post_gate_rms_mean", "gamma_scaled_ffn_rms_mean",
            "input_grad_rms_mean", "beta_grad_rms_mean",
            "conv1.weight_grad_rms_mean",
            "time_embedding.linear3.weight_grad_rms_mean",
        ):
            assert key in record, (name, key)
            assert record[key] is not None
            assert math.isfinite(record[key]), (name, key)
            assert record[key] >= 0.0
    assert any(
        record["input_grad_rms_mean"] > 0
        for record in result["blocks"].values()
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), name
    assert recorder._handles == []
    assert recorder._tensor_handles == []


def test_hooks_clean_up_after_exception():
    model = tiny_model()
    recorder = InternalPathRecorder(model)
    with pytest.raises(RuntimeError, match="test failure"):
        with recorder.capture():
            raise RuntimeError("test failure")
    assert not recorder._handles
    assert not recorder._tensor_handles
    for block in recorder.blocks.values():
        assert not block.norm1._forward_hooks
        assert not block.conv1._forward_pre_hooks
        assert not block.time_embedding.linear3._forward_hooks


def test_empty_gradient_is_reported_without_fake_zero():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    assert parameter_grad_rms(parameter) is None
    (parameter * 3).sum().backward()
    assert parameter_grad_rms(parameter) == pytest.approx(3.0)
