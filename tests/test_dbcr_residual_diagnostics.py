"""Tests for source-guided DB-CR residual measurement, no SEN12MS-CR data needed."""

from __future__ import annotations

import math

import pytest
import torch

from mscloudr.cli.diagnose_dbcr_residuals import evenly_spaced_indices
from mscloudr.models.patent_blocks import (
    PatentTimeEmbeddedNAFBlock,
    PatentTimeEmbeddingConfig,
)
from mscloudr.models.source_guided_dbcr import (
    SourceGuidedDBCRConfig,
    SourceGuidedDBCRNet,
)
from mscloudr.residual_diagnostics import (
    ResidualContributionRecorder,
    relative_residual_norms,
)


def small_model():
    config = SourceGuidedDBCRConfig(
        widths=(8, 16, 32, 64),
        encoder_blocks=(1, 1, 1, 2),
        decoder_blocks=(1, 1, 1, 1),
        sf_heads=(1, 1, 2, 4),
        time_embedding=PatentTimeEmbeddingConfig(
            positional_dim=8, hidden_dim=8
        ),
    )
    return SourceGuidedDBCRNet(config).eval()


def test_known_tensor_ratios_are_correct():
    x = torch.tensor([[[[3., 4.]]]])
    z = torch.tensor([[[[6., 8.]]]])
    y = torch.tensor([[[[12., 16.]]]])
    mbconv, ffn, combined = relative_residual_norms(x, z, y)
    assert mbconv.shape == (1,)
    assert float(mbconv.item()) == pytest.approx(1.0)
    assert float(ffn.item()) == pytest.approx(1.0)
    assert float(combined.item()) == pytest.approx(3.0)


def test_evenly_spaced_indices_are_unique_and_fixed():
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(9, 1) == [4]
    assert evenly_spaced_indices(10, 10) == list(range(10))
    with pytest.raises(ValueError):
        evenly_spaced_indices(3, 4)


def test_hook_measurements_zero_when_residual_scales_zero():
    torch.manual_seed(0)
    model = small_model()
    recorder = ResidualContributionRecorder(model)
    cloudy = torch.randn(2, 13, 16, 16)
    sar = torch.randn(2, 2, 16, 16)

    with torch.inference_mode(), recorder.capture():
        prediction = model(cloudy, torch.tensor([1000., 1000.]), sar)
        assert torch.isfinite(prediction).all()

    stats = recorder.summary()
    assert stats["num_samples"] == 2
    assert stats["num_blocks"] == 14
    assert stats["num_fusion_blocks"] == 4
    assert len(stats["fusion_blocks"]) == 4
    for entry in stats["blocks"].values():
        assert entry["mbconv_relative_l2_mean"] == 0.0
        assert entry["ffn_relative_l2_mean"] == 0.0
        assert entry["total_relative_l2_mean"] == 0.0
        assert entry["mbconv_update_l2_mean"] == 0.0
        assert entry["ffn_update_rms_mean"] == 0.0
        assert entry["total_update_rms_mean"] == 0.0
        assert entry["input_l2_mean"] >= 0.0
        assert entry["input_rms_mean"] >= 0.0
    for fusion in stats["fusion_blocks"].values():
        for key, value in fusion.items():
            assert math.isfinite(value), key
        assert fusion["optical_input_l2_mean"] >= 0.0
        assert fusion["sar_input_l2_mean"] >= 0.0
        assert fusion["output_minus_optical_l2_mean"] >= 0.0
    assert not recorder._handles
    assert not recorder._active
    assert not recorder._fusion_active


def test_hooks_do_not_change_prediction_and_measure_nonzero_residuals():
    torch.manual_seed(1)
    model = small_model()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PatentTimeEmbeddedNAFBlock):
                module.beta.fill_(0.2)
                module.gamma.fill_(0.2)
    cloudy = torch.randn(1, 13, 16, 16)
    sar = torch.randn(1, 2, 16, 16)
    t = torch.tensor([1000.])
    recorder = ResidualContributionRecorder(model)

    with torch.inference_mode():
        baseline = model(cloudy, t, sar)
        with recorder.capture():
            hooked = model(cloudy, t, sar)
    assert torch.equal(baseline, hooked)

    result = recorder.summary()
    assert result["num_samples"] == 1
    assert result["num_blocks"] == 14
    assert result["num_fusion_blocks"] == 4
    assert any(
        entry["mbconv_relative_l2_mean"] > 0
        or entry["ffn_relative_l2_mean"] > 0
        for entry in result["blocks"].values()
    )
    for entry in result["blocks"].values():
        for key in (
            "mbconv_relative_l2_mean",
            "ffn_relative_l2_mean",
            "total_relative_l2_mean",
        ):
            assert math.isfinite(entry[key])


def test_absolute_and_rms_fields_are_self_consistent():
    torch.manual_seed(9)
    model = small_model()
    cloudy = torch.randn(2, 13, 16, 16)
    sar = torch.randn(2, 2, 16, 16)
    recorder = ResidualContributionRecorder(model)
    with torch.inference_mode(), recorder.capture():
        model(cloudy, torch.full((2,), 1000.0), sar)
    result = recorder.summary()
    # Input RMS is L2 divided by sqrt(number of feature elements).
    first = result["blocks"]["opt_encoders.0.0"]
    assert first["input_l2_mean"] == pytest.approx(
        first["input_rms_mean"] * math.sqrt(8 * 16 * 16), rel=1e-5
    )
    for name, fusion in result["fusion_blocks"].items():
        stage_number = int(name.split(".")[1])
        c = [8, 16, 32, 64][stage_number]
        h = 16 // (2 ** stage_number)
        assert fusion["output_minus_optical_l2_mean"] == pytest.approx(
            fusion["output_minus_optical_rms_mean"] * math.sqrt(c * h * h),
            rel=1e-5,
        )


def test_hook_cleanup_on_exception():
    model = small_model()
    recorder = ResidualContributionRecorder(model)
    with pytest.raises(RuntimeError, match="injected error"):
        with recorder.capture():
            raise RuntimeError("injected error")
    assert not recorder._handles
    assert not recorder._active
    assert not recorder._fusion_active
    assert all(not block._forward_hooks for block in recorder.blocks.values())
    assert all(not fusion._forward_hooks for fusion in recorder.fusions.values())
