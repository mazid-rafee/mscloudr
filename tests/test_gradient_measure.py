import math

import pytest
import torch

from mscloudr.cli.pilot_train import (
    _validate_gradient_measure_args,
    build_parser,
)
from mscloudr.gradient_measure import (
    DEFAULT_GRADIENT_MIX,
    GRADIENT_PROFILE_ALPHAS,
    GRADIENT_PROFILE_RMS,
    gradient_measure_summary,
    gradient_mixture_schedule,
)
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING, RAW_T_CONDITIONING


def test_gradient_calibration_has_expected_knots_and_positive_scores():
    assert len(GRADIENT_PROFILE_ALPHAS) == 11
    assert len(GRADIENT_PROFILE_RMS) == 11
    assert GRADIENT_PROFILE_ALPHAS[0] == 0.0
    assert GRADIENT_PROFILE_ALPHAS[-1] == 1.0
    assert all(score > 0.0 for score in GRADIENT_PROFILE_RMS)


def test_eta_zero_reduces_exactly_to_canonical_alpha():
    schedule = gradient_mixture_schedule(0.0)
    t = torch.tensor([0.0, 100.0, 250.0, 500.0, 900.0, 1000.0])
    alpha = schedule(t, 1000)
    assert torch.allclose(alpha, t / 1000.0, atol=1e-6, rtol=0.0)


def test_gradient_schedule_is_monotone_and_preserves_endpoints():
    schedule = gradient_mixture_schedule(DEFAULT_GRADIENT_MIX)
    t = torch.arange(0, 1001, dtype=torch.float32)
    alpha = schedule(t, 1000)

    assert float(alpha[0]) == 0.0
    assert float(alpha[-1]) == 1.0
    assert torch.all(alpha[1:] >= alpha[:-1])
    assert torch.all(alpha >= 0.0)
    assert torch.all(alpha <= 1.0)


def test_gradient_mix_changes_measure_without_endpoint_atom():
    schedule = gradient_mixture_schedule(DEFAULT_GRADIENT_MIX)
    t = torch.arange(0, 1001, dtype=torch.float32)
    alpha = schedule(t, 1000)

    # Interior quantiles move relative to canonical alpha, but only the final
    # base quantile maps exactly to the endpoint; there is no endpoint atom.
    assert not torch.allclose(alpha, t / 1000.0)
    assert int(torch.sum(alpha == 1.0).item()) == 1


def test_gradient_measure_summary_is_normalized_and_records_provenance():
    summary = gradient_measure_summary(DEFAULT_GRADIENT_MIX)
    assert summary["training_bridge_measure"] == "gradient_rms_mixture_physical_alpha"
    assert summary["gradient_mix_eta"] == DEFAULT_GRADIENT_MIX
    assert math.isclose(sum(summary["gradient_segment_masses"]), 1.0, abs_tol=1e-12)
    assert summary["coverage_preserving_uniform_component"] == 0.5
    assert summary["gradient_profile_provenance"]["profile_statistic"] == (
        "batch_gradient_l2.rms"
    )
    assert summary["gradient_profile_provenance"]["source_checkpoint_epoch"] == 20


def test_gradient_measure_validation_requires_canonical_physical_alpha():
    with pytest.raises(ValueError, match="canonical_alpha"):
        _validate_gradient_measure_args(
            "original",
            PHYSICAL_ALPHA_CONDITIONING,
            0.5,
        )

    with pytest.raises(ValueError, match="physical_alpha"):
        _validate_gradient_measure_args(
            "canonical_alpha",
            RAW_T_CONDITIONING,
            0.5,
        )

    assert _validate_gradient_measure_args(
        "canonical_alpha",
        PHYSICAL_ALPHA_CONDITIONING,
        0.5,
    ) == 0.5


def test_gradient_mix_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="0 <= eta <= 1"):
        gradient_mixture_schedule(-0.01)
    with pytest.raises(ValueError, match="0 <= eta <= 1"):
        gradient_mixture_schedule(1.01)


def test_pilot_cli_accepts_gradient_mix():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-name",
            "unit-test",
            "--schedule",
            "canonical_alpha",
            "--conditioning",
            "physical_alpha",
            "--gradient-mix",
            "0.5",
        ]
    )
    assert args.gradient_mix == 0.5
