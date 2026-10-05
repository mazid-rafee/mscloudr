import math

import pytest
import torch

from mscloudr.cli.pilot_train import (
    _validate_generalization_measure_args,
    build_parser,
)
from mscloudr.generalization_measure import (
    DEFAULT_GENERALIZATION_MIX,
    GENERALIZATION_PROFILE_ALPHAS,
    GENERALIZATION_PROFILE_GAPS,
    generalization_measure_summary,
    generalization_mixture_schedule,
)
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING, RAW_T_CONDITIONING


def test_generalization_calibration_has_expected_knots_and_positive_gaps():
    assert len(GENERALIZATION_PROFILE_ALPHAS) == 11
    assert len(GENERALIZATION_PROFILE_GAPS) == 11
    assert GENERALIZATION_PROFILE_ALPHAS[0] == 0.0
    assert GENERALIZATION_PROFILE_ALPHAS[-1] == 1.0
    assert all(gap > 0.0 for gap in GENERALIZATION_PROFILE_GAPS)


def test_generalization_gap_peaks_before_endpoint_at_alpha_point_eight():
    peak_index = max(
        range(len(GENERALIZATION_PROFILE_GAPS)),
        key=lambda i: GENERALIZATION_PROFILE_GAPS[i],
    )
    assert GENERALIZATION_PROFILE_ALPHAS[peak_index] == 0.8
    assert GENERALIZATION_PROFILE_GAPS[peak_index] > GENERALIZATION_PROFILE_GAPS[-1]


def test_eta_zero_reduces_exactly_to_canonical_alpha():
    schedule = generalization_mixture_schedule(0.0)
    t = torch.tensor([0.0, 100.0, 250.0, 500.0, 900.0, 1000.0])
    alpha = schedule(t, 1000)
    assert torch.allclose(alpha, t / 1000.0, atol=1e-6, rtol=0.0)


def test_generalization_schedule_is_monotone_and_preserves_endpoints():
    schedule = generalization_mixture_schedule(DEFAULT_GENERALIZATION_MIX)
    t = torch.arange(0, 1001, dtype=torch.float32)
    alpha = schedule(t, 1000)

    assert float(alpha[0]) == 0.0
    assert float(alpha[-1]) == 1.0
    assert torch.all(alpha[1:] >= alpha[:-1])
    assert torch.all(alpha >= 0.0)
    assert torch.all(alpha <= 1.0)


def test_generalization_mix_changes_measure_without_endpoint_atom():
    schedule = generalization_mixture_schedule(DEFAULT_GENERALIZATION_MIX)
    t = torch.arange(0, 1001, dtype=torch.float32)
    alpha = schedule(t, 1000)

    assert not torch.allclose(alpha, t / 1000.0)
    assert int(torch.sum(alpha == 1.0).item()) == 1


def test_generalization_measure_summary_is_normalized_and_records_provenance():
    summary = generalization_measure_summary(DEFAULT_GENERALIZATION_MIX)
    assert summary["training_bridge_measure"] == (
        "generalization_gap_mixture_physical_alpha"
    )
    assert summary["generalization_mix_eta"] == DEFAULT_GENERALIZATION_MIX
    assert math.isclose(
        sum(summary["generalization_segment_masses"]),
        1.0,
        abs_tol=1e-12,
    )
    assert summary["coverage_preserving_uniform_component"] == 0.5
    assert summary["generalization_gap_peak_alpha"] == 0.8
    provenance = summary["generalization_profile_provenance"]
    assert provenance["profile_statistic"] == "sample_val_minus_train_l1"
    assert provenance["source_checkpoint_epoch"] == 20
    assert provenance["final_test_used_for_sampler_design"] is False


def test_generalization_measure_validation_requires_canonical_physical_alpha():
    with pytest.raises(ValueError, match="canonical_alpha"):
        _validate_generalization_measure_args(
            "original",
            PHYSICAL_ALPHA_CONDITIONING,
            0.5,
        )

    with pytest.raises(ValueError, match="physical_alpha"):
        _validate_generalization_measure_args(
            "canonical_alpha",
            RAW_T_CONDITIONING,
            0.5,
        )

    assert _validate_generalization_measure_args(
        "canonical_alpha",
        PHYSICAL_ALPHA_CONDITIONING,
        0.5,
    ) == 0.5


def test_generalization_mix_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="0 <= eta <= 1"):
        generalization_mixture_schedule(-0.01)
    with pytest.raises(ValueError, match="0 <= eta <= 1"):
        generalization_mixture_schedule(1.01)


def test_pilot_cli_accepts_generalization_mix():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-name",
            "unit-test",
            "--schedule",
            "canonical_alpha",
            "--conditioning",
            "physical_alpha",
            "--generalization-mix",
            "0.5",
        ]
    )
    assert args.generalization_mix == 0.5
