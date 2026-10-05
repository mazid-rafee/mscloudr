import math

import pytest
import torch

from mscloudr.cli.pilot_train import (
    _validate_difficulty_measure_args,
    build_parser,
)
from mscloudr.difficulty_measure import (
    DEFAULT_DIFFICULTY_MIX,
    DIFFICULTY_PROFILE_ALPHAS,
    DIFFICULTY_PROFILE_TRAIN_L1,
    difficulty_measure_summary,
    difficulty_mixture_schedule,
)
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING, RAW_T_CONDITIONING


def test_difficulty_calibration_has_expected_knots_and_positive_scores():
    assert len(DIFFICULTY_PROFILE_ALPHAS) == 11
    assert len(DIFFICULTY_PROFILE_TRAIN_L1) == 11
    assert DIFFICULTY_PROFILE_ALPHAS[0] == 0.0
    assert DIFFICULTY_PROFILE_ALPHAS[-1] == 1.0
    assert all(score > 0.0 for score in DIFFICULTY_PROFILE_TRAIN_L1)


def test_difficulty_profile_rises_toward_endpoint():
    assert DIFFICULTY_PROFILE_TRAIN_L1[-1] > DIFFICULTY_PROFILE_TRAIN_L1[0]
    assert all(
        right >= left
        for left, right in zip(
            DIFFICULTY_PROFILE_TRAIN_L1,
            DIFFICULTY_PROFILE_TRAIN_L1[1:],
        )
    )


def test_eta_zero_reduces_exactly_to_canonical_alpha():
    schedule = difficulty_mixture_schedule(0.0)
    t = torch.tensor([0.0, 100.0, 250.0, 500.0, 900.0, 1000.0])
    alpha = schedule(t, 1000)
    assert torch.allclose(alpha, t / 1000.0, atol=1e-6, rtol=0.0)


def test_difficulty_schedule_is_monotone_and_preserves_endpoints():
    schedule = difficulty_mixture_schedule(DEFAULT_DIFFICULTY_MIX)
    t = torch.arange(0, 1001, dtype=torch.float32)
    alpha = schedule(t, 1000)

    assert float(alpha[0]) == 0.0
    assert float(alpha[-1]) == 1.0
    assert torch.all(alpha[1:] >= alpha[:-1])
    assert torch.all(alpha >= 0.0)
    assert torch.all(alpha <= 1.0)


def test_difficulty_mix_changes_measure_without_endpoint_atom():
    schedule = difficulty_mixture_schedule(DEFAULT_DIFFICULTY_MIX)
    t = torch.arange(0, 1001, dtype=torch.float32)
    alpha = schedule(t, 1000)

    assert not torch.allclose(alpha, t / 1000.0)
    assert int(torch.sum(alpha == 1.0).item()) == 1


def test_difficulty_measure_summary_is_normalized_and_records_provenance():
    summary = difficulty_measure_summary(DEFAULT_DIFFICULTY_MIX)
    assert summary["training_bridge_measure"] == "difficulty_l1_mixture_physical_alpha"
    assert summary["difficulty_mix_eta"] == DEFAULT_DIFFICULTY_MIX
    assert math.isclose(sum(summary["difficulty_segment_masses"]), 1.0, abs_tol=1e-12)
    assert summary["coverage_preserving_uniform_component"] == 0.5
    provenance = summary["difficulty_profile_provenance"]
    assert provenance["profile_statistic"] == "fixed_alpha_train_mean_l1"
    assert provenance["profile_samples_per_alpha"] == 10714
    assert provenance["sampler_design_split"] == "train_only"
    assert provenance["validation_used_for_sampler_design"] is False
    assert provenance["test_used_for_sampler_design"] is False


def test_difficulty_measure_validation_requires_canonical_physical_alpha():
    with pytest.raises(ValueError, match="canonical_alpha"):
        _validate_difficulty_measure_args(
            "original",
            PHYSICAL_ALPHA_CONDITIONING,
            0.5,
        )

    with pytest.raises(ValueError, match="physical_alpha"):
        _validate_difficulty_measure_args(
            "canonical_alpha",
            RAW_T_CONDITIONING,
            0.5,
        )

    assert _validate_difficulty_measure_args(
        "canonical_alpha",
        PHYSICAL_ALPHA_CONDITIONING,
        0.5,
    ) == 0.5


def test_difficulty_mix_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="0 <= eta <= 1"):
        difficulty_mixture_schedule(-0.01)
    with pytest.raises(ValueError, match="0 <= eta <= 1"):
        difficulty_mixture_schedule(1.01)


def test_pilot_cli_accepts_difficulty_mix():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-name",
            "unit-test",
            "--schedule",
            "canonical_alpha",
            "--conditioning",
            "physical_alpha",
            "--difficulty-mix",
            "0.5",
        ]
    )
    assert args.difficulty_mix == 0.5
