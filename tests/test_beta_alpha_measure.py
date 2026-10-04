import math

import pytest
import torch

from mscloudr.cli.pilot_train import (
    _beta_a1_measure_metadata,
    _beta_a1_schedule,
    _validate_measure_args,
    build_parser,
)
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING


def test_beta_a2_quantile_transform_matches_sqrt_uniform_coordinate():
    schedule = _beta_a1_schedule(2.0)
    t = torch.tensor([0.0, 250.0, 1000.0])
    alpha = schedule(t, 1000)
    expected = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(alpha, expected, atol=1e-7)


def test_beta_a4_quantile_transform_is_fourth_root():
    schedule = _beta_a1_schedule(4.0)
    t = torch.tensor([0.0, 62.5, 1000.0])
    alpha = schedule(t, 1000)
    expected = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(alpha, expected, atol=1e-7)


def test_beta_a1_reduces_to_uniform_physical_alpha():
    schedule = _beta_a1_schedule(1.0)
    t = torch.tensor([0.0, 100.0, 500.0, 1000.0])
    assert torch.allclose(schedule(t, 1000), t / 1000.0)


def test_beta_measure_requires_canonical_schedule_and_physical_conditioning():
    _validate_measure_args(
        "canonical_alpha",
        PHYSICAL_ALPHA_CONDITIONING,
        endpoint_probability=0.0,
        beta_a=2.0,
    )

    with pytest.raises(ValueError, match="beta-a != 1 requires --schedule canonical_alpha"):
        _validate_measure_args(
            "original",
            PHYSICAL_ALPHA_CONDITIONING,
            endpoint_probability=0.0,
            beta_a=2.0,
        )

    with pytest.raises(ValueError, match="beta-a != 1 requires --conditioning physical_alpha"):
        _validate_measure_args(
            "canonical_alpha",
            "raw_t",
            endpoint_probability=0.0,
            beta_a=2.0,
        )


def test_beta_measure_and_endpoint_mixture_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_measure_args(
            "canonical_alpha",
            PHYSICAL_ALPHA_CONDITIONING,
            endpoint_probability=0.25,
            beta_a=2.0,
        )


def test_beta_a_must_be_positive():
    with pytest.raises(ValueError, match="beta-a must be positive"):
        _validate_measure_args(
            "canonical_alpha",
            PHYSICAL_ALPHA_CONDITIONING,
            endpoint_probability=0.0,
            beta_a=0.0,
        )


def test_beta_metadata_records_smooth_endpoint_bias():
    metadata = _beta_a1_measure_metadata(beta_a=2.0)
    assert metadata["training_bridge_measure"] == "beta_a1_physical_alpha"
    assert metadata["alpha_sampling_identity"] == "beta_a1_quantile_transform"
    assert metadata["beta_a"] == 2.0
    assert metadata["beta_b"] == 1.0
    assert metadata["training_bridge_measure_density"] == "q(alpha)=a*alpha^(a-1)"
    assert metadata["smooth_endpoint_biased_measure"] is True


def test_pilot_cli_accepts_beta_a2_control():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-name",
            "unit-test",
            "--schedule",
            "canonical_alpha",
            "--conditioning",
            "physical_alpha",
            "--beta-a",
            "2.0",
        ]
    )
    assert math.isclose(args.beta_a, 2.0)
