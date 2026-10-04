import math

import pytest
import torch

from mscloudr.cli.pilot_train import (
    _endpoint_measure_metadata,
    _endpoint_mixture_schedule,
    _validate_endpoint_measure_args,
    build_parser,
)
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING


def test_endpoint_mixture_quantile_map_lambda_025():
    schedule = _endpoint_mixture_schedule(0.25)
    t = torch.tensor([0.0, 375.0, 749.0, 750.0, 1000.0])
    alpha = schedule(t, 1000)

    expected = torch.tensor(
        [
            0.0,
            0.5,
            749.0 / 750.0,
            1.0,
            1.0,
        ]
    )
    assert torch.allclose(alpha, expected, atol=1e-7)


def test_endpoint_mixture_preserves_bridge_endpoints():
    schedule = _endpoint_mixture_schedule(0.10)
    alpha = schedule(torch.tensor([0.0, 1000.0]), 1000)
    assert torch.allclose(alpha, torch.tensor([0.0, 1.0]))


def test_endpoint_mixture_requires_canonical_physical_alpha_pair():
    value = _validate_endpoint_measure_args(
        "canonical_alpha",
        PHYSICAL_ALPHA_CONDITIONING,
        0.25,
    )
    assert value == 0.25

    with pytest.raises(ValueError, match="canonical_alpha"):
        _validate_endpoint_measure_args(
            "original",
            PHYSICAL_ALPHA_CONDITIONING,
            0.25,
        )

    with pytest.raises(ValueError, match="physical_alpha"):
        _validate_endpoint_measure_args(
            "canonical_alpha",
            "raw_t",
            0.25,
        )


def test_endpoint_probability_bounds():
    with pytest.raises(ValueError, match="0 <= lambda < 1"):
        _validate_endpoint_measure_args(
            "canonical_alpha",
            PHYSICAL_ALPHA_CONDITIONING,
            -0.01,
        )

    with pytest.raises(ValueError, match="0 <= lambda < 1"):
        _validate_endpoint_measure_args(
            "canonical_alpha",
            PHYSICAL_ALPHA_CONDITIONING,
            1.0,
        )


def test_lambda_zero_keeps_existing_controls_valid():
    assert (
        _validate_endpoint_measure_args(
            "original",
            "raw_t",
            0.0,
        )
        == 0.0
    )


def test_endpoint_measure_metadata_records_discrete_mass():
    metadata = _endpoint_measure_metadata(
        endpoint_probability=0.25,
        total_steps=1000,
    )

    # t >= 750 is assigned exactly to alpha=1: 251 of 1001 grid points.
    assert metadata["endpoint_grid_count"] == 251
    assert metadata["endpoint_first_discrete_t"] == 750
    assert math.isclose(
        metadata["endpoint_probability_realized_discrete"],
        251.0 / 1001.0,
    )
    assert metadata["deployment_aware_measure"] is True


def test_pilot_cli_accepts_endpoint_probability():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-name",
            "unit-test",
            "--schedule",
            "canonical_alpha",
            "--conditioning",
            "physical_alpha",
            "--endpoint-prob",
            "0.25",
        ]
    )

    assert args.schedule == "canonical_alpha"
    assert args.conditioning == PHYSICAL_ALPHA_CONDITIONING
    assert args.endpoint_prob == 0.25
