import math

import pytest
import torch

from mscloudr.cli.pilot_train import (
    _conditioning_metadata,
    _validate_conditioning_schedule_pair,
    build_parser,
)
from mscloudr.training import (
    INVERSE_SINE_CONDITIONING,
    PHYSICAL_ALPHA_CONDITIONING,
    model_conditioning_coordinate,
)


def test_inverse_sine_conditioning_matches_original_schedule_inverse():
    total_steps = 1000
    alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    timesteps = torch.tensor([0, 500, 1000], dtype=torch.long)

    coordinate = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=INVERSE_SINE_CONDITIONING,
    )

    expected = torch.tensor(
        [
            0.0,
            total_steps / 3.0,
            float(total_steps),
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(coordinate, expected, atol=1e-4)


def test_inverse_sine_differs_from_physical_alpha_inside_bridge():
    total_steps = 1000
    alpha = torch.tensor([0.5], dtype=torch.float32)
    timesteps = torch.tensor([500], dtype=torch.long)

    sine_coordinate = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=INVERSE_SINE_CONDITIONING,
    )
    physical_coordinate = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )

    assert abs(float(sine_coordinate[0]) - 1000.0 / 3.0) < 1e-3
    assert float(physical_coordinate[0]) == 500.0
    assert not torch.allclose(sine_coordinate, physical_coordinate)


def test_inverse_sine_endpoints_preserve_legacy_conditioning_range():
    total_steps = 1000
    alpha = torch.tensor([0.0, 1.0], dtype=torch.float32)
    timesteps = torch.tensor([0, 1000], dtype=torch.long)

    coordinate = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=INVERSE_SINE_CONDITIONING,
    )

    assert torch.allclose(coordinate, torch.tensor([0.0, 1000.0]))


def test_inverse_sine_requires_uniform_physical_alpha_schedule():
    _validate_conditioning_schedule_pair(
        "canonical_alpha",
        INVERSE_SINE_CONDITIONING,
    )

    with pytest.raises(ValueError, match="requires --schedule canonical_alpha"):
        _validate_conditioning_schedule_pair(
            "original",
            INVERSE_SINE_CONDITIONING,
        )

    with pytest.raises(ValueError, match="requires --schedule canonical_alpha"):
        _validate_conditioning_schedule_pair(
            "mr_r3",
            INVERSE_SINE_CONDITIONING,
        )


def test_pilot_cli_accepts_uniform_alpha_sine_coordinate_control():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-name",
            "unit-test",
            "--schedule",
            "canonical_alpha",
            "--conditioning",
            "inverse_sine",
        ]
    )

    assert args.schedule == "canonical_alpha"
    assert args.conditioning == INVERSE_SINE_CONDITIONING


def test_inverse_sine_metadata_records_matched_measure_control():
    metadata = _conditioning_metadata(INVERSE_SINE_CONDITIONING)

    assert metadata["conditioning_mode"] == INVERSE_SINE_CONDITIONING
    assert metadata["conditioning_definition"] == "c=T*(2/pi)*asin(alpha)"
    assert metadata["matched_measure_control"] is True
    assert metadata["conditioning_reference_schedule"] == "original"
    assert metadata["coordinate_invariant_conditioning"] is False
