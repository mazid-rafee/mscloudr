import math

import pytest
import torch

from mscloudr.cli.fixed_alpha_sweep import conditioning_for_alpha
from mscloudr.cli.pilot_train import (
    _conditioning_metadata,
    _validate_conditioning_schedule_pair,
    build_parser,
)
from mscloudr.training import (
    INVERSE_MR_CONDITIONING,
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


def test_inverse_mr_conditioning_matches_r3_schedule_inverse():
    total_steps = 1000
    alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    timesteps = torch.tensor([0, 500, 1000], dtype=torch.long)

    coordinate = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=INVERSE_MR_CONDITIONING,
    )

    rate = 3.0
    expected_mid = (
        -math.log(1.0 - (1.0 - math.exp(-rate)) * 0.5)
        / rate
        * total_steps
    )
    expected = torch.tensor(
        [0.0, expected_mid, float(total_steps)],
        dtype=torch.float32,
    )
    assert torch.allclose(coordinate, expected, atol=1e-4)


def test_inverse_mr_differs_from_physical_alpha_inside_bridge():
    total_steps = 1000
    alpha = torch.tensor([0.5], dtype=torch.float32)
    timesteps = torch.tensor([500], dtype=torch.long)

    mr_coordinate = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=INVERSE_MR_CONDITIONING,
    )
    physical_coordinate = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )

    assert float(physical_coordinate[0]) == 500.0
    assert not torch.allclose(mr_coordinate, physical_coordinate)


def test_inverse_mr_endpoints_preserve_legacy_conditioning_range():
    total_steps = 1000
    alpha = torch.tensor([0.0, 1.0], dtype=torch.float32)
    timesteps = torch.tensor([0, 1000], dtype=torch.long)

    coordinate = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=INVERSE_MR_CONDITIONING,
    )

    assert torch.allclose(
        coordinate,
        torch.tensor([0.0, 1000.0]),
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "mode",
    [INVERSE_SINE_CONDITIONING, INVERSE_MR_CONDITIONING],
)
def test_inverse_controls_require_uniform_physical_alpha_schedule(mode):
    _validate_conditioning_schedule_pair(
        "canonical_alpha",
        mode,
    )

    with pytest.raises(ValueError, match="requires --schedule canonical_alpha"):
        _validate_conditioning_schedule_pair(
            "original",
            mode,
        )

    with pytest.raises(ValueError, match="requires --schedule canonical_alpha"):
        _validate_conditioning_schedule_pair(
            "mr_r3",
            mode,
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


def test_pilot_cli_accepts_uniform_alpha_mr_coordinate_control():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-name",
            "unit-test",
            "--schedule",
            "canonical_alpha",
            "--conditioning",
            "inverse_mr",
        ]
    )

    assert args.schedule == "canonical_alpha"
    assert args.conditioning == INVERSE_MR_CONDITIONING


def test_inverse_sine_metadata_records_matched_measure_control():
    metadata = _conditioning_metadata(INVERSE_SINE_CONDITIONING)

    assert metadata["conditioning_mode"] == INVERSE_SINE_CONDITIONING
    assert metadata["conditioning_definition"] == "c=T*(2/pi)*asin(alpha)"
    assert metadata["matched_measure_control"] is True
    assert metadata["conditioning_reference_schedule"] == "original"
    assert metadata["coordinate_invariant_conditioning"] is False


def test_inverse_mr_metadata_records_matched_measure_control():
    metadata = _conditioning_metadata(INVERSE_MR_CONDITIONING)

    assert metadata["conditioning_mode"] == INVERSE_MR_CONDITIONING
    assert metadata["conditioning_definition"] == (
        "c=-(T/3)*log(1-(1-exp(-3))*alpha)"
    )
    assert metadata["matched_measure_control"] is True
    assert metadata["conditioning_reference_schedule"] == "mr_r3"
    assert metadata["conditioning_reference_mean_reversion_rate"] == 3.0
    assert metadata["coordinate_invariant_conditioning"] is False


def test_fixed_alpha_sweep_uses_inverse_sine_coordinate():
    result = conditioning_for_alpha(
        0.5,
        schedule_name="canonical_alpha",
        total_steps=1000,
        conditioning_mode=INVERSE_SINE_CONDITIONING,
    )

    assert result["model_conditioning_mode"] == INVERSE_SINE_CONDITIONING
    assert abs(float(result["model_conditioning_value"]) - 1000.0 / 3.0) < 1e-9
    assert result["conditioning_alpha"] == 0.5
    assert result["conditioning_alpha_error"] == 0.0
