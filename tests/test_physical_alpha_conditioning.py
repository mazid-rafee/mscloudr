import math

import torch

from mscloudr.bridge import get_bridge_schedule
from mscloudr.cli.fixed_alpha_sweep import conditioning_for_alpha
from mscloudr.cli.pilot_train import build_parser
from mscloudr.training import (
    PHYSICAL_ALPHA_CONDITIONING,
    RAW_T_CONDITIONING,
    model_conditioning_coordinate,
)


def test_physical_alpha_conditioning_uses_actual_bridge_state_not_raw_t():
    total_steps = 1000
    timesteps = torch.tensor([0, 250, 500, 750, 1000], dtype=torch.long)
    schedule = get_bridge_schedule("original")
    alpha = schedule(timesteps.float(), total_steps)

    raw = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=RAW_T_CONDITIONING,
    )
    physical = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )

    assert torch.equal(raw, timesteps)
    assert torch.allclose(
        physical,
        alpha * total_steps,
        atol=1e-6,
    )
    # At t=500 the sine schedule is alpha=sqrt(1/2), so the physical
    # coordinate is about 707 rather than the arbitrary schedule parameter 500.
    assert abs(float(physical[2]) - 1000.0 / math.sqrt(2.0)) < 1e-3
    assert float(physical[2]) != float(raw[2])


def test_equal_physical_alpha_gives_equal_conditioning_across_schedules():
    total_steps = 1000
    alpha = torch.tensor([0.8, 0.8], dtype=torch.float32)
    different_raw_timesteps = torch.tensor([590, 476], dtype=torch.long)

    physical = model_conditioning_coordinate(
        different_raw_timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )

    assert torch.allclose(physical, torch.tensor([800.0, 800.0]))


def test_endpoint_coordinate_is_identical_for_raw_t_and_physical_alpha():
    total_steps = 1000
    timesteps = torch.tensor([1000, 1000], dtype=torch.long)
    alpha = torch.ones(2)

    raw = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=RAW_T_CONDITIONING,
    ).float()
    physical = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )

    assert torch.allclose(raw, physical)
    assert torch.allclose(physical, torch.tensor([1000.0, 1000.0]))


def test_fixed_alpha_diagnostic_uses_c_equal_T_alpha_for_physical_conditioning():
    result = conditioning_for_alpha(
        0.8,
        schedule_name="original",
        total_steps=1000,
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )

    assert result["model_conditioning_mode"] == PHYSICAL_ALPHA_CONDITIONING
    assert result["model_conditioning_value"] == 800.0
    assert result["conditioning_alpha"] == 0.8
    assert result["conditioning_alpha_error"] == 0.0
    # The underlying sine schedule reaches alpha=0.8 at a different raw t.
    assert result["t_rounded"] != 800


def test_pilot_cli_accepts_physical_alpha_conditioning():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-name",
            "unit-test",
            "--schedule",
            "original",
            "--conditioning",
            "physical_alpha",
        ]
    )

    assert args.schedule == "original"
    assert args.conditioning == PHYSICAL_ALPHA_CONDITIONING
