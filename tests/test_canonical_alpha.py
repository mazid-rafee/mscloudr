import torch

from mscloudr.bridge import get_bridge_schedule, linear_alpha
from mscloudr.cli import fixed_alpha_sweep, pilot_train


def test_linear_alpha_is_physical_corruption_coordinate():
    t = torch.tensor([0.0, 250.0, 500.0, 750.0, 1000.0])
    alpha = linear_alpha(t, 1000)

    assert torch.allclose(
        alpha,
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
    )


def test_canonical_alpha_schedule_alias_resolves_linear_coordinate():
    schedule = get_bridge_schedule("canonical_alpha")
    t = torch.tensor([125.0, 625.0])

    assert torch.allclose(
        schedule(t, 1000),
        linear_alpha(t, 1000),
    )


def test_pilot_parser_accepts_canonical_alpha():
    args = pilot_train.build_parser().parse_args(
        [
            "--run-name",
            "canonical_alpha_test",
            "--schedule",
            "canonical_alpha",
        ]
    )

    assert args.schedule == "canonical_alpha"
    assert args.epochs == 25


def test_fixed_alpha_inverse_for_canonical_coordinate_is_identity():
    assert fixed_alpha_sweep.normalized_time_from_alpha(
        0.37,
        "canonical_alpha",
    ) == 0.37

    conditioning = fixed_alpha_sweep.conditioning_for_alpha(
        0.5,
        schedule_name="canonical_alpha",
        total_steps=1000,
    )

    assert conditioning["t_rounded"] == 500
    assert abs(conditioning["conditioning_alpha"] - 0.5) < 1e-7
    assert abs(conditioning["conditioning_alpha_error"]) < 1e-7
