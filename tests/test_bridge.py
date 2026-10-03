import torch

from mscloudr.bridge import (
    deterministic_reverse_step,
    get_bridge_schedule,
    make_bridge_state,
    make_reverse_timesteps,
    mean_reverting_alpha,
    reverse_step_from_times,
    sample_timesteps,
    sine_alpha,
)
from mscloudr.reproducibility import make_torch_generator


def test_sine_schedule_has_bridge_endpoints():
    t = torch.tensor([0.0, 1000.0])
    alpha = sine_alpha(t, 1000)

    assert abs(float(alpha[0])) < 1e-7
    assert abs(float(alpha[1]) - 1.0) < 1e-6


def test_mr_r3_schedule_has_bridge_endpoints_and_is_monotone():
    t = torch.linspace(0, 1000, 101)
    alpha = mean_reverting_alpha(
        t,
        1000,
        rate=3.0,
    )

    assert abs(float(alpha[0])) < 1e-7
    assert abs(float(alpha[-1]) - 1.0) < 1e-6
    assert torch.all(
        alpha[1:] >= alpha[:-1]
    )


def test_mr_schedule_rejects_nonpositive_rate():
    try:
        mean_reverting_alpha(
            torch.tensor([1.0]),
            1000,
            rate=0.0,
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError(
            "expected ValueError"
        )


def test_bridge_state_recovers_clean_and_cloudy_endpoints():
    torch.manual_seed(0)
    clean = torch.randn(
        2,
        13,
        8,
        8,
    )
    cloudy = torch.randn_like(clean)

    at_zero = make_bridge_state(
        clean,
        cloudy,
        torch.zeros(2),
    )
    at_one = make_bridge_state(
        clean,
        cloudy,
        torch.ones(2),
    )

    assert torch.equal(
        at_zero,
        clean,
    )
    assert torch.equal(
        at_one,
        cloudy,
    )


def test_explicit_sampler_generator_is_reproducible_and_global_rng_independent():
    first_generator = make_torch_generator(
        42
    )
    second_generator = make_torch_generator(
        42
    )

    first = sample_timesteps(
        128,
        1000,
        generator=first_generator,
    )

    # Consume the global RNG. This must not alter the explicit sampler stream.
    torch.manual_seed(999)
    _ = torch.rand(1000)

    second = sample_timesteps(
        128,
        1000,
        generator=second_generator,
    )

    assert torch.equal(
        first,
        second,
    )
    assert first.dtype == torch.long
    assert int(first.min()) >= 0
    assert int(first.max()) <= 1000


def test_reverse_grid_has_requested_nfe_and_exact_endpoints():
    timesteps = make_reverse_timesteps(
        1000,
        5,
    )

    assert len(timesteps) == 6
    assert timesteps[0].item() == 1000
    assert timesteps[-1].item() == 0
    assert torch.all(
        timesteps[1:]
        <= timesteps[:-1]
    )


def test_reverse_step_with_same_alpha_is_identity_on_current_state():
    torch.manual_seed(0)
    x_t = torch.randn(
        2,
        13,
        8,
        8,
    )
    x0_hat = torch.randn_like(x_t)

    output = deterministic_reverse_step(
        x_t,
        x0_hat,
        alpha_t=torch.tensor(
            [0.5, 0.8]
        ),
        alpha_s=torch.tensor(
            [0.5, 0.8]
        ),
    )

    assert torch.allclose(
        output,
        x_t,
    )


def test_nfe1_endpoint_map_is_schedule_independent_for_sine_and_mr():
    torch.manual_seed(0)
    cloudy = torch.randn(
        2,
        13,
        8,
        8,
    )
    x0_hat = torch.randn_like(cloudy)

    sine_output = reverse_step_from_times(
        cloudy,
        x0_hat,
        t_current=torch.tensor(
            [1000.0, 1000.0]
        ),
        t_next=torch.tensor(
            [0.0, 0.0]
        ),
        total_steps=1000,
        schedule=sine_alpha,
    )

    mr_schedule = get_bridge_schedule(
        "mr_r3",
        mean_reversion_rate=3.0,
    )
    mr_output = reverse_step_from_times(
        cloudy,
        x0_hat,
        t_current=torch.tensor(
            [1000.0, 1000.0]
        ),
        t_next=torch.tensor(
            [0.0, 0.0]
        ),
        total_steps=1000,
        schedule=mr_schedule,
    )

    assert torch.allclose(
        sine_output,
        x0_hat,
        atol=1e-6,
    )
    assert torch.allclose(
        mr_output,
        x0_hat,
        atol=1e-6,
    )
    assert torch.allclose(
        sine_output,
        mr_output,
        atol=1e-6,
    )


def test_schedule_aliases_resolve_expected_functions():
    t = torch.tensor(
        [250.0]
    )

    original = get_bridge_schedule(
        "original"
    )
    mr = get_bridge_schedule(
        "mean_reverting",
        mean_reversion_rate=3.0,
    )

    assert torch.allclose(
        original(t, 1000),
        sine_alpha(t, 1000),
    )
    assert torch.allclose(
        mr(t, 1000),
        mean_reverting_alpha(
            t,
            1000,
            rate=3.0,
        ),
    )
