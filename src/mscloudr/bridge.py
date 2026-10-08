"""Diffusion-bridge mechanics for controlled DB-CR experiments.

The module keeps bridge geometry separate from the neural network so schedules
can be compared without changing the backbone.

For cloudy optical y, clean target x0, and bridge weight alpha:

    x_t = (1 - alpha_t) * x0 + alpha_t * y

The deterministic DB-CR reverse update from t to s is:

    x_s = (1 - alpha_s / alpha_t) * x0_hat
          + (alpha_s / alpha_t) * x_t

All stochastic timestep sampling requires an explicit torch.Generator.
"""

from __future__ import annotations

import math
from typing import Callable

import torch


BridgeSchedule = Callable[[torch.Tensor, int], torch.Tensor]


def _float_tensor(
    value,
    *,
    device=None,
) -> torch.Tensor:
    tensor = (
        value
        if torch.is_tensor(value)
        else torch.as_tensor(value)
    )
    if not tensor.is_floating_point():
        tensor = tensor.float()
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _validate_total_steps(total_steps: int) -> int:
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive"
        )
    return total_steps


def linear_alpha(
    t,
    total_steps: int,
) -> torch.Tensor:
    """Canonical physical-corruption coordinate alpha=t/T.

    Combined with the historical uniform discrete timestep sampler, this makes
    the training measure uniform over the physical bridge grid
    {0, 1/T, ..., 1}.  Unlike sine or mean-reverting parameterizations, the
    conditioning coordinate and physical corruption coordinate are identical
    up to the scale factor T.
    """

    total_steps = _validate_total_steps(
        total_steps
    )
    t = _float_tensor(t)
    return t / float(total_steps)


def sine_alpha(
    t,
    total_steps: int,
) -> torch.Tensor:
    """Original DB-CR sine bridge schedule."""

    total_steps = _validate_total_steps(
        total_steps
    )
    t = _float_tensor(t)
    return torch.sin(
        (t / float(total_steps))
        * (math.pi / 2.0)
    )


def mean_reverting_alpha(
    t,
    total_steps: int,
    *,
    rate: float = 3.0,
) -> torch.Tensor:
    """Normalized deterministic mean-reverting schedule.

    alpha(s) = (1 - exp(-rate * s))
               / (1 - exp(-rate)),
    where s = t / T.

    The expm1 form improves numerical stability and preserves the intended
    alpha(0)=0 and alpha(T)=1 endpoints up to floating-point precision.
    """

    total_steps = _validate_total_steps(
        total_steps
    )
    rate = float(rate)
    if rate <= 0.0:
        raise ValueError(
            "rate must be positive"
        )

    t = _float_tensor(t)
    normalized_time = (
        t / float(total_steps)
    )
    rate_tensor = torch.as_tensor(
        rate,
        dtype=t.dtype,
        device=t.device,
    )
    return (
        torch.expm1(
            -rate_tensor * normalized_time
        )
        / torch.expm1(-rate_tensor)
    )


def get_bridge_schedule(
    name: str,
    *,
    mean_reversion_rate: float = 3.0,
) -> BridgeSchedule:
    """Resolve a named parameterization without touching model code."""

    normalized = (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
    )

    if normalized in {
        "canonical_alpha",
        "canonical",
        "linear",
        "uniform_alpha",
    }:
        return linear_alpha

    if normalized in {
        "original",
        "sine",
        "dbcr",
    }:
        return sine_alpha

    if normalized in {
        "mean_reverting",
        "mr",
        "mr_r3",
    }:
        rate = float(
            mean_reversion_rate
        )

        def schedule(
            t,
            total_steps: int,
        ) -> torch.Tensor:
            return mean_reverting_alpha(
                t,
                total_steps,
                rate=rate,
            )

        return schedule

    raise ValueError(
        f"unknown bridge schedule: {name!r}"
    )


def _as_batch_coefficient(
    value,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    coefficient = _float_tensor(
        value,
        device=reference.device,
    ).to(dtype=reference.dtype)

    if coefficient.ndim == 0:
        return coefficient

    if coefficient.ndim == 1:
        if coefficient.shape[0] not in {
            1,
            reference.shape[0],
        }:
            raise ValueError(
                "batch coefficient length must be 1 "
                "or equal the reference batch size"
            )
        return coefficient.view(
            -1,
            1,
            1,
            1,
        )

    if coefficient.ndim == 4:
        return coefficient

    raise ValueError(
        "coefficient must be scalar, [B], "
        "or broadcastable [B,1,1,1]"
    )


def make_bridge_state(
    clean: torch.Tensor,
    cloudy: torch.Tensor,
    alpha,
) -> torch.Tensor:
    """Construct x_t on the straight cloudy-to-clean bridge."""

    if clean.shape != cloudy.shape:
        raise ValueError(
            "clean and cloudy tensors must have identical shapes"
        )
    if clean.ndim != 4:
        raise ValueError(
            "clean and cloudy must have shape [B,C,H,W]"
        )

    alpha = _as_batch_coefficient(
        alpha,
        reference=clean,
    )
    return (
        (1.0 - alpha) * clean
        + alpha * cloudy
    )


def make_sar_curved_bridge_state(
    x0_hat: torch.Tensor,
    cloudy: torch.Tensor,
    curvature: torch.Tensor,
    alpha,
) -> torch.Tensor:
    """Project a clean prediction onto the SAR-curved bridge at alpha.

    The curvature input is the model-predicted optical-space field phi(y,z),
    already bounded and scaled by kappa. The endpoint-preserving envelope
    h(alpha)=4*alpha*(1-alpha) matches training exactly.
    """

    if x0_hat.shape != cloudy.shape or curvature.shape != cloudy.shape:
        raise ValueError(
            "x0_hat, cloudy, and curvature must have identical shapes"
        )
    if x0_hat.ndim != 4:
        raise ValueError(
            "x0_hat, cloudy, and curvature must have shape [B,C,H,W]"
        )

    alpha = _as_batch_coefficient(
        alpha,
        reference=x0_hat,
    )
    envelope = 4.0 * alpha * (1.0 - alpha)
    return (
        (1.0 - alpha) * x0_hat
        + alpha * cloudy
        + envelope * curvature
    )


def sample_timesteps(
    batch_size: int,
    total_steps: int,
    *,
    generator: torch.Generator,
    device=None,
) -> torch.Tensor:
    """Sample integer t uniformly from {0, ..., T} with explicit RNG.

    Sampling occurs on the generator's own device, then the result is moved to
    the requested target device. This keeps sampler randomness independent from
    the global PyTorch RNG stream.
    """

    batch_size = int(batch_size)
    total_steps = _validate_total_steps(
        total_steps
    )
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )
    if generator is None:
        raise ValueError(
            "an explicit torch.Generator is required"
        )

    sample_device = generator.device
    timesteps = torch.randint(
        low=0,
        high=total_steps + 1,
        size=(batch_size,),
        generator=generator,
        device=sample_device,
        dtype=torch.long,
    )
    if device is not None:
        timesteps = timesteps.to(
            device=device
        )
    return timesteps


def make_reverse_timesteps(
    total_steps: int,
    nfe: int,
    *,
    device=None,
) -> torch.Tensor:
    """Create the historical deterministic T -> 0 reverse-time grid."""

    total_steps = _validate_total_steps(
        total_steps
    )
    nfe = int(nfe)
    if nfe <= 0:
        raise ValueError(
            "nfe must be positive"
        )

    steps = torch.linspace(
        float(total_steps),
        0.0,
        nfe + 1,
        device=device,
    )
    timesteps = torch.round(
        steps
    ).to(torch.long)

    if timesteps[0].item() != total_steps:
        raise RuntimeError(
            "reverse grid must start at T"
        )
    if timesteps[-1].item() != 0:
        raise RuntimeError(
            "reverse grid must end at 0"
        )
    if torch.any(
        timesteps[1:] > timesteps[:-1]
    ):
        raise RuntimeError(
            "reverse grid must be non-increasing"
        )
    return timesteps


def deterministic_reverse_step(
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    *,
    alpha_t,
    alpha_s,
) -> torch.Tensor:
    """Apply one deterministic DB-CR reverse step."""

    if x_t.shape != x0_hat.shape:
        raise ValueError(
            "x_t and x0_hat must have identical shapes"
        )
    if x_t.ndim != 4:
        raise ValueError(
            "x_t and x0_hat must have shape [B,C,H,W]"
        )

    alpha_t = _as_batch_coefficient(
        alpha_t,
        reference=x_t,
    )
    alpha_s = _as_batch_coefficient(
        alpha_s,
        reference=x_t,
    )

    if torch.any(
        torch.abs(alpha_t)
        <= torch.finfo(x_t.dtype).eps
    ):
        raise ValueError(
            "alpha_t must be non-zero for a reverse step"
        )

    ratio = alpha_s / alpha_t
    return (
        (1.0 - ratio) * x0_hat
        + ratio * x_t
    )


def reverse_step_from_times(
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    *,
    t_current,
    t_next,
    total_steps: int,
    schedule: BridgeSchedule,
) -> torch.Tensor:
    """Resolve schedule weights and apply one deterministic reverse step."""

    alpha_t = schedule(
        _float_tensor(
            t_current,
            device=x_t.device,
        ),
        total_steps,
    )
    alpha_s = schedule(
        _float_tensor(
            t_next,
            device=x_t.device,
        ),
        total_steps,
    )
    return deterministic_reverse_step(
        x_t,
        x0_hat,
        alpha_t=alpha_t,
        alpha_s=alpha_s,
    )
