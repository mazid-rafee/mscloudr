"""Gradient-informed physical-alpha training measure for DB-CR pilots.

The calibration below comes from the fixed-alpha gradient diagnostic on the
CanonicalAlpha pilot checkpoint (seed 42, epoch 20).  At alpha knots
0.0,0.1,...,1.0, the diagnostic measured the RMS batch gradient norm

    G_rms(alpha) = sqrt(E_B[||grad_theta L1_B(alpha)||_2^2]).

For training we form a continuous, coverage-preserving density

    q_eta(alpha) = (1-eta) * U(0,1)
                   + eta * G_tilde(alpha),

where G_tilde is the piecewise-linear interpolation of the measured RMS
profile normalized to integrate to one.  Sampling is implemented as a
quantile transform of the existing uniform discrete timestep variable, so the
sampler RNG stream stays deterministic and the bridge remains in physical
alpha coordinates.
"""

from __future__ import annotations

import math

import torch


GRADIENT_PROFILE_ALPHAS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)

GRADIENT_PROFILE_RMS = (
    0.7596105044880722,
    1.4000102998106325,
    0.6307097292582952,
    0.8941610073875346,
    0.5173143302774076,
    0.5506940300344573,
    0.8532423595470414,
    0.7494845423788181,
    1.0748238656057805,
    0.6329137550519158,
    0.8148502750040192,
)

GRADIENT_PROFILE_PROVENANCE = {
    "diagnostic": "fixed_physical_alpha_gradient_profile",
    "diagnostic_branch": "exp/gradient-alpha-diagnostic",
    "diagnostic_commit": "d324a6264b535b9bb3fe94cb8e48b0d827833c11",
    "source_run": "DBCR_CanonicalAlpha_pilot10_seed42",
    "source_checkpoint": "best_endpoint.pt",
    "source_checkpoint_epoch": 20,
    "pilot_protocol": "uncrtaints_roi_disjoint_pilot10_roi_season_stratified_v1",
    "profile_statistic": "batch_gradient_l2.rms",
    "profile_batches_per_alpha": 256,
    "profile_samples_per_alpha": 1024,
}

DEFAULT_GRADIENT_MIX = 0.5


def _validated_mix(gradient_mix: float) -> float:
    eta = float(gradient_mix)
    if not math.isfinite(eta):
        raise ValueError("gradient-mix must be finite")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("gradient-mix must satisfy 0 <= eta <= 1")
    return eta


def _density_knots(gradient_mix: float) -> tuple[list[float], list[float]]:
    """Return alpha knots and normalized mixed-density values at those knots."""

    eta = _validated_mix(gradient_mix)
    alphas = list(GRADIENT_PROFILE_ALPHAS)
    scores = list(GRADIENT_PROFILE_RMS)

    area = 0.0
    for i in range(len(alphas) - 1):
        width = alphas[i + 1] - alphas[i]
        area += 0.5 * width * (scores[i] + scores[i + 1])
    if not area > 0.0:
        raise RuntimeError("gradient calibration integral must be positive")

    normalized = [score / area for score in scores]
    density = [(1.0 - eta) + eta * value for value in normalized]
    return alphas, density


def gradient_measure_summary(gradient_mix: float = DEFAULT_GRADIENT_MIX) -> dict:
    """Return reproducibility metadata for the gradient-informed measure."""

    eta = _validated_mix(gradient_mix)
    alphas, density = _density_knots(eta)

    segment_masses: list[float] = []
    for i in range(len(alphas) - 1):
        width = alphas[i + 1] - alphas[i]
        segment_masses.append(0.5 * width * (density[i] + density[i + 1]))

    total = sum(segment_masses)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        segment_masses = [mass / total for mass in segment_masses]

    return {
        "training_bridge_measure": "gradient_rms_mixture_physical_alpha",
        "training_bridge_measure_definition": (
            "q_eta(alpha)=(1-eta)U(0,1)+eta*G_tilde_rms(alpha), where "
            "G_tilde_rms is the normalized piecewise-linear interpolation of "
            "the fixed-alpha RMS batch-gradient profile"
        ),
        "alpha_sampling_identity": "gradient_profile_quantile_transform",
        "gradient_mix_eta": eta,
        "gradient_profile_alphas": list(GRADIENT_PROFILE_ALPHAS),
        "gradient_profile_rms": list(GRADIENT_PROFILE_RMS),
        "gradient_density_knots": density,
        "gradient_segment_masses": segment_masses,
        "base_measure": "uniform_physical_alpha",
        "coverage_preserving_uniform_component": 1.0 - eta,
        "gradient_profile_provenance": dict(GRADIENT_PROFILE_PROVENANCE),
    }


def gradient_mixture_schedule(gradient_mix: float = DEFAULT_GRADIENT_MIX):
    """Create the inverse-CDF map for the continuous gradient-informed density.

    The returned callable has the standard bridge-schedule signature
    ``schedule(t, total_steps)``.  It interprets ``u=t/T`` as a base uniform
    quantile and maps it through the inverse CDF of the piecewise-linear mixed
    density.  With eta=0 this reduces exactly to alpha=t/T.
    """

    eta = _validated_mix(gradient_mix)
    alphas, density = _density_knots(eta)

    segment_masses: list[float] = []
    cdf = [0.0]
    for i in range(len(alphas) - 1):
        width = alphas[i + 1] - alphas[i]
        mass = 0.5 * width * (density[i] + density[i + 1])
        segment_masses.append(mass)
        cdf.append(cdf[-1] + mass)

    total_mass = cdf[-1]
    if not total_mass > 0.0:
        raise RuntimeError("gradient-informed density must have positive mass")
    segment_masses = [mass / total_mass for mass in segment_masses]
    cdf = [0.0]
    for mass in segment_masses:
        cdf.append(cdf[-1] + mass)
    cdf[-1] = 1.0

    def schedule(t, total_steps: int):
        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")

        t_float = t if torch.is_tensor(t) else torch.as_tensor(t)
        if not t_float.is_floating_point():
            t_float = t_float.float()
        u = torch.clamp(t_float / float(total_steps), 0.0, 1.0)

        dtype = u.dtype
        device = u.device
        cdf_tensor = torch.tensor(cdf, dtype=dtype, device=device)
        alpha_tensor = torch.tensor(alphas, dtype=dtype, device=device)
        density_tensor = torch.tensor(density, dtype=dtype, device=device)

        # bucketize against interior CDF boundaries gives segment indices 0..9.
        segment = torch.bucketize(u, cdf_tensor[1:-1], right=False)
        left_cdf = cdf_tensor[segment]
        left_alpha = alpha_tensor[segment]
        right_alpha = alpha_tensor[segment + 1]
        left_density = density_tensor[segment] / float(total_mass)
        right_density = density_tensor[segment + 1] / float(total_mass)

        width = right_alpha - left_alpha
        slope = (right_density - left_density) / width
        residual_mass = u - left_cdf

        # Solve residual_mass = A*x + 0.5*B*x^2.  The alternate quadratic
        # form below is numerically stable when B is small or negative.
        discriminant = torch.clamp(
            left_density * left_density + 2.0 * slope * residual_mass,
            min=0.0,
        )
        sqrt_disc = torch.sqrt(discriminant)
        denom = left_density + sqrt_disc
        curved_x = torch.where(
            denom > 0,
            2.0 * residual_mass / denom,
            torch.zeros_like(residual_mass),
        )
        flat_x = residual_mass / left_density
        local_x = torch.where(torch.abs(slope) < 1e-8, flat_x, curved_x)
        local_x = torch.clamp(local_x, min=0.0)
        local_x = torch.minimum(local_x, width)

        alpha = left_alpha + local_x
        alpha = torch.where(u <= 0.0, torch.zeros_like(alpha), alpha)
        alpha = torch.where(u >= 1.0, torch.ones_like(alpha), alpha)
        return torch.clamp(alpha, 0.0, 1.0)

    return schedule
