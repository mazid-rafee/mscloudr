"""Difficulty-adaptive physical-alpha training measure for DB-CR pilots.

The calibration below comes from the fixed-alpha training-difficulty diagnostic
on the CanonicalAlpha pilot checkpoint (seed 42, epoch 20). At alpha knots
0.0, 0.1, ..., 1.0, the diagnostic measured

    E(alpha) = mean_train L1(R_theta(x_alpha, T*alpha, z), x0).

For training we form a continuous, coverage-preserving density

    q_eta(alpha) = (1-eta) * U(0,1)
                   + eta * E_tilde(alpha),

where E_tilde is the piecewise-linear interpolation of the measured training
L1 profile normalized to integrate to one. Sampling is implemented as a
quantile transform of the existing uniform discrete timestep variable, so the
sampler RNG stream stays deterministic and the bridge remains in physical
alpha coordinates.
"""

from __future__ import annotations

import math

import torch


DIFFICULTY_PROFILE_ALPHAS = (
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

DIFFICULTY_PROFILE_TRAIN_L1 = (
    0.007730761459532764,
    0.008820822968999394,
    0.01028091533032008,
    0.01191121278243671,
    0.013563057973753853,
    0.015141403912052001,
    0.01688424975847534,
    0.018292789886083537,
    0.02033418242282232,
    0.023105324425534476,
    0.026150330238467415,
)

DIFFICULTY_PROFILE_PROVENANCE = {
    "diagnostic": "fixed_physical_alpha_train_difficulty_profile",
    "diagnostic_branch": "exp/difficulty-adaptive-measure",
    "diagnostic_commit": "54db89eb65e6ddf68b5b6a96d1fdaac9b955eba1",
    "source_run": "DBCR_CanonicalAlpha_pilot10_seed42",
    "source_checkpoint": "best_endpoint.pt",
    "source_checkpoint_epoch": 20,
    "pilot_protocol": "uncrtaints_roi_disjoint_pilot10_roi_season_stratified_v1",
    "profile_statistic": "fixed_alpha_train_mean_l1",
    "profile_batches_per_alpha": 2679,
    "profile_samples_per_alpha": 10714,
    "sampler_design_split": "train_only",
    "validation_used_for_sampler_design": False,
    "test_used_for_sampler_design": False,
}

DEFAULT_DIFFICULTY_MIX = 0.5


def _validated_mix(difficulty_mix: float) -> float:
    eta = float(difficulty_mix)
    if not math.isfinite(eta):
        raise ValueError("difficulty-mix must be finite")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("difficulty-mix must satisfy 0 <= eta <= 1")
    return eta


def _density_knots(difficulty_mix: float) -> tuple[list[float], list[float]]:
    """Return alpha knots and normalized mixed-density values at those knots."""

    eta = _validated_mix(difficulty_mix)
    alphas = list(DIFFICULTY_PROFILE_ALPHAS)
    scores = list(DIFFICULTY_PROFILE_TRAIN_L1)

    area = 0.0
    for i in range(len(alphas) - 1):
        width = alphas[i + 1] - alphas[i]
        area += 0.5 * width * (scores[i] + scores[i + 1])
    if not area > 0.0:
        raise RuntimeError("difficulty calibration integral must be positive")

    normalized = [score / area for score in scores]
    density = [(1.0 - eta) + eta * value for value in normalized]
    return alphas, density


def difficulty_measure_summary(difficulty_mix: float = DEFAULT_DIFFICULTY_MIX) -> dict:
    """Return reproducibility metadata for the difficulty-adaptive measure."""

    eta = _validated_mix(difficulty_mix)
    alphas, density = _density_knots(eta)

    segment_masses: list[float] = []
    for i in range(len(alphas) - 1):
        width = alphas[i + 1] - alphas[i]
        segment_masses.append(0.5 * width * (density[i] + density[i + 1]))

    total = sum(segment_masses)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        segment_masses = [mass / total for mass in segment_masses]

    return {
        "training_bridge_measure": "difficulty_l1_mixture_physical_alpha",
        "training_bridge_measure_definition": (
            "q_eta(alpha)=(1-eta)U(0,1)+eta*E_tilde_train(alpha), where "
            "E_tilde_train is the normalized piecewise-linear interpolation of "
            "the fixed-alpha train L1 difficulty profile"
        ),
        "alpha_sampling_identity": "difficulty_profile_quantile_transform",
        "difficulty_mix_eta": eta,
        "difficulty_profile_alphas": list(DIFFICULTY_PROFILE_ALPHAS),
        "difficulty_profile_train_l1": list(DIFFICULTY_PROFILE_TRAIN_L1),
        "difficulty_density_knots": density,
        "difficulty_segment_masses": segment_masses,
        "base_measure": "uniform_physical_alpha",
        "coverage_preserving_uniform_component": 1.0 - eta,
        "difficulty_profile_provenance": dict(DIFFICULTY_PROFILE_PROVENANCE),
    }


def difficulty_mixture_schedule(difficulty_mix: float = DEFAULT_DIFFICULTY_MIX):
    """Create the inverse-CDF map for the continuous difficulty-informed density.

    The returned callable has the standard bridge-schedule signature
    ``schedule(t, total_steps)``. It interprets ``u=t/T`` as a base uniform
    quantile and maps it through the inverse CDF of the piecewise-linear mixed
    density. With eta=0 this reduces exactly to alpha=t/T.
    """

    eta = _validated_mix(difficulty_mix)
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
        raise RuntimeError("difficulty-informed density must have positive mass")
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

        segment = torch.bucketize(u, cdf_tensor[1:-1], right=False)
        left_cdf = cdf_tensor[segment]
        left_alpha = alpha_tensor[segment]
        right_alpha = alpha_tensor[segment + 1]
        left_density = density_tensor[segment] / float(total_mass)
        right_density = density_tensor[segment + 1] / float(total_mass)

        width = right_alpha - left_alpha
        slope = (right_density - left_density) / width
        residual_mass = u - left_cdf

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
