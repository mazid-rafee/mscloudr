"""Generalization-gap-informed physical-alpha training measure for DB-CR pilots.

The calibration below comes from the fixed-alpha train/validation diagnostic on
CanonicalAlpha pilot data (seed 42, checkpoint epoch 20). At alpha knots
0.0,0.1,...,1.0, the diagnostic measured the sample-weighted held-out gap

    D_gen(alpha) = L_val(alpha) - L_train(alpha).

For training we form a continuous, coverage-preserving density

    q_eta(alpha) = (1-eta) * U(0,1)
                   + eta * D_tilde_gen(alpha),

where D_tilde_gen is the piecewise-linear interpolation of the positive measured
gap profile normalized to integrate to one. Sampling is implemented as a
quantile transform of the existing uniform discrete timestep variable. This
keeps the bridge in physical-alpha coordinates while emphasizing corruption
levels where held-out validation error exceeds training error most strongly.

The gap is a held-out validation signal, not a causal estimate of geographic
difficulty. The final test split was not used to construct this measure.
"""

from __future__ import annotations

import math

import torch


GENERALIZATION_PROFILE_ALPHAS = (
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

# sample_mean_l1(validation) - train_mean_l1(training), measured at fixed alpha.
GENERALIZATION_PROFILE_GAPS = (
    0.0022379570004837064,
    0.002803083874302585,
    0.003530749017421725,
    0.00427989397998383,
    0.00480748697662794,
    0.006027356686253104,
    0.006786838075076719,
    0.007331960804414856,
    0.00763425321427539,
    0.006696816407440159,
    0.006480441086141816,
)

GENERALIZATION_PROFILE_PROVENANCE = {
    "diagnostic": "generalization_aware_fixed_alpha_group_risk",
    "diagnostic_branch": "exp/generalization-aware-measure",
    "diagnostic_commit": "5f9c17216766d5e0d5f610cf943a20c31e60ab85",
    "source_run": "DBCR_CanonicalAlpha_pilot10_seed42",
    "source_checkpoint": "best_endpoint.pt",
    "source_checkpoint_epoch": 20,
    "source_train_profile": "difficulty_profile_train.json",
    "source_validation_profile": "generalization_profile_with_train_gap.json",
    "pilot_protocol": "uncrtaints_roi_disjoint_pilot10_roi_season_stratified_v1",
    "profile_statistic": "sample_val_minus_train_l1",
    "final_test_used_for_sampler_design": False,
}

DEFAULT_GENERALIZATION_MIX = 0.5


def _validated_mix(generalization_mix: float) -> float:
    eta = float(generalization_mix)
    if not math.isfinite(eta):
        raise ValueError("generalization-mix must be finite")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("generalization-mix must satisfy 0 <= eta <= 1")
    return eta


def _density_knots(generalization_mix: float) -> tuple[list[float], list[float]]:
    """Return alpha knots and normalized mixed-density values at those knots."""

    eta = _validated_mix(generalization_mix)
    alphas = list(GENERALIZATION_PROFILE_ALPHAS)
    scores = [max(0.0, float(v)) for v in GENERALIZATION_PROFILE_GAPS]

    area = 0.0
    for i in range(len(alphas) - 1):
        width = alphas[i + 1] - alphas[i]
        area += 0.5 * width * (scores[i] + scores[i + 1])
    if not area > 0.0:
        raise RuntimeError("generalization-gap calibration integral must be positive")

    normalized = [score / area for score in scores]
    density = [(1.0 - eta) + eta * value for value in normalized]
    return alphas, density


def generalization_measure_summary(
    generalization_mix: float = DEFAULT_GENERALIZATION_MIX,
) -> dict:
    """Return reproducibility metadata for the generalization-informed measure."""

    eta = _validated_mix(generalization_mix)
    alphas, density = _density_knots(eta)

    segment_masses: list[float] = []
    for i in range(len(alphas) - 1):
        width = alphas[i + 1] - alphas[i]
        segment_masses.append(0.5 * width * (density[i] + density[i + 1]))

    total = sum(segment_masses)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        segment_masses = [mass / total for mass in segment_masses]

    peak_index = max(
        range(len(GENERALIZATION_PROFILE_GAPS)),
        key=lambda i: GENERALIZATION_PROFILE_GAPS[i],
    )

    return {
        "training_bridge_measure": "generalization_gap_mixture_physical_alpha",
        "training_bridge_measure_definition": (
            "q_eta(alpha)=(1-eta)U(0,1)+eta*D_tilde_gen(alpha), where "
            "D_tilde_gen is the normalized piecewise-linear interpolation of "
            "sample-weighted held-out validation-minus-training L1"
        ),
        "alpha_sampling_identity": "generalization_gap_quantile_transform",
        "generalization_mix_eta": eta,
        "generalization_profile_alphas": list(GENERALIZATION_PROFILE_ALPHAS),
        "generalization_profile_gaps": list(GENERALIZATION_PROFILE_GAPS),
        "generalization_density_knots": density,
        "generalization_segment_masses": segment_masses,
        "generalization_gap_peak_alpha": GENERALIZATION_PROFILE_ALPHAS[peak_index],
        "generalization_gap_peak_value": GENERALIZATION_PROFILE_GAPS[peak_index],
        "base_measure": "uniform_physical_alpha",
        "coverage_preserving_uniform_component": 1.0 - eta,
        "generalization_profile_provenance": dict(GENERALIZATION_PROFILE_PROVENANCE),
    }


def generalization_mixture_schedule(
    generalization_mix: float = DEFAULT_GENERALIZATION_MIX,
):
    """Create inverse-CDF sampling for the continuous generalization-gap density.

    The returned callable follows the bridge-schedule signature
    ``schedule(t, total_steps)``. It treats ``u=t/T`` as a base uniform quantile
    and maps it through the inverse CDF of the piecewise-linear mixed density.
    With eta=0 this reduces exactly to alpha=t/T.
    """

    eta = _validated_mix(generalization_mix)
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
        raise RuntimeError("generalization-informed density must have positive mass")

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
