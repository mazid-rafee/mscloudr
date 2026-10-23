"""Component-supervised training step for CanonicalBridgeNet.

Keeps original bridge sampling and MR-inverse conditioning. Training adds
independent losses on two *orthogonal output subspaces*. NFE=1 evaluation
uses the same reconstructed x0 as any direct-prediction CanonicalBridge model.
"""
from __future__ import annotations

from typing import Mapping

import torch

from .bridge import BridgeSchedule, make_bridge_state, sample_timesteps
from .models.component_bridge import (
    CanonicalComponentBridgeNet, low_pass_projection, high_pass_projection
)
from .training import BridgeStepResult, MR_INVERSE_CONDITIONING, model_conditioning_coordinate


def component_training_step(
    model: CanonicalComponentBridgeNet,
    batch: Mapping[str, torch.Tensor],
    *,
    schedule: BridgeSchedule,
    total_steps: int,
    sampler_generator: torch.Generator,
    conditioning_mode: str = MR_INVERSE_CONDITIONING,
    endpoint_probability: float = 0.0,
    component_weight: float = 0.5,
) -> BridgeStepResult:
    if not isinstance(model, CanonicalComponentBridgeNet):
        raise TypeError("component training requires CanonicalComponentBridgeNet")
    if conditioning_mode != MR_INVERSE_CONDITIONING:
        raise ValueError("component experiment requires MR-inverse conditioning")
    if endpoint_probability != 0.0:
        raise ValueError("component experiment forbids endpoint mixture")
    if not 0.0 <= float(component_weight) <= 10.0:
        raise ValueError("component_weight must lie in [0,10]")
    cloudy, sar, target = batch["cloudy"], batch["sar"], batch["target"]
    if cloudy.shape != target.shape or cloudy.shape[-2:] != sar.shape[-2:]:
        raise ValueError("invalid optical/SAR/target shape")
    timesteps = sample_timesteps(
        target.shape[0], total_steps, generator=sampler_generator,
        device=target.device,
    )
    alpha = schedule(timesteps.float(), total_steps)
    x_alpha = make_bridge_state(target, cloudy, alpha)
    conditioning = model_conditioning_coordinate(
        timesteps, alpha, total_steps=total_steps, conditioning_mode=conditioning_mode
    )
    predicted_low, predicted_high = model.forward_components(x_alpha, conditioning, sar)
    reconstructed = predicted_low + predicted_high
    reconstruction_loss = (reconstructed - target).abs().mean()
    low_loss = (predicted_low - low_pass_projection(target)).abs().mean()
    high_loss = (predicted_high - high_pass_projection(target)).abs().mean()
    objective = reconstruction_loss + float(component_weight) * (low_loss + high_loss) / 2.0
    return BridgeStepResult(
        loss=objective, prediction=reconstructed,
        timesteps=timesteps, alpha=alpha.to(dtype=target.dtype).view(-1, 1, 1, 1),
        bridge_state=x_alpha,
    )
