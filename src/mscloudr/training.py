"""Pure training and validation computations for bridge experiments.

This module deliberately stops short of owning an optimizer, epoch loop, or
checkpoint writer. It defines only the batch computations every training loop
must use.

Random-t training and random-t validation require an explicit sampler
torch.Generator. Endpoint validation is deterministic and schedule independent:
x_T is exactly the cloudy optical observation and the endpoint conditioning
coordinate is T for both raw-t and physical-alpha conditioning.

The canonical checkpoint-selection metric for NFE=1 experiments is
val_endpoint_l1. Random-t validation is diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from .bridge import (
    BridgeSchedule,
    make_bridge_state,
    sample_timesteps,
)


CHECKPOINT_SELECTION_METRIC = "val_endpoint_l1"
DIAGNOSTIC_RANDOM_T_METRIC = "val_random_t_l1"

RAW_T_CONDITIONING = "raw_t"
PHYSICAL_ALPHA_CONDITIONING = "physical_alpha"
SUPPORTED_CONDITIONING_MODES = (
    RAW_T_CONDITIONING,
    PHYSICAL_ALPHA_CONDITIONING,
)


@dataclass
class BridgeStepResult:
    """Outputs needed for logging, testing, and later epoch aggregation."""

    loss: torch.Tensor
    prediction: torch.Tensor
    timesteps: torch.Tensor
    alpha: torch.Tensor
    bridge_state: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.prediction.shape[0])

    def detached_scalars(self) -> dict[str, float]:
        return {
            "l1": float(self.loss.detach().item()),
        }


def _require_batch(
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = {
        "cloudy",
        "sar",
        "target",
    }
    missing = sorted(
        required - set(batch)
    )
    if missing:
        raise KeyError(
            f"batch missing required keys: {missing}"
        )

    cloudy = batch["cloudy"]
    sar = batch["sar"]
    target = batch["target"]

    if not (
        torch.is_tensor(cloudy)
        and torch.is_tensor(sar)
        and torch.is_tensor(target)
    ):
        raise TypeError(
            "cloudy, sar, and target must be torch tensors"
        )

    if cloudy.ndim != 4 or target.ndim != 4 or sar.ndim != 4:
        raise ValueError(
            "cloudy, sar, and target must have shape [B,C,H,W]"
        )
    if cloudy.shape != target.shape:
        raise ValueError(
            "cloudy and target must have identical shapes"
        )
    if cloudy.shape[0] != sar.shape[0]:
        raise ValueError(
            "cloudy, SAR, and target batch sizes must match"
        )
    if cloudy.shape[-2:] != sar.shape[-2:]:
        raise ValueError(
            "cloudy, SAR, and target spatial sizes must match"
        )

    return cloudy, sar, target


def _batch_alpha(
    alpha: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    return alpha.to(
        device=reference.device,
        dtype=reference.dtype,
    ).view(-1, 1, 1, 1)


def model_conditioning_coordinate(
    timesteps: torch.Tensor,
    alpha: torch.Tensor,
    *,
    total_steps: int,
    conditioning_mode: str = RAW_T_CONDITIONING,
) -> torch.Tensor:
    """Map a bridge state to the scalar coordinate seen by the network.

    ``raw_t`` reproduces historical DB-CR conditioning exactly: the model sees
    the sampled schedule parameter t.

    ``physical_alpha`` removes schedule-dependent coordinate semantics: the
    model sees c = T * alpha, where alpha is the actual physical interpolation
    coefficient used to construct x_alpha. Scaling by T preserves the legacy
    numerical conditioning range [0, T] while making equal physical bridge
    states receive equal conditioning values under every schedule.
    """

    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    mode = str(conditioning_mode).strip().lower()
    if mode not in SUPPORTED_CONDITIONING_MODES:
        raise ValueError(
            f"unsupported conditioning mode: {conditioning_mode!r}"
        )

    if mode == RAW_T_CONDITIONING:
        return timesteps

    return alpha.to(
        device=timesteps.device,
        dtype=torch.float32,
    ) * float(total_steps)


def random_t_bridge_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    schedule: BridgeSchedule,
    total_steps: int,
    sampler_generator: torch.Generator,
    conditioning_mode: str = RAW_T_CONDITIONING,
    curvature_kappa: float = 0.0,
) -> BridgeStepResult:
    """Compute one random-t direct-x0-prediction L1 batch.

    This function is used for both training and diagnostic random-t validation.
    The caller decides whether gradients are enabled. ``curvature_kappa`` only
    changes the physical bridge geometry; timestep sampling and conditioning
    remain controlled separately.
    """

    cloudy, sar, target = _require_batch(
        batch
    )

    timesteps = sample_timesteps(
        target.shape[0],
        total_steps,
        generator=sampler_generator,
        device=target.device,
    )
    alpha = schedule(
        timesteps.float(),
        total_steps,
    )
    bridge_state = make_bridge_state(
        target,
        cloudy,
        alpha,
        curvature_kappa=curvature_kappa,
    )
    conditioning = model_conditioning_coordinate(
        timesteps,
        alpha,
        total_steps=total_steps,
        conditioning_mode=conditioning_mode,
    )

    prediction = model(
        bridge_state,
        conditioning,
        sar,
    )
    if prediction.shape != target.shape:
        raise ValueError(
            "model prediction shape must match target shape"
        )

    loss = torch.mean(
        torch.abs(
            prediction - target
        )
    )

    return BridgeStepResult(
        loss=loss,
        prediction=prediction,
        timesteps=timesteps,
        alpha=_batch_alpha(
            alpha,
            target,
        ),
        bridge_state=bridge_state,
    )


def training_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    schedule: BridgeSchedule,
    total_steps: int,
    sampler_generator: torch.Generator,
    conditioning_mode: str = RAW_T_CONDITIONING,
    curvature_kappa: float = 0.0,
) -> BridgeStepResult:
    """Training computation before optimizer zero_grad/backward/step."""

    return random_t_bridge_step(
        model,
        batch,
        schedule=schedule,
        total_steps=total_steps,
        sampler_generator=sampler_generator,
        conditioning_mode=conditioning_mode,
        curvature_kappa=curvature_kappa,
    )


def random_t_validation_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    schedule: BridgeSchedule,
    total_steps: int,
    sampler_generator: torch.Generator,
    conditioning_mode: str = RAW_T_CONDITIONING,
    curvature_kappa: float = 0.0,
) -> BridgeStepResult:
    """Diagnostic validation under the schedule-dependent random-t measure."""

    with torch.no_grad():
        return random_t_bridge_step(
            model,
            batch,
            schedule=schedule,
            total_steps=total_steps,
            sampler_generator=sampler_generator,
            conditioning_mode=conditioning_mode,
            curvature_kappa=curvature_kappa,
        )


def endpoint_validation_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    total_steps: int,
    conditioning_mode: str = RAW_T_CONDITIONING,
) -> BridgeStepResult:
    """Schedule- and curvature-invariant endpoint validation for NFE=1.

    Since every admissible bridge geometry used here satisfies x(alpha=1)=y,
    the endpoint model input is constructed directly as x_T = cloudy. Both
    supported conditioning modes equal T at alpha=1, so endpoint inference
    remains identical in coordinate value.
    """

    total_steps = int(
        total_steps
    )
    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive"
        )

    cloudy, sar, target = _require_batch(
        batch
    )
    batch_size = target.shape[0]

    timesteps = torch.full(
        (batch_size,),
        total_steps,
        device=target.device,
        dtype=torch.long,
    )
    alpha_vector = torch.ones(
        (batch_size,),
        device=target.device,
        dtype=target.dtype,
    )
    alpha = alpha_vector.view(
        batch_size,
        1,
        1,
        1,
    )
    conditioning = model_conditioning_coordinate(
        timesteps,
        alpha_vector,
        total_steps=total_steps,
        conditioning_mode=conditioning_mode,
    )

    with torch.no_grad():
        prediction = model(
            cloudy,
            conditioning,
            sar,
        )
        if prediction.shape != target.shape:
            raise ValueError(
                "model prediction shape must match target shape"
            )
        loss = torch.mean(
            torch.abs(
                prediction - target
            )
        )

    return BridgeStepResult(
        loss=loss,
        prediction=prediction,
        timesteps=timesteps,
        alpha=alpha,
        bridge_state=cloudy,
    )


@dataclass
class WeightedMean:
    """Sample-weighted scalar accumulator for unequal final batch sizes."""

    weighted_sum: float = 0.0
    weight: int = 0

    def update(
        self,
        value: float | torch.Tensor,
        *,
        n: int,
    ) -> None:
        n = int(n)
        if n <= 0:
            raise ValueError(
                "n must be positive"
            )
        scalar = (
            float(value.detach().item())
            if torch.is_tensor(value)
            else float(value)
        )
        self.weighted_sum += scalar * n
        self.weight += n

    @property
    def mean(self) -> float:
        if self.weight == 0:
            raise ValueError(
                "cannot compute mean before any updates"
            )
        return (
            self.weighted_sum
            / float(self.weight)
        )


def checkpoint_metric_payload(
    *,
    endpoint_l1: float,
    random_t_l1: float | None = None,
) -> dict[str, float]:
    """Create the validation metric payload used by the future epoch loop."""

    payload = {
        CHECKPOINT_SELECTION_METRIC: float(
            endpoint_l1
        ),
    }
    if random_t_l1 is not None:
        payload[
            DIAGNOSTIC_RANDOM_T_METRIC
        ] = float(
            random_t_l1
        )
    return payload
