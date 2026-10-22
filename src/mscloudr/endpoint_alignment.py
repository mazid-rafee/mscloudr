"""Read-only endpoint-gradient alignment profiler for deterministic DB-CR.

For each TRAIN batch, compare the parameter gradient of the endpoint objective
(t=T, input=cloudy) with gradients at eight disjoint interior timestep bins.
The profiler never calls backward() or an optimizer, and never uses test data.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn

from .bridge import BridgeSchedule, make_bridge_state
from .training import RAW_T_CONDITIONING, model_conditioning_coordinate


def gradient_alignment(
    endpoint_grads: tuple[torch.Tensor | None, ...],
    candidate_grads: tuple[torch.Tensor | None, ...],
) -> dict[str, float]:
    """Compute full-parameter dot/cosine without concatenating huge vectors."""
    if len(endpoint_grads) != len(candidate_grads):
        raise ValueError("gradient lists have different lengths")
    dot = norm_e2 = norm_c2 = None
    for e, c in zip(endpoint_grads, candidate_grads):
        if e is None and c is None:
            continue
        anchor = e if e is not None else c
        assert anchor is not None
        if dot is None:
            dot = torch.zeros((), device=anchor.device, dtype=torch.float64)
            norm_e2 = torch.zeros_like(dot)
            norm_c2 = torch.zeros_like(dot)
        if e is not None:
            norm_e2 = norm_e2 + e.detach().double().square().sum()
        if c is not None:
            norm_c2 = norm_c2 + c.detach().double().square().sum()
        if e is not None and c is not None:
            dot = dot + (e.detach().double() * c.detach().double()).sum()
    if dot is None or norm_e2 is None or norm_c2 is None:
        raise ValueError("no parameter gradients were produced")
    if norm_e2.item() <= 0 or norm_c2.item() <= 0:
        raise ValueError("cannot measure cosine of zero gradient")
    cosine = dot / (norm_e2 * norm_c2).sqrt()
    return {
        "dot_product": float(dot.item()),
        "cosine": float(cosine.clamp(-1.0, 1.0).item()),
        "endpoint_grad_norm": float(norm_e2.sqrt().item()),
        "candidate_grad_norm": float(norm_c2.sqrt().item()),
        "endpoint_projection": float((dot / norm_e2).item()),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("empty diagnostic samples")
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def profile_endpoint_alignment(
    model: nn.Module,
    train_loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    schedule: BridgeSchedule,
    total_steps: int,
    device: str | torch.device,
    max_batches: int = 4,
    num_bins: int = 8,
    sampler_seed: int = 42,
) -> dict[str, Any]:
    """Return per-bin TRAIN-gradient diagnostics, with model weights unchanged.

    Each bin is [floor(i*T/K), floor((i+1)*T/K)); the endpoint T is
    evaluated separately. All bins use exactly the same TRAIN image batches
    as the endpoint gradient so results estimate local gradient agreement.
    """
    total_steps, max_batches, num_bins = int(total_steps), int(max_batches), int(num_bins)
    if not 1 <= num_bins <= total_steps:
        raise ValueError("num_bins must be in [1, total_steps]")
    if max_batches <= 0 or sampler_seed < 0:
        raise ValueError("max_batches must be positive and seed nonnegative")
    dev = torch.device(device)
    first_param = next(model.parameters(), None)
    if first_param is not None and first_param.device != dev:
        raise ValueError("model must already be on requested device")
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if not params:
        raise ValueError("model has no trainable parameters")
    generator = torch.Generator(device="cpu").manual_seed(sampler_seed)
    bounds = [(i * total_steps // num_bins, (i + 1) * total_steps // num_bins)
              for i in range(num_bins)]
    observations: list[list[dict[str, float]]] = [[] for _ in bounds]
    processed_batches = 0
    was_training = model.training
    try:
        model.eval()  # Avoid modifying running statistics; no optimizer steps.
        with torch.enable_grad():
            for batch in train_loader:
                if processed_batches >= max_batches:
                    break
                cloudy = batch["cloudy"].to(dev)
                sar = batch["sar"].to(dev)
                target = batch["target"].to(dev)
                if cloudy.shape != target.shape or cloudy.shape[0] != sar.shape[0]:
                    raise ValueError("invalid batch dimensions")
                batch_size = target.shape[0]
                endpoint_t = torch.full((batch_size,), total_steps, device=dev,
                                        dtype=torch.long)
                pred_endpoint = model(cloudy, endpoint_t, sar)
                endpoint_loss = (pred_endpoint - target).abs().mean()
                endpoint_grads = torch.autograd.grad(endpoint_loss, params, allow_unused=True)
                endpoint_l1 = float(endpoint_loss.detach().item())
                del pred_endpoint, endpoint_loss

                for bin_id, (lo, hi) in enumerate(bounds):
                    ts = torch.randint(lo, hi, (batch_size,), generator=generator).to(dev)
                    with torch.no_grad():
                        alpha = schedule(ts.float(), total_steps)
                        x_t = make_bridge_state(target, cloudy, alpha)
                        conditioning = model_conditioning_coordinate(
                            ts, alpha, total_steps=total_steps,
                            conditioning_mode=RAW_T_CONDITIONING,
                        )
                    predicted = model(x_t, conditioning, sar)
                    loss = (predicted - target).abs().mean()
                    candidate_grads = torch.autograd.grad(loss, params, allow_unused=True)
                    stats = gradient_alignment(endpoint_grads, candidate_grads)
                    stats.update({
                        "endpoint_l1": endpoint_l1,
                        "candidate_l1": float(loss.detach().item()),
                        "mean_alpha": float(alpha.float().mean().item()),
                        "mean_t": float(ts.float().mean().item()),
                    })
                    observations[bin_id].append(stats)
                    del candidate_grads, predicted, loss
                del endpoint_grads
                processed_batches += 1
    finally:
        model.train(was_training)
    if processed_batches == 0:
        raise ValueError("training loader yielded no batches")
    summaries = []
    fields = ("cosine", "dot_product", "endpoint_projection", "endpoint_l1",
              "candidate_l1", "endpoint_grad_norm", "candidate_grad_norm",
              "mean_alpha", "mean_t")
    for (lo, hi), group in zip(bounds, observations):
        summaries.append({
            "t_start_inclusive": lo,
            "t_end_exclusive": hi,
            "num_batches": len(group),
            "positive_cosine_fraction": float(sum(o["cosine"] > 0 for o in group) / len(group)),
            **{field: _mean_std([o[field] for o in group]) for field in fields},
        })
    return {
        "diagnostic": "endpoint_gradient_alignment_v1",
        "split_used": "train_only",
        "no_parameter_updates": True,
        "prediction_target": "direct_x0",
        "conditioning_mode": "raw_t",
        "endpoint_t": total_steps,
        "interior_t_includes_endpoint": False,
        "total_steps": total_steps,
        "num_bins": num_bins,
        "num_batches": processed_batches,
        "sampler_seed": sampler_seed,
        "parameters_profiled": sum(p.numel() for p in params),
        "bins": summaries,
    }
