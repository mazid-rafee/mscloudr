"""Deterministic NFE=1 endpoint evaluation for paper-facing SEN12MS-CR runs.

The canonical DB-CR paper protocol evaluates the model at the bridge endpoint:

    x_T = cloudy,  t = T

For NFE=1 this inference input is schedule-independent. The schedule identity is
therefore read from checkpoint metadata for provenance, but no schedule callable
is used during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import ReferenceMetricAccumulator
from .training import endpoint_validation_step


@dataclass(frozen=True)
class EvaluationProgress:
    batch: int
    total_batches: int
    num_samples: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvaluationResult:
    num_samples: int
    num_batches: int
    metrics: dict[str, float]
    metric_counts: dict[str, int]


def _move_batch(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(
                device=device,
                non_blocking=True,
            )
        else:
            moved[key] = value
    return moved


def _effective_total_batches(
    loader: DataLoader,
    max_batches: int | None,
) -> int:
    total = len(loader)
    if max_batches is not None:
        total = min(
            total,
            int(max_batches),
        )
    return int(total)


def _should_report(
    batch_number: int,
    total_batches: int,
    progress_every: int | None,
) -> bool:
    if progress_every is None:
        return False
    progress_every = int(progress_every)
    if progress_every <= 0:
        raise ValueError(
            "progress_every must be positive when provided"
        )
    return (
        batch_number == total_batches
        or batch_number % progress_every == 0
    )


def _paper_metric_aliases(
    metrics: Mapping[str, float],
    counts: Mapping[str, int],
) -> tuple[dict[str, float], dict[str, int]]:
    """Expose MAE also as L1 for continuity with DB-CR experiment tables."""

    paper_metrics = dict(metrics)
    paper_counts = dict(counts)
    if "MAE" in paper_metrics:
        paper_metrics["L1"] = paper_metrics["MAE"]
        paper_counts["L1"] = paper_counts.get("MAE", 0)
    return paper_metrics, paper_counts


def evaluate_nfe1_endpoint(
    model: nn.Module,
    loader: DataLoader,
    *,
    total_steps: int,
    device: str | torch.device,
    max_batches: int | None = None,
    progress_every: int | None = None,
    progress_callback: Callable[[EvaluationProgress], None] | None = None,
) -> EvaluationResult:
    """Evaluate one model on a deterministic endpoint DataLoader.

    Metrics are accumulated as arithmetic means over individual images using
    the repository's UnCRtainTS-compatible reference metric implementation.
    Predictions are not clipped before metric computation.
    """

    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive"
        )
    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError(
            "max_batches must be positive when provided"
        )

    device = torch.device(device)
    model.to(device)
    model.eval()

    accumulator = ReferenceMetricAccumulator()
    num_samples = 0
    num_batches = 0
    total_batches = _effective_total_batches(
        loader,
        max_batches,
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if (
                max_batches is not None
                and batch_index >= int(max_batches)
            ):
                break

            batch = _move_batch(
                batch,
                device=device,
            )
            if "target" not in batch:
                raise KeyError(
                    "evaluation batch is missing required key 'target'"
                )

            result = endpoint_validation_step(
                model,
                batch,
                total_steps=total_steps,
            )
            target = batch["target"]
            accumulator.update_batch(
                target,
                result.prediction,
            )

            num_batches += 1
            num_samples += int(
                target.shape[0]
            )

            batch_number = batch_index + 1
            if (
                progress_callback is not None
                and _should_report(
                    batch_number,
                    total_batches,
                    progress_every,
                )
            ):
                running_metrics, _ = _paper_metric_aliases(
                    accumulator.compute(),
                    accumulator.metric_counts(),
                )
                progress_callback(
                    EvaluationProgress(
                        batch=batch_number,
                        total_batches=total_batches,
                        num_samples=num_samples,
                        metrics=running_metrics,
                    )
                )

    metrics, metric_counts = _paper_metric_aliases(
        accumulator.compute(),
        accumulator.metric_counts(),
    )

    if num_samples == 0:
        raise ValueError(
            "evaluation loader produced no samples"
        )

    return EvaluationResult(
        num_samples=num_samples,
        num_batches=num_batches,
        metrics=metrics,
        metric_counts=metric_counts,
    )
