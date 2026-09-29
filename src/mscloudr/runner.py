"""Optimizer, epoch, validation, and checkpoint runner.

The batch mathematics live in mscloudr.training. This module only orchestrates
those already-tested computations.

For NFE=1 experiments, best_endpoint.pt is selected exclusively by
val_endpoint_l1. val_random_t_l1 is logged as a diagnostic and never controls
canonical checkpoint selection.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from .bridge import BridgeSchedule
from .checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from .training import (
    CHECKPOINT_SELECTION_METRIC,
    DIAGNOSTIC_RANDOM_T_METRIC,
    WeightedMean,
    endpoint_validation_step,
    random_t_validation_step,
    training_step,
)


HISTORICAL_LEARNING_RATE = 5e-5


@dataclass(frozen=True)
class BatchProgress:
    """One lightweight within-epoch progress event."""

    epoch: int
    phase: str
    batch: int
    total_batches: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_l1: float
    val_random_t_l1: float
    val_endpoint_l1: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "EpochMetrics":
        return cls(
            epoch=int(value["epoch"]),
            train_l1=float(value["train_l1"]),
            val_random_t_l1=float(value["val_random_t_l1"]),
            val_endpoint_l1=float(value["val_endpoint_l1"]),
        )


def make_adam_optimizer(
    model: nn.Module,
    *,
    lr: float = HISTORICAL_LEARNING_RATE,
) -> Adam:
    """Construct the historical Adam optimizer used by ms-cloudR."""

    lr = float(lr)
    if lr <= 0.0:
        raise ValueError(
            "lr must be positive"
        )
    return Adam(
        model.parameters(),
        lr=lr,
    )


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


def _should_report_progress(
    batch_number: int,
    total_batches: int,
    progress_every: int | None,
) -> bool:
    if progress_every is None:
        return False
    progress_every = int(
        progress_every
    )
    if progress_every <= 0:
        raise ValueError(
            "progress_every must be positive when provided"
        )
    return (
        batch_number == total_batches
        or batch_number % progress_every == 0
    )


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    *,
    schedule: BridgeSchedule,
    total_steps: int,
    sampler_generator: torch.Generator,
    device: torch.device,
    max_batches: int | None = None,
    epoch: int = 1,
    progress_every: int | None = None,
    progress_callback: Callable[[BatchProgress], None] | None = None,
) -> float:
    """Train for one epoch and return sample-weighted mean L1."""

    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when provided")

    model.train()
    meter = WeightedMean()
    total_batches = _effective_total_batches(
        loader,
        max_batches,
    )

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = _move_batch(
            batch,
            device=device,
        )

        result = training_step(
            model,
            batch,
            schedule=schedule,
            total_steps=total_steps,
            sampler_generator=sampler_generator,
        )

        optimizer.zero_grad(
            set_to_none=True
        )
        result.loss.backward()
        optimizer.step()

        meter.update(
            result.loss,
            n=result.batch_size,
        )

        batch_number = batch_index + 1
        if (
            progress_callback is not None
            and _should_report_progress(
                batch_number,
                total_batches,
                progress_every,
            )
        ):
            progress_callback(
                BatchProgress(
                    epoch=int(epoch),
                    phase="train",
                    batch=batch_number,
                    total_batches=total_batches,
                    metrics={
                        "train_l1_running": meter.mean,
                    },
                )
            )

    return meter.mean


def run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    schedule: BridgeSchedule,
    total_steps: int,
    sampler_generator: torch.Generator,
    device: torch.device,
    max_batches: int | None = None,
    epoch: int = 1,
    progress_every: int | None = None,
    progress_callback: Callable[[BatchProgress], None] | None = None,
) -> dict[str, float]:
    """Run diagnostic random-t and canonical endpoint validation."""

    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when provided")

    model.eval()
    random_meter = WeightedMean()
    endpoint_meter = WeightedMean()
    total_batches = _effective_total_batches(
        loader,
        max_batches,
    )

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = _move_batch(
            batch,
            device=device,
        )

        random_result = random_t_validation_step(
            model,
            batch,
            schedule=schedule,
            total_steps=total_steps,
            sampler_generator=sampler_generator,
        )
        endpoint_result = endpoint_validation_step(
            model,
            batch,
            total_steps=total_steps,
        )

        random_meter.update(
            random_result.loss,
            n=random_result.batch_size,
        )
        endpoint_meter.update(
            endpoint_result.loss,
            n=endpoint_result.batch_size,
        )

        batch_number = batch_index + 1
        if (
            progress_callback is not None
            and _should_report_progress(
                batch_number,
                total_batches,
                progress_every,
            )
        ):
            progress_callback(
                BatchProgress(
                    epoch=int(epoch),
                    phase="val",
                    batch=batch_number,
                    total_batches=total_batches,
                    metrics={
                        "val_random_t_l1_running": random_meter.mean,
                        "val_endpoint_l1_running": endpoint_meter.mean,
                    },
                )
            )

    return {
        DIAGNOSTIC_RANDOM_T_METRIC: random_meter.mean,
        CHECKPOINT_SELECTION_METRIC: endpoint_meter.mean,
    }


def _write_history(
    path: Path,
    history: list[EpochMetrics],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            [item.to_dict() for item in history],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(
        temporary,
        path,
    )


def _read_history(
    path: Path,
) -> list[EpochMetrics]:
    if not path.is_file():
        return []

    values = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    return [
        EpochMetrics.from_dict(value)
        for value in values
    ]


def _checkpoint_metadata(
    *,
    total_steps: int,
    epochs: int,
    optimizer: Optimizer,
    schedule_name: str,
    model_identity: str,
    run_metadata: Mapping[str, Any] | None,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict[str, Any]:
    metadata = dict(
        run_metadata or {}
    )
    metadata.update(
        {
            "model_identity": str(model_identity),
            "schedule_name": str(schedule_name),
            "total_steps": int(total_steps),
            "target_epochs": int(epochs),
            "optimizer": optimizer.__class__.__name__,
            "learning_rates": [
                float(group["lr"])
                for group in optimizer.param_groups
            ],
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "diagnostic_random_t_metric": DIAGNOSTIC_RANDOM_T_METRIC,
            "max_train_batches": (
                None if max_train_batches is None else int(max_train_batches)
            ),
            "max_val_batches": (
                None if max_val_batches is None else int(max_val_batches)
            ),
        }
    )
    return metadata


def fit(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    schedule: BridgeSchedule,
    schedule_name: str,
    total_steps: int,
    sampler_generator: torch.Generator,
    output_dir: str | os.PathLike[str],
    epochs: int,
    device: str | torch.device,
    model_identity: str = "legacy_dbcr",
    lr: float = HISTORICAL_LEARNING_RATE,
    optimizer: Optimizer | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    resume_from: str | os.PathLike[str] | None = None,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
    batch_progress_callback: Callable[[BatchProgress], None] | None = None,
    progress_every: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> list[EpochMetrics]:
    """Run epoch-boundary training with endpoint-selected checkpoints."""

    epochs = int(epochs)
    total_steps = int(total_steps)
    if epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )
    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive"
        )

    device = torch.device(
        device
    )
    model.to(device)

    if optimizer is None:
        optimizer = make_adam_optimizer(
            model,
            lr=lr,
        )

    loaders = {
        "train": train_loader,
        "val": val_loader,
    }

    output_dir = Path(
        output_dir
    )
    checkpoint_dir = (
        output_dir / "checkpoints"
    )
    history_path = (
        output_dir / "history.json"
    )

    start_epoch = 1
    best_endpoint_l1 = float("inf")
    history: list[EpochMetrics] = []

    if resume_from is not None:
        payload = load_training_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            sampler_generator=sampler_generator,
            loaders=loaders,
            map_location=device,
            restore_rng=True,
        )
        start_epoch = (
            int(payload["epoch"])
            + 1
        )
        best_endpoint_l1 = float(
            payload["best_endpoint_l1"]
        )
        history = _read_history(
            history_path
        )

        if history and history[-1].epoch != int(payload["epoch"]):
            raise ValueError(
                "history.json and resume checkpoint disagree on last epoch"
            )

    if start_epoch > epochs:
        raise ValueError(
            "resume checkpoint is already at or beyond target epochs"
        )

    metadata = _checkpoint_metadata(
        total_steps=total_steps,
        epochs=epochs,
        optimizer=optimizer,
        schedule_name=schedule_name,
        model_identity=model_identity,
        run_metadata=run_metadata,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        train_l1 = run_training_epoch(
            model,
            train_loader,
            optimizer,
            schedule=schedule,
            total_steps=total_steps,
            sampler_generator=sampler_generator,
            device=device,
            max_batches=max_train_batches,
            epoch=epoch,
            progress_every=progress_every,
            progress_callback=batch_progress_callback,
        )

        validation = run_validation_epoch(
            model,
            val_loader,
            schedule=schedule,
            total_steps=total_steps,
            sampler_generator=sampler_generator,
            device=device,
            max_batches=max_val_batches,
            epoch=epoch,
            progress_every=progress_every,
            progress_callback=batch_progress_callback,
        )

        metrics = EpochMetrics(
            epoch=epoch,
            train_l1=train_l1,
            val_random_t_l1=validation[
                DIAGNOSTIC_RANDOM_T_METRIC
            ],
            val_endpoint_l1=validation[
                CHECKPOINT_SELECTION_METRIC
            ],
        )
        history.append(
            metrics
        )

        improved = (
            metrics.val_endpoint_l1
            < best_endpoint_l1
        )
        if improved:
            best_endpoint_l1 = (
                metrics.val_endpoint_l1
            )

        checkpoint_metrics = (
            metrics.to_dict()
        )

        save_training_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=checkpoint_metrics,
            best_endpoint_l1=best_endpoint_l1,
            sampler_generator=sampler_generator,
            loaders=loaders,
            run_metadata=metadata,
        )

        if improved:
            save_training_checkpoint(
                checkpoint_dir / "best_endpoint.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=checkpoint_metrics,
                best_endpoint_l1=best_endpoint_l1,
                sampler_generator=sampler_generator,
                loaders=loaders,
                run_metadata=metadata,
            )

        _write_history(
            history_path,
            history,
        )

        if epoch_callback is not None:
            epoch_callback(metrics)

    return history
