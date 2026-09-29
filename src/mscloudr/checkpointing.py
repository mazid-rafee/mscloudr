"""Epoch-boundary checkpointing with reproducibility state.

Checkpoints preserve more than model weights. They also store optimizer state,
global RNG state, the explicit bridge sampler generator, and DataLoader
generator states so an epoch-boundary resume continues the intended stochastic
streams.

Mid-epoch resume is intentionally unsupported.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


CHECKPOINT_FORMAT_VERSION = 1


def _loader_generator_state(
    loader: DataLoader,
) -> torch.Tensor | None:
    generator = getattr(loader, "generator", None)
    if generator is None:
        return None
    return generator.get_state()


def capture_rng_state(
    *,
    sampler_generator: torch.Generator,
    loaders: Mapping[str, DataLoader],
) -> dict[str, Any]:
    """Capture stochastic state required for epoch-boundary continuation."""

    if sampler_generator is None:
        raise ValueError(
            "sampler_generator is required"
        )

    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": cuda_state,
        "bridge_sampler": sampler_generator.get_state(),
        "dataloader_generators": {
            name: _loader_generator_state(loader)
            for name, loader in loaders.items()
        },
    }


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    sampler_generator: torch.Generator,
    loaders: Mapping[str, DataLoader],
) -> None:
    """Restore stochastic state captured at an epoch boundary."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    cuda_state = state.get("torch_cuda_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

    sampler_generator.set_state(
        state["bridge_sampler"]
    )

    loader_states = state.get(
        "dataloader_generators",
        {},
    )
    for name, loader in loaders.items():
        saved = loader_states.get(name)
        if saved is None:
            continue

        generator = getattr(
            loader,
            "generator",
            None,
        )
        if generator is None:
            raise ValueError(
                f"checkpoint contains generator state for loader {name!r}, "
                "but the current loader has no generator"
            )
        generator.set_state(saved)


def save_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    best_endpoint_l1: float,
    sampler_generator: torch.Generator,
    loaders: Mapping[str, DataLoader],
    run_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save an epoch-boundary training checkpoint."""

    epoch = int(epoch)
    if epoch <= 0:
        raise ValueError(
            "epoch must be positive"
        )

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
        },
        "best_endpoint_l1": float(
            best_endpoint_l1
        ),
        "run_metadata": dict(
            run_metadata or {}
        ),
        "rng_state": capture_rng_state(
            sampler_generator=sampler_generator,
            loaders=loaders,
        ),
    }

    torch.save(
        payload,
        temporary,
    )
    os.replace(
        temporary,
        destination,
    )


def _torch_load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device | None,
) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    sampler_generator: torch.Generator,
    loaders: Mapping[str, DataLoader],
    map_location: str | torch.device | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load model/optimizer state and optionally restore all RNG streams."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {source}"
        )

    payload = _torch_load_checkpoint(
        source,
        map_location=map_location,
    )

    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported checkpoint format version"
        )

    model.load_state_dict(
        payload["model_state"],
        strict=True,
    )
    optimizer.load_state_dict(
        payload["optimizer_state"]
    )

    if restore_rng:
        restore_rng_state(
            payload["rng_state"],
            sampler_generator=sampler_generator,
            loaders=loaders,
        )

    return payload
