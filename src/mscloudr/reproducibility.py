"""Utilities for deterministic, auditable experiments.

The project keeps dataset splitting, model/training randomness, and stochastic
bridge sampling conceptually separate. A single integer can still be reused
for all three, but it should be passed explicitly to each subsystem.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SeedConfig:
    """Seeds recorded for one experiment."""

    split_seed: int
    train_seed: int
    sampler_seed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy, PyTorch CPU, and all CUDA devices.

    PYTHONHASHSEED only fully controls hash randomization when set before the
    interpreter starts. We still set it here so child processes inherit it.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic_algorithms)


def make_torch_generator(seed: int, *, device: str = "cpu") -> torch.Generator:
    """Return an explicitly seeded generator for a stochastic subsystem."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs inside a PyTorch DataLoader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reproducibility_metadata(
    seeds: SeedConfig,
    *,
    deterministic_algorithms: bool,
) -> dict[str, Any]:
    """Return metadata that should be saved with every experiment."""

    return {
        "seeds": seeds.to_dict(),
        "deterministic_algorithms": deterministic_algorithms,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
