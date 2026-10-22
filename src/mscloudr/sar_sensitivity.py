"""Read-only SAR intervention helpers for source-guided DB-CR.

The experiment runs identical optical inputs at the same endpoint time while
replacing only the SAR tensor. SFBlock hooks record outputs for each run.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import random
from statistics import median

import torch

from mscloudr.models.blocks import SFBlock


def rotate_other_sample_indices(count: int) -> list[int]:
    """Global one-position cyclic permutation: no sample keeps its own SAR."""
    if count < 2:
        raise ValueError("SAR shuffling requires at least two samples")
    result = [(i + 1) % count for i in range(count)]
    if sorted(result) != list(range(count)):
        raise AssertionError("invalid SAR permutation")
    return result


def per_sample_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError("require identically shaped NCHW tensors")
    return (a.float() - b.float()).abs().flatten(start_dim=1).mean(dim=1)


def per_sample_rms(a: torch.Tensor) -> torch.Tensor:
    if a.ndim != 4:
        raise ValueError("require NCHW tensor")
    return torch.linalg.vector_norm(
        a.float().flatten(start_dim=1), dim=1
    ) / math.sqrt(a[0].numel())


def tensor_change(
    baseline: torch.Tensor,
    intervention: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Per-sample intervention-vs-baseline change, never compares mismatched images."""
    if baseline.shape != intervention.shape or baseline.ndim != 4:
        raise ValueError("baseline and intervention must be matching NCHW tensors")
    diff = intervention.float() - baseline.float()
    baseline_norm = torch.linalg.vector_norm(
        baseline.float().flatten(start_dim=1), dim=1
    )
    delta_norm = torch.linalg.vector_norm(diff.flatten(start_dim=1), dim=1)
    return {
        "l1": per_sample_l1(baseline, intervention),
        "rms": per_sample_rms(diff),
        "relative_l2": delta_norm / (baseline_norm + eps),
        "baseline_rms": per_sample_rms(baseline),
    }


class SFBlockSnapshot:
    """Capture SFBlock outputs for a *single* model forward call.

    Short-lived read-only hooks; snapshots are detached and cloned so
    later inference runs cannot mutate them.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.blocks: dict[str, SFBlock] = {
            name: block
            for name, block in model.named_modules()
            if isinstance(block, SFBlock)
        }
        if not self.blocks:
            raise ValueError("no SFBlocks in model")
        self.outputs: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def _capture(self, name: str, output: torch.Tensor) -> None:
        if name in self.outputs:
            raise RuntimeError(f"SFBlock invoked multiple times: {name}")
        self.outputs[name] = output.detach().clone()

    @contextmanager
    def hooked(self):
        if self._handles:
            raise RuntimeError("SFBlock snapshots already hooked")
        self.outputs = {}
        try:
            for name, block in self.blocks.items():
                self._handles.append(
                    block.register_forward_hook(
                        lambda m, inputs, output, name=name:
                            self._capture(name, output)
                    )
                )
            yield self
            if set(self.outputs) != set(self.blocks):
                raise RuntimeError(
                    "not all SFBlocks ran: "
                    f"{sorted(set(self.blocks) - set(self.outputs))}"
                )
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()

    def run(
        self,
        model: torch.nn.Module,
        cloudy: torch.Tensor,
        t: torch.Tensor,
        sar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with self.hooked():
            prediction = model(cloudy, t, sar)
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("nonfinite optical prediction")
        for name, tensor in self.outputs.items():
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"nonfinite SFBlock output: {name}")
        return prediction, self.outputs


def extend_sample_measurements(
    target: list[float],
    measurements: torch.Tensor,
) -> None:
    target.extend(float(v) for v in measurements.detach().cpu().tolist())


def summarize_samples(data: dict[str, list[float]]) -> dict[str, float]:
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("no measurements or inconsistent sample counts")
    return {
        name + "_mean": sum(values) / len(values)
        for name, values in data.items()
    }



def paired_reconstruction_statistics(
    normal_l1: list[float],
    shuffled_l1: list[float],
    roi_groups: list[str],
    *,
    bootstrap_seed: int = 42,
    bootstrap_repetitions: int = 2000,
) -> dict:
    """Paired SHUFFLED-minus-NORMAL L1 with deterministic uncertainty estimates.

    Positive delta indicates correctly paired SAR has lower reconstruction L1.
    The patch bootstrap assumes independent patches; the ROI-cluster bootstrap
    resamples all patches of a region together to reflect spatial dependence.
    Both are descriptive for this fixed subset and *single* SAR permutation.
    """
    n = len(normal_l1)
    if n < 2 or len(shuffled_l1) != n or len(roi_groups) != n:
        raise ValueError("paired analysis requires >=2 aligned samples and ROI IDs")
    if bootstrap_repetitions < 100:
        raise ValueError("bootstrap_repetitions must be >= 100")
    if any(not math.isfinite(float(v)) for v in (*normal_l1, *shuffled_l1)):
        raise ValueError("L1 values must be finite")
    if any(not isinstance(group, str) or not group for group in roi_groups):
        raise ValueError("ROI groups must be nonempty strings")

    deltas = [
        float(shuffled) - float(normal)
        for normal, shuffled in zip(normal_l1, shuffled_l1)
    ]
    groups: dict[str, list[float]] = {}
    for group, delta in zip(roi_groups, deltas):
        groups.setdefault(group, []).append(delta)
    group_names = sorted(groups)
    rng = random.Random(bootstrap_seed)

    def percentile(sorted_values: list[float], probability: float) -> float:
        position = probability * (len(sorted_values) - 1)
        lo = int(math.floor(position))
        hi = int(math.ceil(position))
        fraction = position - lo
        return (1.0 - fraction) * sorted_values[lo] + fraction * sorted_values[hi]

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [percentile(ordered, 0.025), percentile(ordered, 0.975)]

    # Mean of paired deltas; resample matched (NORMAL, SHUFFLED) units together.
    sample_means = []
    for _ in range(bootstrap_repetitions):
        sample_means.append(
            sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        )

    # Preserve ROI-level dependence by resampling entire ROI groups as clusters.
    # Weight by selected patch count in each resampled group.
    cluster_means = []
    group_sums = [sum(groups[group]) for group in group_names]
    group_counts = [len(groups[group]) for group in group_names]
    g = len(group_names)
    for _ in range(bootstrap_repetitions):
        sampled = [rng.randrange(g) for _ in range(g)]
        numerator = sum(group_sums[index] for index in sampled)
        denominator = sum(group_counts[index] for index in sampled)
        cluster_means.append(numerator / denominator)

    positive = sum(delta > 0 for delta in deltas)
    negative = sum(delta < 0 for delta in deltas)
    ties = n - positive - negative
    mean_normal = sum(float(v) for v in normal_l1) / n
    mean_shuffled = sum(float(v) for v in shuffled_l1) / n

    return {
        "paired_delta_definition": "shuffled_l1 - normal_l1; positive favors NORMAL",
        "num_samples": n,
        "num_roi_groups": g,
        "mean_normal_l1": mean_normal,
        "mean_shuffled_l1": mean_shuffled,
        "mean_delta_l1": sum(deltas) / n,
        "median_delta_l1": float(median(deltas)),
        "relative_increase_pct": (
            100.0 * (mean_shuffled - mean_normal) / mean_normal
            if mean_normal > 0 else None
        ),
        "normal_better_count": positive,
        "shuffled_better_count": negative,
        "tie_count": ties,
        "fraction_normal_better": positive / n,
        "patch_bootstrap_95ci": interval(sample_means),
        "roi_cluster_bootstrap_95ci": interval(cluster_means),
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": bootstrap_seed,
        "roi_group_sample_counts": {
            group: len(groups[group]) for group in group_names
        },
        "roi_group_mean_delta_l1": {
            group: sum(groups[group]) / len(groups[group])
            for group in group_names
        },
        "roi_groups_favoring_normal": sum(
            sum(groups[group]) > 0 for group in group_names
        ),
        "caution": (
            "Fixed validation subset, one deterministic SAR permutation, "
            "and only a few spatially dependent ROI groups: these intervals "
            "are descriptive and do not establish population-level significance."
        ),
    }
