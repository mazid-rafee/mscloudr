"""Read-only SAR intervention helpers for source-guided DB-CR.

The experiment runs identical optical inputs at the same endpoint time while
replacing only the SAR tensor. SFBlock hooks record outputs for each run.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

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
