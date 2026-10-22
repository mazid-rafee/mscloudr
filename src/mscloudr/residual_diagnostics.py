"""Read-only forward-hook measurements of source-guided DB-CR residual updates.

For each PatentTimeEmbeddedNAFBlock:
  x = block input
  z = pre-norm2 input = x + beta * MBConv(x, alpha)
  y = block output = z + gamma * FFN(z, alpha)

Per-sample relative L2 norms:
  MBConv = ||z-x||_2 / (||x||_2 + eps)
  FFN    = ||y-z||_2 / (||z||_2 + eps)
  Total  = ||y-x||_2 / (||x||_2 + eps)

Hooks inspect tensors only. No patching, in-place mutation, parameter
updates, optimizer construction, or gradient computation is needed.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import torch

from mscloudr.models.patent_blocks import PatentTimeEmbeddedNAFBlock


def _norm_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("residual diagnostics expect NCHW features")
    return torch.linalg.vector_norm(tensor.float().flatten(start_dim=1), dim=1)


def relative_residual_norms(
    x: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return B-vectors: MBConv, FFN, and combined relative update magnitudes."""
    if x.shape != z.shape or x.shape != y.shape or x.ndim != 4:
        raise ValueError("x, z, y must be identically shaped NCHW tensors")
    if eps <= 0:
        raise ValueError("eps must be positive")
    x_norm = _norm_per_sample(x)
    z_norm = _norm_per_sample(z)
    mbconv = _norm_per_sample(z.float() - x.float()) / (x_norm + eps)
    ffn = _norm_per_sample(y.float() - z.float()) / (z_norm + eps)
    combined = _norm_per_sample(y.float() - x.float()) / (x_norm + eps)
    return mbconv, ffn, combined


class ResidualContributionRecorder:
    """Register temporary forward hooks; record per-sample ratios by NAF block."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.blocks: dict[str, PatentTimeEmbeddedNAFBlock] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PatentTimeEmbeddedNAFBlock)
        }
        if not self.blocks:
            raise ValueError("no PatentTimeEmbeddedNAFBlocks found")
        self._handles: list = []
        self._active: dict[str, dict[str, torch.Tensor]] = {}
        self._observations = defaultdict(
            lambda: {"mbconv": [], "ffn": [], "total": []}
        )

    def _on_enter(self, name: str, inputs: tuple) -> None:
        if name in self._active:
            raise RuntimeError(f"block was re-entered before completing: {name}")
        self._active[name] = {"x": inputs[0]}

    def _before_norm2(self, name: str, inputs: tuple) -> None:
        if name not in self._active:
            raise RuntimeError(f"block norm2 ran without a block input: {name}")
        self._active[name]["z"] = inputs[0]

    def _on_exit(self, name: str, output: torch.Tensor) -> None:
        state = self._active.pop(name)
        if "z" not in state:
            raise RuntimeError(f"missing MBConv/FFN boundary for block {name}")
        ratios = relative_residual_norms(state["x"], state["z"], output)
        for key, vec in zip(("mbconv", "ffn", "total"), ratios):
            self._observations[name][key].extend(
                float(value) for value in vec.detach().cpu().tolist()
            )

    @contextmanager
    def capture(self):
        """Attach to model only during inference and ALWAYS clean up hooks."""
        if self._handles:
            raise RuntimeError("residual hooks already installed")
        if self._active:
            raise RuntimeError("previous block observation still pending")
        try:
            for name, block in self.blocks.items():
                self._handles.append(
                    block.register_forward_pre_hook(
                        lambda m, a, name=name: self._on_enter(name, a)
                    )
                )
                self._handles.append(
                    block.norm2.register_forward_pre_hook(
                        lambda m, a, name=name: self._before_norm2(name, a)
                    )
                )
                self._handles.append(
                    block.register_forward_hook(
                        lambda m, a, output, name=name: self._on_exit(name, output)
                    )
                )
            yield self
            if self._active:
                raise RuntimeError(
                    f"unfinished residual hooks: {sorted(self._active)}"
                )
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._active.clear()

    def summary(self) -> dict:
        if set(self._observations) != set(self.blocks):
            missing = sorted(set(self.blocks) - set(self._observations))
            raise RuntimeError(f"not all NAFBlocks executed: {missing}")
        sizes = {
            len(stats["mbconv"])
            for stats in self._observations.values()
        }
        if len(sizes) != 1 or not sizes or next(iter(sizes)) <= 0:
            raise RuntimeError("residual diagnostic sample counts disagree")
        sample_count = next(iter(sizes))
        blocks: dict[str, dict] = {}
        groups: dict[str, list[dict]] = defaultdict(list)
        for name, block in self.blocks.items():
            observations = self._observations[name]
            group = ".".join(name.split(".")[:2]) if name.startswith(
                ("opt_encoders.", "sar_encoders.", "decoders.")
            ) else name.split(".")[0]
            record = {
                "stage": group,
                "beta": float(block.beta.detach().cpu().item()),
                "gamma": float(block.gamma.detach().cpu().item()),
                "mbconv_relative_l2_mean": sum(observations["mbconv"]) / sample_count,
                "ffn_relative_l2_mean": sum(observations["ffn"]) / sample_count,
                "total_relative_l2_mean": sum(observations["total"]) / sample_count,
            }
            blocks[name] = record
            groups[group].append(record)

        keys = (
            "mbconv_relative_l2_mean",
            "ffn_relative_l2_mean",
            "total_relative_l2_mean",
        )
        stage_summary = {}
        for group, entries in sorted(groups.items()):
            summary = {
                "num_blocks": len(entries),
                "mean_abs_beta": sum(abs(e["beta"]) for e in entries) / len(entries),
                "mean_abs_gamma": sum(abs(e["gamma"]) for e in entries) / len(entries),
            }
            for key in keys:
                vals = [entry[key] for entry in entries]
                summary[key] = sum(vals) / len(vals)
                summary[key.replace("_mean", "_min_block_mean")] = min(vals)
                summary[key.replace("_mean", "_max_block_mean")] = max(vals)
            stage_summary[group] = summary

        return {
            "num_samples": sample_count,
            "num_blocks": len(blocks),
            "normalization": {
                "mbconv_relative_l2": "||z-x||_2/(||x||_2+1e-12)",
                "ffn_relative_l2": "||y-z||_2/(||z||_2+1e-12)",
                "total_relative_l2": "||y-x||_2/(||x||_2+1e-12)",
                "averaging": "mean of per-sample ratios; stage values average block means",
            },
            "stages": stage_summary,
            "blocks": blocks,
        }
