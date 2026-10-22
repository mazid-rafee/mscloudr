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
import math

import torch

from mscloudr.models.blocks import SFBlock
from mscloudr.models.patent_blocks import PatentTimeEmbeddedNAFBlock

NAF_EXTRA_KEYS = (
    "input_l2", "post_mbconv_l2", "output_l2",
    "mbconv_update_l2", "ffn_update_l2", "total_update_l2",
    "input_rms", "mbconv_update_rms", "ffn_update_rms", "total_update_rms",
)
FUSION_KEYS = (
    "optical_input_l2", "sar_input_l2", "output_l2", "output_minus_optical_l2",
    "optical_input_rms", "sar_input_rms", "output_rms",
    "output_minus_optical_rms", "output_minus_optical_relative_l2",
)


def _rms_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    """L2 divided by sqrt(C*H*W), comparable across feature resolutions."""
    return _norm_per_sample(tensor) / math.sqrt(tensor[0].numel())


def _append_values(observations: dict, measures: dict) -> None:
    for key, tensor in measures.items():
        observations[key].extend(
            float(value) for value in tensor.detach().cpu().tolist()
        )



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
        self.fusions: dict[str, SFBlock] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, SFBlock)
        }
        self._handles: list = []
        self._active: dict[str, dict[str, torch.Tensor]] = {}
        self._fusion_active: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._observations = defaultdict(
            lambda: {
                key: []
                for key in ("mbconv", "ffn", "total", *NAF_EXTRA_KEYS)
            }
        )
        self._fusion_observations = defaultdict(
            lambda: {key: [] for key in FUSION_KEYS}
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
        x, z, y = state["x"], state["z"], output
        ratios = relative_residual_norms(x, z, y)
        mb_delta = z.float() - x.float()
        ffn_delta = y.float() - z.float()
        total_delta = y.float() - x.float()
        measures = {
            **dict(zip(("mbconv", "ffn", "total"), ratios)),
            "input_l2": _norm_per_sample(x),
            "post_mbconv_l2": _norm_per_sample(z),
            "output_l2": _norm_per_sample(y),
            "mbconv_update_l2": _norm_per_sample(mb_delta),
            "ffn_update_l2": _norm_per_sample(ffn_delta),
            "total_update_l2": _norm_per_sample(total_delta),
            "input_rms": _rms_per_sample(x),
            "mbconv_update_rms": _rms_per_sample(mb_delta),
            "ffn_update_rms": _rms_per_sample(ffn_delta),
            "total_update_rms": _rms_per_sample(total_delta),
        }
        _append_values(self._observations[name], measures)

    def _fusion_enter(self, name: str, inputs: tuple) -> None:
        if name in self._fusion_active:
            raise RuntimeError(f"SFBlock was re-entered: {name}")
        if len(inputs) != 2 or inputs[0].shape != inputs[1].shape:
            raise ValueError(f"invalid fusion inputs for {name}")
        self._fusion_active[name] = (inputs[0], inputs[1])

    def _fusion_exit(self, name: str, output: torch.Tensor) -> None:
        optical, sar = self._fusion_active.pop(name)
        if output.shape != optical.shape:
            raise ValueError(f"invalid fusion output for {name}")
        delta = output.float() - optical.float()
        optical_l2 = _norm_per_sample(optical)
        measures = {
            "optical_input_l2": optical_l2,
            "sar_input_l2": _norm_per_sample(sar),
            "output_l2": _norm_per_sample(output),
            "output_minus_optical_l2": _norm_per_sample(delta),
            "optical_input_rms": _rms_per_sample(optical),
            "sar_input_rms": _rms_per_sample(sar),
            "output_rms": _rms_per_sample(output),
            "output_minus_optical_rms": _rms_per_sample(delta),
            "output_minus_optical_relative_l2":
                _norm_per_sample(delta) / (optical_l2 + 1e-12),
        }
        _append_values(self._fusion_observations[name], measures)

    @contextmanager
    def capture(self):
        """Attach to model only during inference and ALWAYS clean up hooks."""
        if self._handles:
            raise RuntimeError("residual hooks already installed")
        if self._active or self._fusion_active:
            raise RuntimeError("previous block/fusion observation still pending")
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
            for name, fusion in self.fusions.items():
                self._handles.append(
                    fusion.register_forward_pre_hook(
                        lambda m, a, name=name: self._fusion_enter(name, a)
                    )
                )
                self._handles.append(
                    fusion.register_forward_hook(
                        lambda m, a, output, name=name: self._fusion_exit(name, output)
                    )
                )
            yield self
            if self._active or self._fusion_active:
                raise RuntimeError(
                    "unfinished hooks: "
                    f"NAF={sorted(self._active)}, fusion={sorted(self._fusion_active)}"
                )
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._active.clear()
            self._fusion_active.clear()

    def summary(self) -> dict:
        if set(self._observations) != set(self.blocks):
            missing = sorted(set(self.blocks) - set(self._observations))
            raise RuntimeError(f"not all NAFBlocks executed: {missing}")
        sizes = {
            len(values)
            for stats in self._observations.values()
            for values in stats.values()
        }
        sizes.update(
            len(values)
            for stats in self._fusion_observations.values()
            for values in stats.values()
        )
        if set(self._fusion_observations) != set(self.fusions):
            missing = sorted(set(self.fusions) - set(self._fusion_observations))
            raise RuntimeError(f"not all SFBlocks executed: {missing}")
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
            record.update({
                key + "_mean": sum(observations[key]) / sample_count
                for key in NAF_EXTRA_KEYS
            })
            blocks[name] = record
            groups[group].append(record)

        keys = (
            "mbconv_relative_l2_mean",
            "ffn_relative_l2_mean",
            "total_relative_l2_mean",
            *(key + "_mean" for key in NAF_EXTRA_KEYS),
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

        fusion_summary = {}
        for name, observations in sorted(self._fusion_observations.items()):
            fusion_summary[name] = {
                key + "_mean": sum(observations[key]) / sample_count
                for key in FUSION_KEYS
            }
        return {
            "num_samples": sample_count,
            "num_blocks": len(blocks),
            "num_fusion_blocks": len(fusion_summary),
            "normalization": {
                "mbconv_relative_l2": "||z-x||_2/(||x||_2+1e-12)",
                "ffn_relative_l2": "||y-z||_2/(||z||_2+1e-12)",
                "total_relative_l2": "||y-x||_2/(||x||_2+1e-12)",
                "fusion_relative_l2": "||SF(opt,sar)-opt||_2/(||opt||_2+1e-12)",
                "raw_l2": "Euclidean norm across all C,H,W features",
                "rms": "raw L2 divided by sqrt(C*H*W)",
                "averaging": "mean of per-sample measurements; stage values average block means",
                "fusion_note": "Fusion output change is not the isolated SAR contribution. "
                               "SFBlock has projection and optical paths.",
            },
            "stages": stage_summary,
            "blocks": blocks,
            "fusion_blocks": fusion_summary,
        }
