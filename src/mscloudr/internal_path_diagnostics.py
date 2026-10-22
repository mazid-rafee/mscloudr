"""Read-only forward activation and backward gradient probes of patent NAFBlocks.

The patent time MLP is called through .from_alpha(), not nn.Module.forward(),
so the time-vector probe correctly attaches to its final Linear layer.
Everything is observed via hooks; no model code or weights are modified.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import math

import torch

from mscloudr.models.patent_blocks import PatentTimeEmbeddedNAFBlock


def stage_name(name: str) -> str:
    parts = name.split(".")
    if parts[0] in ("opt_encoders", "sar_encoders", "decoders"):
        return ".".join(parts[:2])
    return parts[0]


def rms(t: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS across all non-batch dimensions, float32 reduction."""
    return t.detach().float().square().flatten(1).mean(1).sqrt()


def parameter_grad_rms(parameter: torch.nn.Parameter) -> float | None:
    if parameter.grad is None:
        return None
    value = parameter.grad.detach().float().square().mean().sqrt().item()
    return float(value)


class InternalPathRecorder:
    """Measure the time, MBConv, SCA, FFN, and gradient paths per NAFBlock."""

    GRAD_PARAMETERS = (
        "beta", "gamma", "conv1.weight", "depthwise.weight",
        "sca.weight", "ffn1.weight", "ffn2.weight",
        "time_embedding.linear3.weight",
    )

    def __init__(self, model: torch.nn.Module) -> None:
        self.blocks = {
            name: module for name, module in model.named_modules()
            if isinstance(module, PatentTimeEmbeddedNAFBlock)
        }
        if not self.blocks:
            raise ValueError("No time-conditioned NAFBlocks found")
        if any(block.time_embedding is None for block in self.blocks.values()):
            raise ValueError("This diagnostic currently requires per_block time embedding")
        self._handles = []
        self._tensor_handles = []
        self.forward = defaultdict(lambda: defaultdict(list))
        self.backward = defaultdict(lambda: defaultdict(list))
        self.batch_count = 0

    def _record(self, dest: dict, key: str, tensor: torch.Tensor) -> None:
        means = rms(tensor)
        values = means.cpu().tolist()
        if not all(math.isfinite(float(v)) for v in values):
            raise RuntimeError(f"nonfinite diagnostic: {key}")
        dest[key].extend(float(v) for v in values)

    def _scalar(self, dest: dict, key: str, tensor: torch.Tensor) -> None:
        values = tensor.detach().float().reshape(-1).cpu().tolist()
        if not all(math.isfinite(float(v)) for v in values):
            raise RuntimeError(f"nonfinite diagnostic: {key}")
        dest[key].extend(float(v) for v in values)

    def _time_mlp(self, name: str, raw: torch.Tensor) -> None:
        b = raw.shape[0]
        if raw.ndim != 2 or raw.shape[1] % 4 != 0:
            raise RuntimeError("unexpected time-MLP output shape")
        modulation = raw.detach().float().reshape(b, 4, -1)
        dest = self.forward[name]
        labels = ("scale_conv", "shift_conv", "scale_ffn", "shift_ffn")
        for i, label in enumerate(labels):
            self._scalar(
                dest, label + "_abs_mean",
                modulation[:, i].abs().mean(1)
            )
            self._scalar(
                dest, label + "_rms",
                modulation[:, i].square().mean(1).sqrt()
            )

    def _before_sca(self, name: str, block: PatentTimeEmbeddedNAFBlock,
                    inputs: tuple) -> None:
        x = inputs[0]
        self._record(self.forward[name], "after_mb_simplegate_rms", x)
        channel_weight = block.sca.weight.detach().float()
        gate = channel_weight * x.detach().float().mean(dim=(2, 3), keepdim=True)
        self._scalar(
            self.forward[name], "sca_gate_abs_mean",
            gate.abs().flatten(1).mean(1)
        )
        self._scalar(
            self.forward[name], "sca_gate_rms",
            gate.square().flatten(1).mean(1).sqrt()
        )

    def _after_sca(self, name: str, block: PatentTimeEmbeddedNAFBlock,
                   output: torch.Tensor) -> None:
        self._record(self.forward[name], "after_sca_rms", output)
        self._record(
            self.forward[name], "beta_scaled_sca_rms",
            output * block.beta.detach()
        )

    def _after_ffn(self, name: str, block: PatentTimeEmbeddedNAFBlock,
                   output: torch.Tensor) -> None:
        self._record(self.forward[name], "ffn_raw_output_rms", output)
        self._record(
            self.forward[name], "gamma_scaled_ffn_rms",
            output * block.gamma.detach()
        )

    def _on_input(self, name: str, inputs: tuple) -> None:
        x = inputs[0]
        self._record(self.forward[name], "block_input_rms", x)
        if not x.requires_grad:
            raise RuntimeError(f"block input has no autograd graph: {name}")
        handle = x.register_hook(
            lambda grad, name=name: self._record(
                self.backward[name], "input_grad_rms", grad
            )
        )
        self._tensor_handles.append(handle)

    @contextmanager
    def capture(self):
        if self._handles:
            raise RuntimeError("hooks already installed")
        try:
            for name, block in self.blocks.items():
                def add(module, fn):
                    self._handles.append(module.register_forward_hook(fn))

                def add_pre(module, fn):
                    self._handles.append(module.register_forward_pre_hook(fn))

                add_pre(block, lambda m, a, name=name: self._on_input(name, a))
                add(block.time_embedding.linear3, lambda m, a, y, name=name:
                    self._time_mlp(name, y))
                add(block.norm1, lambda m, a, y, name=name:
                    self._record(self.forward[name], "norm1_rms", y))
                add_pre(block.conv1, lambda m, a, name=name:
                    self._record(self.forward[name], "mod_conv_rms", a[0]))
                add(block.depthwise, lambda m, a, y, name=name:
                    self._record(self.forward[name], "after_depthwise_rms", y))
                add_pre(block.sca, lambda m, a, name=name, block=block:
                    self._before_sca(name, block, a))
                add(block.sca, lambda m, a, y, name=name, block=block:
                    self._after_sca(name, block, y))
                add(block.norm2, lambda m, a, y, name=name:
                    self._record(self.forward[name], "norm2_rms", y))
                add_pre(block.ffn1, lambda m, a, name=name:
                    self._record(self.forward[name], "mod_ffn_rms", a[0]))
                add(block.ffn1, lambda m, a, y, name=name:
                    self._record(self.forward[name], "ffn_pre_gate_rms", y))
                add_pre(block.ffn2, lambda m, a, name=name:
                    self._record(self.forward[name], "ffn_post_gate_rms", a[0]))
                add(block.ffn2, lambda m, a, y, name=name, block=block:
                    self._after_ffn(name, block, y))
            yield self
        finally:
            for h in self._handles:
                h.remove()
            for h in self._tensor_handles:
                h.remove()
            self._handles.clear()
            self._tensor_handles.clear()

    def record_parameter_grads(self) -> None:
        """Call immediately after .backward() and before zero_grad()."""
        for name, block in self.blocks.items():
            for parameter_name in self.GRAD_PARAMETERS:
                obj = block
                for attr in parameter_name.split("."):
                    obj = getattr(obj, attr)
                value = parameter_grad_rms(obj)
                self.backward[name][parameter_name + "_grad_rms"].append(value)
        self.batch_count += 1

    @staticmethod
    def _average(values: list) -> float | None:
        present = [v for v in values if v is not None]
        if not present:
            return None
        return float(sum(present) / len(present))

    def summary(self) -> dict:
        if self.batch_count < 1:
            raise RuntimeError("no backward pass recorded")
        if set(self.forward) != set(self.blocks):
            raise RuntimeError("some blocks did not execute in forward")
        if set(self.backward) != set(self.blocks):
            raise RuntimeError("some blocks did not execute in backward")
        blocks = {}
        stages = defaultdict(list)
        for name, block in self.blocks.items():
            measurements = {
                **{
                    key + "_mean": self._average(values)
                    for key, values in self.forward[name].items()
                },
                **{
                    key + "_mean": self._average(values)
                    for key, values in self.backward[name].items()
                },
                "beta_abs": abs(float(block.beta.detach().item())),
                "gamma_abs": abs(float(block.gamma.detach().item())),
                "stage": stage_name(name),
            }
            if len(self.backward[name]["input_grad_rms"]) < self.batch_count:
                raise RuntimeError(f"missing input gradients: {name}")
            blocks[name] = measurements
            stages[stage_name(name)].append(measurements)

        stage_stats = {}
        keys = [
            key for key in next(iter(blocks.values()))
            if key.endswith("_mean")
        ]
        for stage, entries in sorted(stages.items()):
            stage_stats[stage] = {"num_blocks": len(entries)}
            for key in keys:
                values = [e.get(key) for e in entries if e.get(key) is not None]
                stage_stats[stage][key] = (
                    sum(values) / len(values) if values else None
                )
            for key in ("beta_abs", "gamma_abs"):
                stage_stats[stage]["mean_" + key] = (
                    sum(entry[key] for entry in entries) / len(entries)
                )
        return {
            "num_blocks": len(blocks),
            "num_batches": self.batch_count,
            "stages": stage_stats,
            "blocks": blocks,
            "definitions": {
                "activation_rms": "sqrt(mean(tensor_elements**2)) per sample",
                "scale_rms": "sqrt(mean(scale_channels**2)) per sample",
                "sca_gate": "sca.weight * mean(simplegate_output, spatial_dims)",
                "gradient_rms": "sqrt(mean(gradient_elements**2)) per batch",
                "gradient_caveat": (
                    "Gradient norms from one endpoint L1 backward pass. "
                    "Values depend on loss reduction and tensor size; "
                    "not optimizer steps or reliable learning speed indicators."
                ),
            },
        }
