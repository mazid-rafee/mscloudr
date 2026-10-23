"""Orthogonally projected, independently supervised CanonicalBridge heads.

Fixed Haar-scale low/high subspaces; this is a *controlled research hypothesis*,
not a validated novel diffusion-bridge theory. Geometry remains straight and
the network continues to output direct clean optical x0 at NFE=1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .canonical_bridge import CanonicalBridgeNet


COMPONENT_MODEL_IDENTITY = "canonical_component_bridge_v1"


def low_pass_projection(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection onto piecewise constant 2x2 pixel blocks."""
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("2x2 projection requires NCHW and even spatial dimensions")
    return F.avg_pool2d(x, kernel_size=2, stride=2).repeat_interleave(
        2, dim=-2
    ).repeat_interleave(2, dim=-1)


def high_pass_projection(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal complement of low_pass_projection."""
    return x - low_pass_projection(x)


class CanonicalComponentBridgeNet(CanonicalBridgeNet):
    """Original CNN backbone with two projected optical output heads.

    Identical shared encoder, SAR fusion, decoder and time conditioning to
    CanonicalBridgeNet(cnn). Only the final output head has 2*C channels.
    """

    def __init__(
        self,
        *,
        in_opt: int = 13,
        in_sar: int = 2,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        alpha_dim: int = 128,
        total_steps: int = 1000,
        mid_blocks: int = 2,
    ):
        super().__init__(
            in_opt=in_opt, in_sar=in_sar, widths=widths, alpha_dim=alpha_dim,
            total_steps=total_steps, mid_blocks=mid_blocks,
        )
        self.head = nn.Conv2d(self.widths[0], 2 * in_opt, kernel_size=3, padding=1)

    def forward_components(
        self, x_alpha: torch.Tensor, conditioning: torch.Tensor, sar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = super().forward(x_alpha, conditioning, sar)
        raw_low, raw_high = outputs.chunk(2, dim=1)
        return low_pass_projection(raw_low), high_pass_projection(raw_high)

    def forward(
        self, x_alpha: torch.Tensor, conditioning: torch.Tensor, sar: torch.Tensor
    ) -> torch.Tensor:
        low, high = self.forward_components(x_alpha, conditioning, sar)
        return low + high
