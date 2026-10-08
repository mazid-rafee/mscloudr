"""SAR-aware bridge geometry for the CanonicalBridge backbone.

This experimental model keeps the existing two-channel Sentinel-1 SAR feature
fusion pathway and gives the same paired SAR observation a second role during
training: it bends the optical bridge trajectory.

For clean optical x0, cloudy optical y, physical coordinate alpha, and paired
SAR z=[VV,VH], the training state is

    x_alpha = (1-alpha) x0 + alpha y
              + kappa * 4 alpha (1-alpha) * phi(y, z)

where phi sees only cloudy optical and SAR.  It never receives x0.  Therefore
the bridge endpoints remain exact and the learned curvature cannot directly
encode the clean target.

At NFE=1 inference alpha=1, so the curvature term is exactly zero and the
restoration network starts from the observed cloudy optical image, as in the
straight CanonicalBridge baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .canonical_bridge import CanonicalBridgeNet


CANONICAL_DUAL_ROLE_SAR_BRIDGE_MODEL_IDENTITY = (
    "canonical_dual_role_sar_bridge_net"
)


class SARBridgeCurvature(nn.Module):
    """Predict a bounded 13-channel optical-space curvature from y and [VV,VH].

    The final convolution is zero initialized so training begins from the exact
    straight bridge.  The tanh bounds each normalized optical-band perturbation
    before multiplication by the fixed kappa envelope.
    """

    def __init__(
        self,
        in_opt: int = 13,
        in_sar: int = 2,
        hidden: int = 32,
        out_opt: int = 13,
        kappa: float = 0.05,
    ):
        super().__init__()
        if float(kappa) <= 0.0:
            raise ValueError("kappa must be positive")
        self.in_opt = int(in_opt)
        self.in_sar = int(in_sar)
        self.out_opt = int(out_opt)
        self.kappa = float(kappa)

        self.conv1 = nn.Conv2d(in_opt + in_sar, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.head = nn.Conv2d(hidden, out_opt, 3, padding=1)
        self.act = nn.SiLU()

        # Start exactly on the straight CanonicalBridge geometry.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        cloudy: torch.Tensor,
        sar: torch.Tensor,
    ) -> torch.Tensor:
        if cloudy.ndim != 4 or sar.ndim != 4:
            raise ValueError("cloudy and sar must have shape [B,C,H,W]")
        if cloudy.shape[0] != sar.shape[0]:
            raise ValueError("cloudy and sar batch sizes must match")
        if cloudy.shape[-2:] != sar.shape[-2:]:
            raise ValueError("cloudy and sar spatial sizes must match")
        if cloudy.shape[1] != self.in_opt:
            raise ValueError(
                f"cloudy must have {self.in_opt} channels, got {cloudy.shape[1]}"
            )
        if sar.shape[1] != self.in_sar:
            raise ValueError(
                f"sar must have {self.in_sar} channels, got {sar.shape[1]}"
            )

        features = torch.cat([cloudy, sar], dim=1)
        features = self.act(self.conv1(features))
        features = self.act(self.conv2(features))
        return self.kappa * torch.tanh(self.head(features))


class CanonicalDualRoleSARBridgeNet(CanonicalBridgeNet):
    """CanonicalBridgeNet with paired SAR used in fusion and bridge geometry."""

    def __init__(
        self,
        *,
        in_opt: int = 13,
        in_sar: int = 2,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        alpha_dim: int = 128,
        total_steps: int = 1000,
        mid_blocks: int = 2,
        sar_bridge_hidden: int = 32,
        sar_bridge_kappa: float = 0.05,
    ):
        super().__init__(
            in_opt=in_opt,
            in_sar=in_sar,
            widths=widths,
            alpha_dim=alpha_dim,
            total_steps=total_steps,
            mid_blocks=mid_blocks,
        )
        self.sar_bridge_curvature = SARBridgeCurvature(
            in_opt=in_opt,
            in_sar=in_sar,
            hidden=sar_bridge_hidden,
            out_opt=in_opt,
            kappa=sar_bridge_kappa,
        )
        self.sar_bridge_kappa = float(sar_bridge_kappa)

    def bridge_curvature(
        self,
        cloudy: torch.Tensor,
        sar: torch.Tensor,
    ) -> torch.Tensor:
        """Return phi(y,z), already bounded and scaled by kappa."""

        return self.sar_bridge_curvature(cloudy, sar)


def count_canonical_dual_role_sar_bridge_parameters(
    model: nn.Module | None = None,
) -> int:
    if model is None:
        model = CanonicalDualRoleSARBridgeNet()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
