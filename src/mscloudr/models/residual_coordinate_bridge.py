"""SAR-conditioned residual-coordinate bridge for CanonicalBridgeNet.

The bridge remains on the clean-to-cloudy residual direction, but paired SAR
learns a spatially varying effective corruption coordinate.

For nominal physical alpha and gate g(y,z) in [-1,1]:

    lambda(alpha,y,z)
        = alpha + kappa * 4*alpha*(1-alpha) * g(y,z)

and the optical bridge state is

    x_alpha = (1-lambda) * x0 + lambda * y.

The gate is one spatial channel shared across all 13 optical bands. Therefore
SAR can accelerate or retard bridge progress locally without inventing an
arbitrary spectral displacement direction.

For 0 < kappa < 0.25, lambda remains in [0,1] and is monotone in alpha for a
fixed gate value. The endpoints are exact: lambda(0)=0 and lambda(1)=1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .canonical_bridge import CanonicalBridgeNet


CANONICAL_SAR_RESIDUAL_COORDINATE_MODEL_IDENTITY = (
    "canonical_sar_residual_coordinate_bridge_net"
)


class SARResidualCoordinateGate(nn.Module):
    """Predict one signed spatial bridge-coordinate gate from cloudy S2 + SAR."""

    def __init__(
        self,
        in_opt: int = 13,
        in_sar: int = 2,
        hidden: int = 32,
    ):
        super().__init__()
        self.in_opt = int(in_opt)
        self.in_sar = int(in_sar)

        self.conv1 = nn.Conv2d(in_opt + in_sar, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.act = nn.SiLU()

        # Begin exactly at the straight canonical bridge.
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
        return torch.tanh(self.head(features))


class CanonicalSARResidualCoordinateBridgeNet(CanonicalBridgeNet):
    """CanonicalBridgeNet with SAR-controlled spatial residual progress."""

    def __init__(
        self,
        *,
        in_opt: int = 13,
        in_sar: int = 2,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        alpha_dim: int = 128,
        total_steps: int = 1000,
        mid_blocks: int = 2,
        gate_hidden: int = 32,
        residual_coordinate_kappa: float = 0.10,
    ):
        kappa = float(residual_coordinate_kappa)
        if not (0.0 < kappa < 0.25):
            raise ValueError(
                "residual_coordinate_kappa must satisfy 0 < kappa < 0.25"
            )

        super().__init__(
            in_opt=in_opt,
            in_sar=in_sar,
            widths=widths,
            alpha_dim=alpha_dim,
            total_steps=total_steps,
            mid_blocks=mid_blocks,
        )

        self.sar_residual_coordinate_gate = SARResidualCoordinateGate(
            in_opt=in_opt,
            in_sar=in_sar,
            hidden=gate_hidden,
        )
        self.residual_coordinate_kappa = kappa

    def bridge_coordinate_gate(
        self,
        cloudy: torch.Tensor,
        sar: torch.Tensor,
    ) -> torch.Tensor:
        """Return g(y,z) with shape [B,1,H,W] and values in [-1,1]."""

        return self.sar_residual_coordinate_gate(cloudy, sar)

    def bridge_effective_alpha(
        self,
        cloudy: torch.Tensor,
        sar: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Return spatial lambda(alpha,y,z) used by the residual bridge."""

        gate = self.bridge_coordinate_gate(cloudy, sar)
        alpha = alpha.to(
            device=cloudy.device,
            dtype=cloudy.dtype,
        )
        if alpha.ndim == 0:
            alpha = alpha.view(1, 1, 1, 1)
        elif alpha.ndim == 1:
            alpha = alpha.view(-1, 1, 1, 1)
        elif alpha.ndim != 4:
            raise ValueError("alpha must be scalar, [B], or [B,1,1,1]")

        if alpha.shape[0] not in {1, cloudy.shape[0]}:
            raise ValueError(
                "alpha batch dimension must be 1 or match cloudy batch size"
            )

        envelope = 4.0 * alpha * (1.0 - alpha)
        effective = (
            alpha
            + self.residual_coordinate_kappa * envelope * gate
        )

        # The kappa constraint guarantees these analytically; clamps only absorb
        # floating-point endpoint roundoff.
        return effective.clamp(0.0, 1.0)


def count_canonical_sar_residual_coordinate_parameters(
    model: nn.Module | None = None,
) -> int:
    if model is None:
        model = CanonicalSARResidualCoordinateBridgeNet()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
