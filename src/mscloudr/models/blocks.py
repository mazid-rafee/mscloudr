"""DB-CR building blocks reconstructed from paper-described architecture.

The base NAFBlock follows the official NAFNet implementation by
megvii-research/NAFNet and the LayerNorm placement described by DB-CR.

The SFBlock follows DB-CR Figure 3(b) and Eqs. (15)-(16):
- optical feature -> Q
- SAR feature -> K and V
- separate LayerNorm + 1x1 projection for Q/K/V
- channel-wise multi-head cross-attention
- optical residual after attention
- residual MLP per head
- concatenate heads and apply final 1x1 projection

DB-CR calls its NAFBlocks "time-embedded", but the paper does not specify the
exact injection operator/placement. Time conditioning is therefore deliberately
not guessed in this module; it will be added at network-integration time once an
explicit design choice is recorded.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW feature maps.

    This matches the normalization semantics used by the official NAFNet
    LayerNorm2d: for every spatial position, normalize over channels only.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"LayerNorm2d expects [N,C,H,W], received shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.weight.numel():
            raise ValueError(
                f"expected {self.weight.numel()} channels, got {x.shape[1]}"
            )

        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return (
            y * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


class SimpleGate(nn.Module):
    """Split channels in half and multiply element-wise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError("SimpleGate requires an even channel count")
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Base NAFBlock with LayerNorm2d and zero-initialized residual scales.

    Structure follows the official NAFNet block used by DB-CR as its base:
    LayerNorm -> pointwise conv -> depthwise 3x3 conv -> SimpleGate ->
    simplified channel attention -> pointwise conv -> residual, then
    LayerNorm -> pointwise FFN -> SimpleGate -> pointwise conv -> residual.

    No time embedding is injected here because DB-CR does not provide enough
    implementation detail to reconstruct that operation unambiguously.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dw_expand <= 0 or ffn_expand <= 0:
            raise ValueError("expansion factors must be positive")

        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        if dw_channels % 2 != 0 or ffn_channels % 2 != 0:
            raise ValueError("expanded channel counts must be even for SimpleGate")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channels // 2,
                dw_channels // 2,
                kernel_size=1,
                bias=True,
            ),
        )
        self.conv3 = nn.Conv2d(
            dw_channels // 2,
            channels,
            kernel_size=1,
            bias=True,
        )
        self.dropout1 = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(
            ffn_channels // 2,
            channels,
            kernel_size=1,
            bias=True,
        )
        self.dropout2 = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        residual = x + y * self.beta

        y = self.conv4(self.norm2(residual))
        y = self.sg2(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        return residual + y * self.gamma


class _HeadMLP(nn.Module):
    """Two-layer GELU MLP used inside one SFBlock attention head."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SFBlock(nn.Module):
    """Paper-described DB-CR SAR Fusion Block.

    Multi-head interpretation:
    ``channels`` denotes the full feature width and each head operates on
    ``channels // heads`` channels. This is the standard multi-head partition
    consistent with the paper's per-head description. Head outputs are
    concatenated and projected with a final 1x1 convolution.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        mlp_expand: int = 2,
    ) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0:
            raise ValueError("channels and heads must be positive")
        if channels % heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by heads ({heads})"
            )
        if mlp_expand <= 0:
            raise ValueError("mlp_expand must be positive")

        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = self.channels // self.heads

        # Figure 3(b) depicts a distinct Norm -> 1x1 path for Q, K, and V.
        self.norm_q = LayerNorm2d(channels)
        self.norm_k = LayerNorm2d(channels)
        self.norm_v = LayerNorm2d(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.mlps = nn.ModuleList(
            [_HeadMLP(self.head_dim, expansion=mlp_expand) for _ in range(heads)]
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(
        self,
        x_opt: torch.Tensor,
        x_sar: torch.Tensor,
    ) -> torch.Tensor:
        if x_opt.shape != x_sar.shape:
            raise ValueError(
                "optical and SAR feature shapes must match; "
                f"got {tuple(x_opt.shape)} vs {tuple(x_sar.shape)}"
            )
        if x_opt.ndim != 4:
            raise ValueError(
                f"SFBlock expects [N,C,H,W], received shape={tuple(x_opt.shape)}"
            )
        if x_opt.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {x_opt.shape[1]}"
            )

        batch, channels, height, width = x_opt.shape
        spatial = height * width

        q = self.q(self.norm_q(x_opt))
        k = self.k(self.norm_k(x_sar))
        v = self.v(self.norm_v(x_sar))

        # [B, heads, head_dim, HW]
        q = q.reshape(batch, self.heads, self.head_dim, spatial)
        k = k.reshape(batch, self.heads, self.head_dim, spatial)
        v = v.reshape(batch, self.heads, self.head_dim, spatial)

        # Paper Eq. (15): Q^T K forms channel-channel attention. In the above
        # layout, summing over HW gives [B, heads, head_dim, head_dim].
        scores = torch.einsum("bhcn,bhdn->bhcd", q, k)
        scores = scores / math.sqrt(float(self.head_dim))
        attention = torch.softmax(scores, dim=-1)

        # V * attention, returned to [B, heads, head_dim, H, W].
        attended = torch.einsum("bhcn,bhcd->bhdn", v, attention)
        attended = attended.reshape(
            batch, self.heads, self.head_dim, height, width
        )
        optical_heads = x_opt.reshape(
            batch, self.heads, self.head_dim, height, width
        )

        head_outputs = []
        for head_index, mlp in enumerate(self.mlps):
            z_sum = optical_heads[:, head_index] + attended[:, head_index]
            z_output = z_sum + mlp(z_sum)
            head_outputs.append(z_output)

        fused = torch.cat(head_outputs, dim=1)
        return self.proj(fused)


def count_trainable_parameters(module: nn.Module) -> int:
    """Count trainable parameters for audit/reporting."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
