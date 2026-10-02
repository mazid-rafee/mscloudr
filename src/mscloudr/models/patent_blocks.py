"""Patent-disclosed DB-CR time embedding and time-conditioned NAFBlock.

Source basis: U.S. patent application US20260289738A1, especially FIG. 7A,
FIG. 7B, and the accompanying description.

The disclosure fixes several details that were ambiguous in the paper:
- time conditioning is derived from alpha_t, not directly from raw t;
- alpha_t is sinusoidally position-embedded;
- the time MLP has three Linear layers with SimpleGates between them;
- its output is four C-vectors: Scale_conv, Shift_conv, Scale_FFN, Shift_FFN;
- scale/shift is applied to normalized features before MBConv and FFN;
- normalization is over all elements in each feature map;
- beta and gamma are described as learnable scalar residual weights.

The patent does NOT state the positional-embedding width, hidden width, or
whether the time MLP is shared between NAFBlocks. Those remain explicit
configuration choices here and must not be presented as recovered facts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sine_bridge_alpha(
    t: torch.Tensor,
    total_steps: float | int,
) -> torch.Tensor:
    """Original DB-CR alpha_t = sin(pi t / (2T))."""

    if float(total_steps) <= 0:
        raise ValueError("total_steps must be positive")
    return torch.sin((t.float() / float(total_steps)) * math.pi / 2.0)


def sinusoidal_scalar_embedding(
    value: torch.Tensor,
    dim: int,
    *,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Sinusoidally embed one scalar per batch item."""

    if value.ndim != 1:
        raise ValueError(
            f"value must have shape [B], got {tuple(value.shape)}"
        )
    if dim <= 0:
        raise ValueError("dim must be positive")

    half = dim // 2
    if half == 0:
        return value[:, None]

    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(
            half,
            device=value.device,
            dtype=value.dtype,
        )
        / float(half)
    )
    args = value[:, None] * freqs[None]
    embedding = torch.cat(
        (torch.sin(args), torch.cos(args)),
        dim=-1,
    )
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class VectorSimpleGate(nn.Module):
    """SimpleGate over the final dimension of a vector embedding."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                "VectorSimpleGate requires an even final dimension"
            )
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * x2


class PatentFeatureMapLayerNorm(nn.Module):
    """Whole-feature-map normalization described by the DB-CR patent.

    The patent says the mean and standard deviation are computed from all
    elements in a feature map. For NCHW tensors this means normalization over
    C, H, and W independently for every batch item.

    Per-channel affine parameters are retained because the disclosed operator
    is called LayerNorm and the historical implementation used affine
    GroupNorm(1, C). The reduction axes, not the affine parameterization, are
    the important newly recovered detail.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.eps = float(eps)
        self.affine = bool(affine)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.channels))
            self.bias = nn.Parameter(torch.zeros(self.channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        variance = (x - mean).square().mean(
            dim=(1, 2, 3),
            keepdim=True,
        )
        y = (x - mean) / torch.sqrt(variance + self.eps)

        if self.affine:
            y = (
                y * self.weight.view(1, -1, 1, 1)
                + self.bias.view(1, -1, 1, 1)
            )
        return y


@dataclass(frozen=True)
class PatentTimeEmbeddingConfig:
    """Dimensions not specified by the patent.

    Defaults of 128 are carried over only as a diagnostic continuity choice
    from the historical ms-cloudR time_dim=128 implementation.
    """

    positional_dim: int = 128
    hidden_dim: int = 128

    def validate(self) -> None:
        if self.positional_dim <= 0:
            raise ValueError("positional_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")


class PatentTimeEmbedding(nn.Module):
    """Three-Linear/SimpleGate time embedding disclosed in FIG. 7A."""

    def __init__(
        self,
        channels: int,
        config: PatentTimeEmbeddingConfig | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.channels = int(channels)
        self.config = config or PatentTimeEmbeddingConfig()
        self.config.validate()

        embedding_dim = self.config.positional_dim
        hidden_dim = self.config.hidden_dim

        # The patent specifies three Linear layers and a SimpleGate between
        # each adjacent pair. It does not specify the intermediate widths.
        # 2*hidden -> SimpleGate -> hidden is the minimal explicit realization.
        self.linear1 = nn.Linear(
            embedding_dim,
            2 * hidden_dim,
        )
        self.gate1 = VectorSimpleGate()
        self.linear2 = nn.Linear(
            hidden_dim,
            2 * hidden_dim,
        )
        self.gate2 = VectorSimpleGate()
        self.linear3 = nn.Linear(
            hidden_dim,
            4 * self.channels,
        )

    def from_alpha(
        self,
        alpha: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Produce Scale/Shift vectors from a supplied bridge alpha.

        This entry point is important for later schedule experiments: if alpha
        changes, the conditioning should follow the actual bridge state rather
        than silently continuing to encode raw timestep.
        """

        if alpha.ndim != 1:
            raise ValueError(
                f"alpha must have shape [B], got {tuple(alpha.shape)}"
            )

        embedding = sinusoidal_scalar_embedding(
            alpha.float(),
            self.config.positional_dim,
        )
        hidden = self.gate1(self.linear1(embedding))
        hidden = self.gate2(self.linear2(hidden))
        modulation = self.linear3(hidden).view(
            alpha.shape[0],
            4,
            self.channels,
            1,
            1,
        )
        return tuple(
            modulation[:, index]
            for index in range(4)
        )

    def forward(
        self,
        t: torch.Tensor,
        total_steps: float | int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        alpha = sine_bridge_alpha(t, total_steps)
        return self.from_alpha(alpha)


class PatentSimplifiedChannelAttention(nn.Module):
    """SCA using the disclosed learnable channel vector w in R^C."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.weight = nn.Parameter(
            torch.ones(1, channels, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(
            dim=(2, 3),
            keepdim=True,
        )
        return x * (self.weight * pooled)


class PatentTimeEmbeddedNAFBlock(nn.Module):
    """Time-conditioned NAFBlock following patent Eq. (14).

    The textual list in the patent names a 1x1 convolution before SaS, while
    Eq. (14) places SaS immediately after LayerNorm and before MBConv/FFN.
    Because the four disclosed modulation vectors each have dimension C,
    Eq. (14) is dimensionally unambiguous; this implementation follows Eq. (14).

    The patent lists one 1x1 convolution in MBConv before the depthwise
    convolution and does not state a post-SCA projection. The FFN explicitly
    contains two 1x1 convolutions. This module follows that disclosed topology.

    By default each block owns a time-embedding MLP. Set own_time_embedding
    to False to study sharing without changing the NAFBlock computation.
    """

    def __init__(
        self,
        channels: int,
        *,
        time_config: PatentTimeEmbeddingConfig | None = None,
        own_time_embedding: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.channels = int(channels)
        self.time_embedding = (
            PatentTimeEmbedding(
                self.channels,
                time_config,
            )
            if own_time_embedding
            else None
        )

        self.norm1 = PatentFeatureMapLayerNorm(
            self.channels
        )
        self.conv1 = nn.Conv2d(
            self.channels,
            2 * self.channels,
            kernel_size=1,
        )
        self.depthwise = nn.Conv2d(
            2 * self.channels,
            2 * self.channels,
            kernel_size=3,
            padding=1,
            groups=2 * self.channels,
        )
        self.sca = PatentSimplifiedChannelAttention(
            self.channels
        )
        self.beta = nn.Parameter(torch.zeros(()))

        self.norm2 = PatentFeatureMapLayerNorm(
            self.channels
        )
        self.ffn1 = nn.Conv2d(
            self.channels,
            2 * self.channels,
            kernel_size=1,
        )
        self.ffn2 = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
        )
        self.gamma = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _simple_gate_channels(
        x: torch.Tensor,
    ) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

    @staticmethod
    def _scale_shift(
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return x * scale + shift

    def forward_with_modulation(
        self,
        x: torch.Tensor,
        modulation: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            scale_conv,
            shift_conv,
            scale_ffn,
            shift_ffn,
        ) = modulation

        y = self._scale_shift(
            self.norm1(x),
            scale_conv,
            shift_conv,
        )
        y = self.conv1(y)
        y = self.depthwise(y)
        y = self._simple_gate_channels(y)
        y = self.sca(y)
        z_mbconv = x + self.beta * y

        y = self._scale_shift(
            self.norm2(z_mbconv),
            scale_ffn,
            shift_ffn,
        )
        y = self.ffn1(y)
        y = self._simple_gate_channels(y)
        y = self.ffn2(y)
        return z_mbconv + self.gamma * y

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        total_steps: float | int | None = None,
        *,
        alpha: torch.Tensor | None = None,
        modulation: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ) -> torch.Tensor:
        if modulation is None:
            if self.time_embedding is None:
                raise ValueError(
                    "modulation is required when own_time_embedding=False"
                )

            if alpha is not None:
                modulation = self.time_embedding.from_alpha(
                    alpha
                )
            else:
                if t is None or total_steps is None:
                    raise ValueError(
                        "provide alpha or both t and total_steps"
                    )
                modulation = self.time_embedding(
                    t,
                    total_steps,
                )

        return self.forward_with_modulation(
            x,
            modulation,
        )


def count_trainable_parameters(
    module: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
