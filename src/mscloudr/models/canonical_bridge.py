"""Architecture-independent bridge restoration backbone.

The model deliberately avoids DB-CR-specific NAF/SF blocks.  It consumes the
physical bridge state x_alpha, SAR guidance, and a physical-alpha conditioning
coordinate.  Bridge geometry and the training measure remain external to the
network so they can be varied independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


CANONICAL_BRIDGE_MODEL_IDENTITY = "canonical_bridge_net"


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.act(self.norm1(x)))
        y = self.conv2(self.act(self.norm2(y)))
        return x + y


class PhysicalAlphaEmbedding(nn.Module):
    """Embed the physical corruption coordinate alpha in [0, 1]."""

    def __init__(self, dim: int = 128, total_steps: int = 1000):
        super().__init__()
        self.total_steps = float(total_steps)
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.ndim != 1:
            conditioning = conditioning.reshape(conditioning.shape[0])
        alpha = conditioning.float() / self.total_steps
        alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)
        return self.net(alpha)


class AlphaFiLM(nn.Module):
    """Identity-initialized feature modulation from physical alpha."""

    def __init__(self, embedding_dim: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(embedding_dim, 2 * channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(emb).chunk(2, dim=-1)
        scale = scale[..., None, None]
        shift = shift[..., None, None]
        return x * (1.0 + scale) + shift


class GatedSARFusion(nn.Module):
    """Simple local SAR-to-optical gated additive fusion."""

    def __init__(self, channels: int):
        super().__init__()
        self.sar_proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.gate(torch.cat([optical, sar], dim=1))
        return optical + gate * self.sar_proj(sar)


class CanonicalBridgeNet(nn.Module):
    """Minimal dual-stream U-Net for controlled bridge experiments.

    Spatial hierarchy for a 256x256 input:
      stage 0: 32 x 256 x 256
      stage 1: 64 x 128 x 128
      stage 2: 128 x 64 x 64
      stage 3: 256 x 32 x 32

    The network predicts x0 directly.  It never receives schedule identity or
    raw diffusion timestep semantics; the supplied conditioning coordinate is
    interpreted as c=T*alpha and normalized back to physical alpha internally.
    """

    def __init__(
        self,
        in_opt: int = 13,
        in_sar: int = 2,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        alpha_dim: int = 128,
        total_steps: int = 1000,
        mid_blocks: int = 2,
    ):
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must contain at least two stages")

        self.in_opt = int(in_opt)
        self.in_sar = int(in_sar)
        self.widths = tuple(int(v) for v in widths)
        self.total_steps = int(total_steps)

        self.alpha_embedding = PhysicalAlphaEmbedding(
            dim=alpha_dim,
            total_steps=total_steps,
        )

        self.opt_stem = nn.Conv2d(in_opt, widths[0], 3, padding=1)
        self.sar_stem = nn.Conv2d(in_sar, widths[0], 3, padding=1)

        self.opt_blocks = nn.ModuleList()
        self.sar_blocks = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.enc_films = nn.ModuleList()
        self.opt_downs = nn.ModuleList()
        self.sar_downs = nn.ModuleList()

        for i, channels in enumerate(widths):
            self.opt_blocks.append(ResidualConvBlock(channels))
            self.sar_blocks.append(ResidualConvBlock(channels))
            self.fusions.append(GatedSARFusion(channels))
            self.enc_films.append(AlphaFiLM(alpha_dim, channels))
            if i < len(widths) - 1:
                next_channels = widths[i + 1]
                self.opt_downs.append(
                    nn.Conv2d(channels, next_channels, 3, stride=2, padding=1)
                )
                self.sar_downs.append(
                    nn.Conv2d(channels, next_channels, 3, stride=2, padding=1)
                )

        self.mid_blocks = nn.ModuleList(
            [ResidualConvBlock(widths[-1]) for _ in range(mid_blocks)]
        )
        self.mid_films = nn.ModuleList(
            [AlphaFiLM(alpha_dim, widths[-1]) for _ in range(mid_blocks)]
        )

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_films = nn.ModuleList()

        for i in range(len(widths) - 1, 0, -1):
            in_channels = widths[i]
            out_channels = widths[i - 1]
            self.up_convs.append(
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
            )
            self.dec_blocks.append(ResidualConvBlock(out_channels))
            self.dec_films.append(AlphaFiLM(alpha_dim, out_channels))

        self.head = nn.Conv2d(widths[0], in_opt, 3, padding=1)

    def forward(
        self,
        x_alpha: torch.Tensor,
        conditioning: torch.Tensor,
        sar: torch.Tensor,
    ) -> torch.Tensor:
        if x_alpha.ndim != 4 or sar.ndim != 4:
            raise ValueError("x_alpha and sar must have shape [B,C,H,W]")
        if x_alpha.shape[0] != sar.shape[0]:
            raise ValueError("x_alpha and sar batch sizes must match")
        if x_alpha.shape[-2:] != sar.shape[-2:]:
            raise ValueError("x_alpha and sar spatial sizes must match")

        emb = self.alpha_embedding(conditioning)
        optical = self.opt_stem(x_alpha)
        sar_feat = self.sar_stem(sar)
        skips: list[torch.Tensor] = []

        for i in range(len(self.widths)):
            optical = self.opt_blocks[i](optical)
            optical = self.enc_films[i](optical, emb)
            sar_feat = self.sar_blocks[i](sar_feat)
            optical = self.fusions[i](optical, sar_feat)
            skips.append(optical)

            if i < len(self.widths) - 1:
                optical = self.opt_downs[i](optical)
                sar_feat = self.sar_downs[i](sar_feat)

        for block, film in zip(self.mid_blocks, self.mid_films):
            optical = block(optical)
            optical = film(optical, emb)

        for i, (up, block, film) in enumerate(
            zip(self.up_convs, self.dec_blocks, self.dec_films)
        ):
            optical = F.interpolate(
                optical,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )
            optical = up(optical)
            skip = skips[-(i + 2)]
            if optical.shape[-2:] != skip.shape[-2:]:
                optical = F.interpolate(
                    optical,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            optical = optical + skip
            optical = block(optical)
            optical = film(optical, emb)

        return self.head(optical)


CANONICAL_BRIDGE_OPTICAL_ONLY_MODEL_IDENTITY = "canonical_bridge_optical_only_net"


class CanonicalBridgeOpticalOnlyNet(nn.Module):
    """True 13-channel optical-only control for CanonicalBridgeNet.

    The model retains CanonicalBridgeNet's optical encoder, alpha FiLM
    conditioning, bottleneck, decoder, skip connections, and direct-x0 head,
    but removes the complete SAR stem, SAR encoder, and gated fusion pathway.

    The optional sar argument is accepted only so the shared bridge
    training/evaluation step can call this model through the same interface.
    It is never read or used by the network.
    """

    def __init__(
        self,
        in_opt: int = 13,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        alpha_dim: int = 128,
        total_steps: int = 1000,
        mid_blocks: int = 2,
    ):
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must contain at least two stages")

        self.in_opt = int(in_opt)
        self.widths = tuple(int(v) for v in widths)
        self.total_steps = int(total_steps)

        self.alpha_embedding = PhysicalAlphaEmbedding(
            dim=alpha_dim,
            total_steps=total_steps,
        )
        self.opt_stem = nn.Conv2d(in_opt, widths[0], 3, padding=1)

        self.opt_blocks = nn.ModuleList()
        self.enc_films = nn.ModuleList()
        self.opt_downs = nn.ModuleList()

        for i, channels in enumerate(widths):
            self.opt_blocks.append(ResidualConvBlock(channels))
            self.enc_films.append(AlphaFiLM(alpha_dim, channels))
            if i < len(widths) - 1:
                self.opt_downs.append(
                    nn.Conv2d(
                        channels,
                        widths[i + 1],
                        3,
                        stride=2,
                        padding=1,
                    )
                )

        self.mid_blocks = nn.ModuleList(
            [ResidualConvBlock(widths[-1]) for _ in range(mid_blocks)]
        )
        self.mid_films = nn.ModuleList(
            [AlphaFiLM(alpha_dim, widths[-1]) for _ in range(mid_blocks)]
        )

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_films = nn.ModuleList()

        for i in range(len(widths) - 1, 0, -1):
            in_channels = widths[i]
            out_channels = widths[i - 1]
            self.up_convs.append(
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
            )
            self.dec_blocks.append(ResidualConvBlock(out_channels))
            self.dec_films.append(AlphaFiLM(alpha_dim, out_channels))

        self.head = nn.Conv2d(widths[0], in_opt, 3, padding=1)

    def forward(
        self,
        x_alpha: torch.Tensor,
        conditioning: torch.Tensor,
        sar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del sar
        if x_alpha.ndim != 4:
            raise ValueError("x_alpha must have shape [B,C,H,W]")
        if x_alpha.shape[1] != self.in_opt:
            raise ValueError(
                f"x_alpha must have {self.in_opt} optical channels, "
                f"got {x_alpha.shape[1]}"
            )

        emb = self.alpha_embedding(conditioning)
        optical = self.opt_stem(x_alpha)
        skips: list[torch.Tensor] = []

        for i in range(len(self.widths)):
            optical = self.opt_blocks[i](optical)
            optical = self.enc_films[i](optical, emb)
            skips.append(optical)

            if i < len(self.widths) - 1:
                optical = self.opt_downs[i](optical)

        for block, film in zip(self.mid_blocks, self.mid_films):
            optical = block(optical)
            optical = film(optical, emb)

        for i, (up, block, film) in enumerate(
            zip(self.up_convs, self.dec_blocks, self.dec_films)
        ):
            optical = F.interpolate(
                optical,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )
            optical = up(optical)
            skip = skips[-(i + 2)]
            if optical.shape[-2:] != skip.shape[-2:]:
                optical = F.interpolate(
                    optical,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            optical = optical + skip
            optical = block(optical)
            optical = film(optical, emb)

        return self.head(optical)


def count_canonical_bridge_optical_only_parameters(
    model: nn.Module | None = None,
) -> int:
    if model is None:
        model = CanonicalBridgeOpticalOnlyNet()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_canonical_bridge_parameters(model: nn.Module | None = None) -> int:
    if model is None:
        model = CanonicalBridgeNet()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
