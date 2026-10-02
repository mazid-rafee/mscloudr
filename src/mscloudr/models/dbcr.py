"""Audited DB-CR network scaffold.

This module assembles the paper-described dual-branch topology using the
previously audited NAFBlock and SFBlock implementations.

DB-CR calls its blocks "time-embedded NAFBlocks" but the paper does not specify
the exact injection operator or placement. The historical implementation's
stage-level additive optical time bias is therefore exposed explicitly as a
legacy reproducibility choice rather than presented as paper-specified behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import NAFBlock, SFBlock, count_trainable_parameters

PUBLISHED_WIDTHS = (22, 44, 88, 176)
PUBLISHED_ENCODER_BLOCKS = (1, 1, 1, 28)
PUBLISHED_DECODER_BLOCKS = (1, 1, 1, 1)
PUBLISHED_SF_HEADS = (1, 1, 2, 4)
PUBLISHED_PARAMETER_COUNT = 18_060_000
HISTORICAL_MSCLOUDR_PARAMETER_COUNT = 13_588_641


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
        half = self.dim // 2
        if half == 0:
            return t[:, None]
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / float(half)
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class LegacyStageTimeConditioner(nn.Module):
    """Historical stage-level optical time bias; not claimed as paper-exact."""

    def __init__(self, widths: tuple[int, ...], time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = int(time_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.GELU(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )
        self.to_channels = nn.ModuleList(
            [nn.Linear(self.time_dim, width) for width in widths]
        )

    def forward(self, t: torch.Tensor) -> list[torch.Tensor]:
        embedding = self.time_mlp(t.float())
        return [
            projection(embedding).unsqueeze(-1).unsqueeze(-1)
            for projection in self.to_channels
        ]


@dataclass(frozen=True)
class DBCRArchitectureConfig:
    in_opt: int = 13
    in_sar: int = 2
    widths: tuple[int, ...] = PUBLISHED_WIDTHS
    encoder_blocks: tuple[int, ...] = PUBLISHED_ENCODER_BLOCKS
    decoder_blocks: tuple[int, ...] = PUBLISHED_DECODER_BLOCKS
    sf_heads: tuple[int, ...] = PUBLISHED_SF_HEADS
    time_dim: int = 128
    stem_kernel_size: int = 1
    output_kernel_size: int = 3
    separate_modality_downsamplers: bool = True
    time_conditioning: str = "legacy_stage_bias"

    def validate(self) -> None:
        depth = len(self.widths)
        if depth < 2:
            raise ValueError("at least two feature levels are required")
        if len(self.encoder_blocks) != depth:
            raise ValueError("encoder_blocks must match widths length")
        if len(self.decoder_blocks) != depth:
            raise ValueError("decoder_blocks must match widths length")
        if len(self.sf_heads) != depth:
            raise ValueError("sf_heads must match widths length")
        if any(width <= 0 for width in self.widths):
            raise ValueError("all widths must be positive")
        if any(blocks <= 0 for blocks in self.encoder_blocks):
            raise ValueError("encoder block counts must be positive")
        if any(blocks <= 0 for blocks in self.decoder_blocks):
            raise ValueError("decoder block counts must be positive")
        for width, heads in zip(self.widths, self.sf_heads):
            if width % heads != 0:
                raise ValueError(f"width {width} is not divisible by heads {heads}")
        if self.stem_kernel_size not in {1, 3}:
            raise ValueError("stem_kernel_size must be 1 or 3")
        if self.output_kernel_size not in {1, 3}:
            raise ValueError("output_kernel_size must be 1 or 3")
        if self.time_conditioning not in {"legacy_stage_bias", "none"}:
            raise ValueError("time_conditioning must be 'legacy_stage_bias' or 'none'")


class AuditedDBCRNet(nn.Module):
    """Full DB-CR topology with unresolved implementation choices explicit."""

    def __init__(self, config: DBCRArchitectureConfig | None = None) -> None:
        super().__init__()
        self.config = config or DBCRArchitectureConfig()
        self.config.validate()
        cfg = self.config
        widths = cfg.widths
        depth = len(widths)

        stem_padding = cfg.stem_kernel_size // 2
        self.opt_stem = nn.Conv2d(
            cfg.in_opt, widths[0], cfg.stem_kernel_size, padding=stem_padding
        )
        self.sar_stem = nn.Conv2d(
            cfg.in_sar, widths[0], cfg.stem_kernel_size, padding=stem_padding
        )

        self.opt_encoders = nn.ModuleList([
            nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
            for width, num_blocks in zip(widths, cfg.encoder_blocks)
        ])
        self.sar_encoders = nn.ModuleList([
            nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
            for width, num_blocks in zip(widths, cfg.encoder_blocks)
        ])
        self.fusion_blocks = nn.ModuleList([
            SFBlock(width, heads) for width, heads in zip(widths, cfg.sf_heads)
        ])

        if cfg.separate_modality_downsamplers:
            self.opt_downsamplers = nn.ModuleList([
                nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)
                for i in range(depth - 1)
            ])
            self.sar_downsamplers = nn.ModuleList([
                nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)
                for i in range(depth - 1)
            ])
            self.shared_downsamplers = None
        else:
            self.opt_downsamplers = None
            self.sar_downsamplers = None
            self.shared_downsamplers = nn.ModuleList([
                nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)
                for i in range(depth - 1)
            ])

        self.bottleneck = nn.Sequential(
            *[NAFBlock(widths[-1]) for _ in range(cfg.decoder_blocks[0])]
        )
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(depth - 1, 0, -1):
            self.upsamplers.append(
                nn.ConvTranspose2d(widths[level], widths[level - 1], 2, stride=2)
            )
            decoder_index = depth - level
            self.decoders.append(nn.Sequential(*[
                NAFBlock(widths[level - 1])
                for _ in range(cfg.decoder_blocks[decoder_index])
            ]))

        out_padding = cfg.output_kernel_size // 2
        self.output_head = nn.Conv2d(
            widths[0], cfg.in_opt, cfg.output_kernel_size, padding=out_padding
        )

        self.time_conditioner = (
            LegacyStageTimeConditioner(widths, cfg.time_dim)
            if cfg.time_conditioning == "legacy_stage_bias"
            else None
        )

    def _downsample_pair(self, level, optical, sar):
        if self.config.separate_modality_downsamplers:
            assert self.opt_downsamplers is not None
            assert self.sar_downsamplers is not None
            return (
                self.opt_downsamplers[level](optical),
                self.sar_downsamplers[level](sar),
            )
        assert self.shared_downsamplers is not None
        down = self.shared_downsamplers[level]
        return down(optical), down(sar)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4 or sar.ndim != 4:
            raise ValueError("x_t and sar must both have shape [B,C,H,W]")
        if x_t.shape[0] != sar.shape[0]:
            raise ValueError("x_t and sar batch sizes must match")
        if x_t.shape[-2:] != sar.shape[-2:]:
            raise ValueError("x_t and sar spatial dimensions must match")
        if x_t.shape[1] != self.config.in_opt:
            raise ValueError(f"expected {self.config.in_opt} optical channels")
        if sar.shape[1] != self.config.in_sar:
            raise ValueError(f"expected {self.config.in_sar} SAR channels")
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError(f"t must be scalar or shape [B={x_t.shape[0]}]")

        factor = 2 ** (len(self.config.widths) - 1)
        h, w = x_t.shape[-2:]
        if h % factor != 0 or w % factor != 0:
            raise ValueError(f"spatial dimensions must be divisible by {factor}")

        optical = self.opt_stem(x_t)
        sar_features = self.sar_stem(sar)
        time_biases = (
            self.time_conditioner(t.to(device=x_t.device, dtype=x_t.dtype))
            if self.time_conditioner is not None
            else [None] * len(self.config.widths)
        )

        skips = []
        for level, (opt_encoder, sar_encoder, fusion) in enumerate(
            zip(self.opt_encoders, self.sar_encoders, self.fusion_blocks)
        ):
            bias = time_biases[level]
            if bias is not None:
                optical = optical + bias
            optical = opt_encoder(optical)
            sar_features = sar_encoder(sar_features)
            optical = fusion(optical, sar_features)
            skips.append(optical)
            if level < len(self.config.widths) - 1:
                optical, sar_features = self._downsample_pair(
                    level, optical, sar_features
                )

        optical = self.bottleneck(optical)
        for decoder_index, (up, decoder) in enumerate(
            zip(self.upsamplers, self.decoders)
        ):
            optical = up(optical)
            skip = skips[-(decoder_index + 2)]
            if optical.shape != skip.shape:
                raise RuntimeError("decoder/skip shape mismatch")
            optical = decoder(optical + skip)

        return self.output_head(optical)

    def architecture_audit(self) -> dict[str, object]:
        parameters = count_trainable_parameters(self)
        return {
            "trainable_parameters": parameters,
            "published_dbcr_parameters": PUBLISHED_PARAMETER_COUNT,
            "parameter_gap_vs_published": PUBLISHED_PARAMETER_COUNT - parameters,
            "historical_mscloudr_parameters": HISTORICAL_MSCLOUDR_PARAMETER_COUNT,
            "time_conditioning": self.config.time_conditioning,
            "separate_modality_downsamplers": self.config.separate_modality_downsamplers,
            "stem_kernel_size": self.config.stem_kernel_size,
            "output_kernel_size": self.config.output_kernel_size,
            "widths": list(self.config.widths),
            "encoder_blocks": list(self.config.encoder_blocks),
            "decoder_blocks": list(self.config.decoder_blocks),
            "sf_heads": list(self.config.sf_heads),
        }
