"""Source-guided DB-CR backbone: published topology + patent-disclosed time NAFBlocks.

This is NOT claimed to be the authors' exact implementation. Remaining
unknowns (time-embedding widths and sharing, several operator choices)
are frozen explicitly in SourceGuidedDBCRConfig and written into the audit.

Unlike LegacyDBCRNet/AuditedDBCRNet, this variant time-conditions EACH
NAFBlock in the optical encoder, SAR encoder, bottleneck, and decoder.
The alpha input to those blocks is alpha(t)=sin(pi*t/(2*T)), matching
the original DB-CR bridge. Forward signature is compatible with the
existing training/validation runner: forward(x_t, t, sar) -> x0_hat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from .blocks import SFBlock, count_trainable_parameters
from .dbcr import (
    PUBLISHED_DECODER_BLOCKS,
    PUBLISHED_ENCODER_BLOCKS,
    PUBLISHED_PARAMETER_COUNT,
    PUBLISHED_SF_HEADS,
    PUBLISHED_WIDTHS,
)
from .patent_blocks import (
    PatentTimeEmbeddedNAFBlock,
    PatentTimeEmbedding,
    PatentTimeEmbeddingConfig,
    SCALE_SHIFT_MODES,
    sine_bridge_alpha,
)

SOURCE_GUIDED_DBCR_IDENTITY = "dbcr_source_guided_v1"
TIME_SHARING_MODES = ("per_block", "per_width")


@dataclass(frozen=True)
class SourceGuidedDBCRConfig:
    in_opt: int = 13
    in_sar: int = 2
    widths: tuple[int, ...] = PUBLISHED_WIDTHS
    encoder_blocks: tuple[int, ...] = PUBLISHED_ENCODER_BLOCKS
    decoder_blocks: tuple[int, ...] = PUBLISHED_DECODER_BLOCKS
    sf_heads: tuple[int, ...] = PUBLISHED_SF_HEADS
    total_steps: int = 1000
    time_embedding: PatentTimeEmbeddingConfig = field(
        default_factory=PatentTimeEmbeddingConfig
    )
    time_sharing: str = "per_block"
    scale_shift_mode: str = "patent"
    separate_modality_downsamplers: bool = True
    stem_kernel_size: int = 1
    output_kernel_size: int = 3

    def validate(self) -> None:
        depth = len(self.widths)
        if depth != 4:
            raise ValueError("DB-CR requires exactly four feature levels")
        if not (
            len(self.encoder_blocks)
            == len(self.decoder_blocks)
            == len(self.sf_heads)
            == depth
        ):
            raise ValueError("widths/encoder_blocks/decoder_blocks/sf_heads lengths differ")
        if any(v <= 0 for v in (*self.widths, *self.encoder_blocks, *self.decoder_blocks)):
            raise ValueError("widths and block counts must be positive")
        if any(h <= 0 or w % h != 0 for w, h in zip(self.widths, self.sf_heads)):
            raise ValueError("each width must be divisible by its SF head count")
        if self.in_opt <= 0 or self.in_sar <= 0:
            raise ValueError("input channels must be positive")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.time_sharing not in TIME_SHARING_MODES:
            raise ValueError("time_sharing must be per_block or per_width")
        if self.scale_shift_mode not in SCALE_SHIFT_MODES:
            raise ValueError(f"scale_shift_mode must be one of {SCALE_SHIFT_MODES}")
        if self.stem_kernel_size not in (1, 3) or self.output_kernel_size not in (1, 3):
            raise ValueError("stem/output kernel sizes must be 1 or 3")
        self.time_embedding.validate()

    def audit_config(self) -> dict:
        config = asdict(self)
        for key in ("widths", "encoder_blocks", "decoder_blocks", "sf_heads"):
            config[key] = list(config[key])
        return config


class SourceGuidedDBCRNet(nn.Module):
    """Four-level SAR/optical encoder with source-guided per-NAFBlock time FiLM.

    Unresolved design settings are NOT inferred from parameter count.
    The published encoder blocks and SF head counts are used unchanged.
    The mid NAFBlock is configured via decoder_blocks[0] following the
    audited scaffold, and remaining decoder blocks map from coarse to fine.
    """

    def __init__(self, config: SourceGuidedDBCRConfig | None = None):
        super().__init__()
        self.config = config or SourceGuidedDBCRConfig()
        self.config.validate()
        cfg = self.config
        widths = cfg.widths
        self.time_modules = (
            nn.ModuleDict({
                str(width): PatentTimeEmbedding(width, cfg.time_embedding)
                for width in sorted(set(widths))
            })
            if cfg.time_sharing == "per_width"
            else None
        )

        def blocks(width: int, count: int) -> nn.ModuleList:
            return nn.ModuleList([
                PatentTimeEmbeddedNAFBlock(
                    width,
                    time_config=cfg.time_embedding,
                    own_time_embedding=(cfg.time_sharing == "per_block"),
                    scale_shift_mode=cfg.scale_shift_mode,
                )
                for _ in range(count)
            ])

        self.opt_stem = nn.Conv2d(
            cfg.in_opt, widths[0], cfg.stem_kernel_size,
            padding=cfg.stem_kernel_size // 2,
        )
        self.sar_stem = nn.Conv2d(
            cfg.in_sar, widths[0], cfg.stem_kernel_size,
            padding=cfg.stem_kernel_size // 2,
        )
        self.opt_encoders = nn.ModuleList([
            blocks(width, count)
            for width, count in zip(widths, cfg.encoder_blocks)
        ])
        self.sar_encoders = nn.ModuleList([
            blocks(width, count)
            for width, count in zip(widths, cfg.encoder_blocks)
        ])
        self.fusion_blocks = nn.ModuleList([
            SFBlock(width, heads)
            for width, heads in zip(widths, cfg.sf_heads)
        ])

        if cfg.separate_modality_downsamplers:
            self.opt_downsamplers = nn.ModuleList([
                nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)
                for i in range(3)
            ])
            self.sar_downsamplers = nn.ModuleList([
                nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)
                for i in range(3)
            ])
            self.shared_downsamplers = None
        else:
            self.opt_downsamplers = None
            self.sar_downsamplers = None
            self.shared_downsamplers = nn.ModuleList([
                nn.Conv2d(widths[i], widths[i + 1], 2, stride=2)
                for i in range(3)
            ])

        self.bottleneck = blocks(widths[-1], cfg.decoder_blocks[0])
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(3, 0, -1):
            self.upsamplers.append(
                nn.ConvTranspose2d(widths[level], widths[level - 1], 2, stride=2)
            )
            self.decoders.append(
                blocks(widths[level - 1], cfg.decoder_blocks[4 - level])
            )
        self.output_head = nn.Conv2d(
            widths[0], cfg.in_opt, cfg.output_kernel_size,
            padding=cfg.output_kernel_size // 2,
        )

    def _apply_blocks(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        alpha: torch.Tensor,
        modulations: dict[str, tuple[torch.Tensor, ...]] | None,
    ) -> torch.Tensor:
        if not blocks:
            return x
        if modulations is None:
            for block in blocks:
                x = block(x, alpha=alpha)
        else:
            # Every block with the same feature width sees the same modulation;
            # this is an EXPLICIT alternative assumption, not paper-established.
            modulation = modulations[str(x.shape[1])]
            for block in blocks:
                x = block(x, modulation=modulation)
        return x

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, sar: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        if x_t.ndim != 4 or sar.ndim != 4:
            raise ValueError("x_t and sar must be NCHW tensors")
        if x_t.shape[1] != cfg.in_opt or sar.shape[1] != cfg.in_sar:
            raise ValueError("incorrect optical/SAR channel count")
        if x_t.shape[0] != sar.shape[0] or x_t.shape[-2:] != sar.shape[-2:]:
            raise ValueError("optical and SAR batch/spatial shapes differ")
        if any(d % 8 != 0 for d in x_t.shape[-2:]):
            raise ValueError("image height/width must be divisible by 8")
        if not torch.is_tensor(t):
            raise TypeError("t must be a torch tensor")
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError("t must be scalar or have shape [B]")
        t = t.to(device=x_t.device, dtype=torch.float32)
        if not torch.isfinite(t).all():
            raise ValueError("t must be finite")
        if bool(((t < 0) | (t > cfg.total_steps)).any()):
            raise ValueError("t must be in [0,total_steps]")

        alpha = sine_bridge_alpha(t, cfg.total_steps)
        modulations = (
            {
                key: module.from_alpha(alpha)
                for key, module in self.time_modules.items()
            }
            if self.time_modules is not None
            else None
        )

        optical = self.opt_stem(x_t)
        sar_features = self.sar_stem(sar)
        skips = []
        for level, (opt_blocks, sar_blocks, fusion) in enumerate(zip(
            self.opt_encoders, self.sar_encoders, self.fusion_blocks
        )):
            optical = self._apply_blocks(optical, opt_blocks, alpha, modulations)
            sar_features = self._apply_blocks(sar_features, sar_blocks, alpha, modulations)
            optical = fusion(optical, sar_features)
            skips.append(optical)
            if level < 3:
                if cfg.separate_modality_downsamplers:
                    optical = self.opt_downsamplers[level](optical)
                    sar_features = self.sar_downsamplers[level](sar_features)
                else:
                    optical = self.shared_downsamplers[level](optical)
                    sar_features = self.shared_downsamplers[level](sar_features)

        optical = self._apply_blocks(optical, self.bottleneck, alpha, modulations)
        for index, (up, decoder) in enumerate(zip(self.upsamplers, self.decoders)):
            optical = up(optical)
            skip = skips[-(index + 2)]
            if optical.shape != skip.shape:
                raise RuntimeError("decoder and optical skip shapes differ")
            optical = self._apply_blocks(optical + skip, decoder, alpha, modulations)
        return self.output_head(optical)

    def architecture_audit(self) -> dict:
        count = count_trainable_parameters(self)
        return {
            "model_identity": SOURCE_GUIDED_DBCR_IDENTITY,
            "status": "source_guided_not_official_paper_exact",
            "trainable_parameters": count,
            "published_dbcr_parameters": PUBLISHED_PARAMETER_COUNT,
            "parameter_gap_vs_published": PUBLISHED_PARAMETER_COUNT - count,
            "num_time_conditioned_nafblocks": sum(
                sum(len(stage) for stage in branch)
                for branch in (self.opt_encoders, self.sar_encoders)
            ) + len(self.bottleneck) + sum(map(len, self.decoders)),
            "time_conditioning": "sinusoidal(alpha_t)->3 linear + 2 SimpleGate->4 scale_shift",
            "scale_shift_equation": (
                "x*scale+shift" if self.config.scale_shift_mode == "patent"
                else "x*(1+scale)+shift"
            ),
            "time_conditioning_branches": "optical_encoder+sar_encoder+optical_decoder+bottleneck",
            "normalization": "patent_whole_feature_map_layernorm_in_NAF;paper_layernorm2d_in_SF",
            "sfblock": "paper_audited_cross_modal_channel_attention",
            "architecture_config": self.config.audit_config(),
            "unresolved_assumptions": [
                "time positional embedding width",
                "time MLP hidden width",
                "per-block versus per-width time MLP ownership",
                "SFBlock normalization axes and exact attention-head arrangement",
                "normalization affine parameterization",
                "exact downsample/upsample operators and output convolution",
            ],
        }
