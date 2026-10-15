"""Attention primitives for controlled CanonicalBridge backbone experiments.

Swin variant: regular/shifted-window spatial attention.
HDiT-inspired variant: global spatial attention only at low resolution.

These are newly initialized restoration blocks, not pretrained Swin/HDiT
models and not a reproduction of EMRDM's complete diffusion architecture.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


ATTENTION_BACKBONES = ("cnn", "swin", "hdit_global")
SWIN_WINDOW_SIZE = 8
DEEP_STAGE_ATTENTION_DEPTH = 2


def attention_heads(channels: int) -> int:
    """Keep head width >=16 when possible, with at most eight heads."""
    for heads in (8, 4, 2, 1):
        if channels % heads == 0 and channels // heads >= 16:
            return heads
    return 1


def _windows(x: torch.Tensor, window: int) -> torch.Tensor:
    """Partition an NHWC tensor with divisible spatial dimensions."""
    batch, height, width, channels = x.shape
    x = x.reshape(
        batch, height // window, window, width // window, window, channels
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(
        -1, window * window, channels
    )


def _unwindows(
    tokens: torch.Tensor,
    *,
    batch: int,
    height: int,
    width: int,
    window: int,
) -> torch.Tensor:
    channels = tokens.shape[-1]
    x = tokens.reshape(
        batch, height // window, width // window, window, window, channels
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(
        batch, height, width, channels
    )


def _shift_attention_mask(
    height: int,
    width: int,
    window: int,
    shift: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Mask region wraparound introduced by cyclic shifts (Swin protocol)."""
    image_mask = torch.zeros((1, height, width, 1), device=device)
    h_slices = (
        slice(0, -window),
        slice(-window, -shift),
        slice(-shift, None),
    )
    w_slices = (
        slice(0, -window),
        slice(-window, -shift),
        slice(-shift, None),
    )
    region = 0
    for hs in h_slices:
        for ws in w_slices:
            image_mask[:, hs, ws, :] = region
            region += 1
    labels = _windows(image_mask, window).squeeze(-1)
    return labels.unsqueeze(1) != labels.unsqueeze(2)


class WindowAttentionBlock(nn.Module):
    """Pre-norm windowed self-attention with optional shifted windows."""

    def __init__(
        self,
        channels: int,
        *,
        window_size: int = SWIN_WINDOW_SIZE,
        shift_size: int = 0,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        if window_size <= 0 or not 0 <= shift_size < window_size:
            raise ValueError("invalid window or shift size")
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.heads = attention_heads(channels)
        self.head_dim = channels // self.heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * int(mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        window = min(self.window_size, height, width)
        shift = self.shift_size if min(height, width) > self.window_size else 0
        # The default 256x256 SEN12MS-CR tiles are divisible by the
        # window size at both attention encoder stages.
        pad_h = (-height) % window
        pad_w = (-width) % window
        residual = x
        y = self.norm1(x.permute(0, 2, 3, 1))
        y = y.permute(0, 3, 1, 2)
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h))
        padded_h, padded_w = height + pad_h, width + pad_w
        y = y.permute(0, 2, 3, 1)

        if shift:
            y = torch.roll(y, shifts=(-shift, -shift), dims=(1, 2))

        tokens = _windows(y, window)
        num_windows = (padded_h // window) * (padded_w // window)
        qkv = self.qkv(tokens).reshape(
            -1, window * window, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if shift:
            mask = _shift_attention_mask(
                padded_h, padded_w, window, shift, device=x.device
            )
            scores = scores.reshape(
                batch, num_windows, self.heads,
                window * window, window * window,
            )
            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(2), -100.0
            )
            scores = scores.reshape(
                -1, self.heads, window * window, window * window
            )

        attention = scores.softmax(dim=-1)
        attended = (attention @ v).transpose(1, 2).reshape(
            -1, window * window, channels
        )
        out = _unwindows(
            self.proj(attended),
            batch=batch,
            height=padded_h,
            width=padded_w,
            window=window,
        )
        if shift:
            out = torch.roll(out, shifts=(shift, shift), dims=(1, 2))
        out = out[:, :height, :width, :].permute(0, 3, 1, 2)
        x = residual + out
        y = x.permute(0, 2, 3, 1)
        y = self.mlp(self.norm2(y)).permute(0, 3, 1, 2)
        return x + y


class SwinStage(nn.Module):
    """Two successive regular/shifted-window attention blocks."""

    def __init__(self, channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            WindowAttentionBlock(channels, shift_size=0),
            WindowAttentionBlock(
                channels, shift_size=SWIN_WINDOW_SIZE // 2
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class GlobalAttentionBlock(nn.Module):
    """HDiT-inspired pre-norm global spatial self-attention.

    Uses PyTorch scaled dot product attention, allowing memory-efficient
    attention kernels when supported by the accelerator.
    """

    def __init__(self, channels: int, *, mlp_ratio: int = 2):
        super().__init__()
        self.heads = attention_heads(channels)
        self.head_dim = channels // self.heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * int(mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels
        )
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch, height * width, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        out = self.proj(
            out.transpose(1, 2).reshape(batch, height * width, channels)
        )
        tokens = tokens + out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.reshape(
            batch, height, width, channels
        ).permute(0, 3, 1, 2).contiguous()
