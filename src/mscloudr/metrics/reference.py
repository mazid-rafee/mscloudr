"""Reference-compatible SEN12MS-CR reconstruction metrics.

The benchmark formulas follow the public UnCRtainTS implementation at
PatrickTUM/UnCRtainTS commit 5e1f1b58e993645e765b64e10b6e9c7ff828b36f.

Important: dataset-level reporting is the arithmetic mean of *per-image*
metrics.  It is not a metric computed over a whole batch and then averaged over
batches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

REFERENCE_REPO = "PatrickTUM/UnCRtainTS"
REFERENCE_COMMIT = "5e1f1b58e993645e765b64e10b6e9c7ff828b36f"


def _validate_pair(target: torch.Tensor, pred: torch.Tensor) -> None:
    if target.shape != pred.shape:
        raise ValueError(
            f"target/pred shape mismatch: {tuple(target.shape)} vs {tuple(pred.shape)}"
        )
    if target.ndim != 4:
        raise ValueError(
            "expected tensors shaped [N, C, H, W]; "
            f"received {tuple(target.shape)}"
        )


def _gaussian_window(
    window_size: int,
    channels: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    sigma: float = 1.5,
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device)
    center = window_size // 2
    weights = torch.exp(-((coords - center) ** 2) / (2.0 * sigma**2))
    weights = weights / weights.sum()
    window_2d = torch.outer(weights, weights)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def reference_ssim(
    target: torch.Tensor,
    pred: torch.Tensor,
    *,
    window_size: int = 11,
) -> torch.Tensor:
    """SSIM matching UnCRtainTS' bundled pytorch_ssim implementation.

    Inputs are expected in [0, 1], with shape [N, C, H, W].
    """

    _validate_pair(target, pred)
    channels = target.shape[1]
    window = _gaussian_window(
        window_size,
        channels,
        dtype=target.dtype,
        device=target.device,
    )
    padding = window_size // 2

    mu_target = F.conv2d(
        target, window, padding=padding, groups=channels
    )
    mu_pred = F.conv2d(
        pred, window, padding=padding, groups=channels
    )

    mu_target_sq = mu_target.square()
    mu_pred_sq = mu_pred.square()
    mu_product = mu_target * mu_pred

    sigma_target_sq = (
        F.conv2d(target * target, window, padding=padding, groups=channels)
        - mu_target_sq
    )
    sigma_pred_sq = (
        F.conv2d(pred * pred, window, padding=padding, groups=channels)
        - mu_pred_sq
    )
    sigma_cross = (
        F.conv2d(target * pred, window, padding=padding, groups=channels)
        - mu_product
    )

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = (
        (2.0 * mu_product + c1) * (2.0 * sigma_cross + c2)
    ) / (
        (mu_target_sq + mu_pred_sq + c1)
        * (sigma_target_sq + sigma_pred_sq + c2)
    )
    return ssim_map.mean()


def reference_image_metrics(
    target: torch.Tensor,
    pred: torch.Tensor,
) -> dict[str, float]:
    """Compute one image's RMSE, MAE, PSNR, SAM, and SSIM.

    This intentionally mirrors UnCRtainTS' metric definitions.  The caller
    should pass one image at a time, normally shaped [1, 13, H, W].
    """

    _validate_pair(target, pred)
    if target.shape[0] != 1:
        raise ValueError(
            "reference_image_metrics expects exactly one image; "
            "use reference_batch_metrics for a batch"
        )

    rmse = torch.sqrt(torch.mean((target - pred).square()))
    psnr = 20.0 * torch.log10(1.0 / rmse)
    mae = torch.mean(torch.abs(target - pred))

    spectral_dot = torch.sum(target * pred, dim=1)
    target_norm = torch.sqrt(torch.sum(target * target, dim=1))
    pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1))
    cosine = spectral_dot / target_norm
    cosine = cosine / pred_norm
    sam = torch.mean(
        torch.acos(torch.clamp(cosine, -1.0, 1.0))
        * (180.0 / math.pi)
    )

    ssim = reference_ssim(target, pred)

    return {
        "RMSE": float(rmse.detach().cpu()),
        "MAE": float(mae.detach().cpu()),
        "PSNR": float(psnr.detach().cpu()),
        "SAM": float(sam.detach().cpu()),
        "SSIM": float(ssim.detach().cpu()),
    }


def reference_batch_metrics(
    target: torch.Tensor,
    pred: torch.Tensor,
) -> list[dict[str, float]]:
    """Return one independent metric dictionary per image in a batch."""

    _validate_pair(target, pred)
    return [
        reference_image_metrics(target[i : i + 1], pred[i : i + 1])
        for i in range(target.shape[0])
    ]


@dataclass
class ReferenceMetricAccumulator:
    """Arithmetic sample mean matching UnCRtainTS' avg_img_metrics behavior.

    Non-finite scalar metric values are skipped independently, as in the
    reference meter.  This matters mainly for degenerate SAM/PSNR cases.
    """

    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            value = float(value)
            if not math.isfinite(value):
                continue
            self.sums[name] = self.sums.get(name, 0.0) + value
            self.counts[name] = self.counts.get(name, 0) + 1

    def update_batch(self, target: torch.Tensor, pred: torch.Tensor) -> None:
        for metrics in reference_batch_metrics(target, pred):
            self.update(metrics)

    def compute(self) -> dict[str, float]:
        return {
            name: self.sums[name] / self.counts[name]
            for name in sorted(self.sums)
            if self.counts.get(name, 0) > 0
        }

    def metric_counts(self) -> dict[str, int]:
        return dict(self.counts)
