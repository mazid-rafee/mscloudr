import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mscloudr.bridge import (
    deterministic_residual_coordinate_reverse_step,
)
from mscloudr.evaluation import (
    evaluate_nfe1_endpoint,
    evaluate_sar_residual_coordinate_reverse,
)


class _ResidualCoordinateDataset(Dataset):
    def __init__(self, n=4):
        self.cloudy = torch.full((n, 13, 8, 8), 0.6)
        self.target = torch.full((n, 13, 8, 8), 0.4)
        self.sar = torch.full((n, 2, 8, 8), 0.5)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        return {
            "cloudy": self.cloudy[index],
            "target": self.target[index],
            "sar": self.sar[index],
        }


class _ResidualCoordinateToy(nn.Module):
    def bridge_effective_alpha(self, cloudy, sar, alpha):
        del sar
        alpha = alpha.to(cloudy).view(-1, 1, 1, 1)
        gate = torch.full(
            (cloudy.shape[0], 1, cloudy.shape[2], cloudy.shape[3]),
            0.5,
            device=cloudy.device,
            dtype=cloudy.dtype,
        )
        return alpha + 0.10 * 4.0 * alpha * (1.0 - alpha) * gate

    def forward(self, x_t, conditioning, sar):
        del sar
        alpha = conditioning.float().view(-1, 1, 1, 1) / 1000.0
        return x_t - 0.1 * alpha


def test_residual_coordinate_reverse_tracks_exact_path_with_perfect_x0():
    x0 = torch.rand(2, 13, 4, 4)
    cloudy = torch.rand_like(x0)

    lambda_t = torch.rand(2, 1, 4, 4) * 0.4 + 0.5
    lambda_s = lambda_t * 0.4

    x_t = (1.0 - lambda_t) * x0 + lambda_t * cloudy
    expected = (1.0 - lambda_s) * x0 + lambda_s * cloudy

    actual = deterministic_residual_coordinate_reverse_step(
        x_t,
        x0,
        lambda_t=lambda_t,
        lambda_s=lambda_s,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)


def test_residual_coordinate_reverse_nfe1_matches_endpoint_direct():
    loader = DataLoader(
        _ResidualCoordinateDataset(),
        batch_size=2,
        shuffle=False,
    )

    direct = evaluate_nfe1_endpoint(
        _ResidualCoordinateToy(),
        loader,
        total_steps=1000,
        device="cpu",
    )
    matched = evaluate_sar_residual_coordinate_reverse(
        _ResidualCoordinateToy(),
        loader,
        total_steps=1000,
        nfe=1,
        device="cpu",
    )

    assert abs(direct.metrics["L1"] - matched.metrics["L1"]) < 1e-8
    assert abs(direct.metrics["PSNR"] - matched.metrics["PSNR"]) < 1e-8


def test_residual_coordinate_reverse_nfe2_uses_intermediate_coordinate():
    loader = DataLoader(
        _ResidualCoordinateDataset(),
        batch_size=2,
        shuffle=False,
    )

    nfe1 = evaluate_sar_residual_coordinate_reverse(
        _ResidualCoordinateToy(),
        loader,
        total_steps=1000,
        nfe=1,
        device="cpu",
    )
    nfe2 = evaluate_sar_residual_coordinate_reverse(
        _ResidualCoordinateToy(),
        loader,
        total_steps=1000,
        nfe=2,
        device="cpu",
    )

    assert abs(nfe1.metrics["L1"] - nfe2.metrics["L1"]) > 1e-4
