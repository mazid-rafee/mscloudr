import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mscloudr.bridge import make_sar_curved_bridge_state
from mscloudr.evaluation import (
    evaluate_nfe1_endpoint,
    evaluate_sar_curved_reverse,
)


class _CurvedDataset(Dataset):
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
            "sample_id": f"sample-{index}",
        }


class _CurvedToy(nn.Module):
    def bridge_curvature(self, cloudy, sar):
        del sar
        return torch.full_like(cloudy, 0.02)

    def forward(self, x_t, conditioning, sar):
        del sar
        alpha = conditioning.float().view(-1, 1, 1, 1) / 1000.0
        return x_t - 0.1 * alpha


def test_curved_projection_preserves_endpoints_and_uses_midpoint_curvature():
    x0_hat = torch.full((2, 13, 4, 4), 0.2)
    cloudy = torch.full_like(x0_hat, 0.8)
    curvature = torch.full_like(x0_hat, 0.05)

    endpoints = make_sar_curved_bridge_state(
        x0_hat,
        cloudy,
        curvature,
        torch.tensor([0.0, 1.0]),
    )
    midpoint = make_sar_curved_bridge_state(
        x0_hat[:1],
        cloudy[:1],
        curvature[:1],
        torch.tensor([0.5]),
    )

    assert torch.equal(endpoints[0], x0_hat[0])
    assert torch.equal(endpoints[1], cloudy[1])
    expected_midpoint = 0.5 * x0_hat[:1] + 0.5 * cloudy[:1] + curvature[:1]
    assert torch.allclose(midpoint, expected_midpoint)


def test_curved_reverse_nfe1_matches_endpoint_direct_prediction():
    loader = DataLoader(_CurvedDataset(), batch_size=2, shuffle=False)
    model_a = _CurvedToy()
    model_b = _CurvedToy()

    direct = evaluate_nfe1_endpoint(
        model_a,
        loader,
        total_steps=1000,
        device="cpu",
    )
    curved = evaluate_sar_curved_reverse(
        model_b,
        loader,
        total_steps=1000,
        nfe=1,
        device="cpu",
    )

    assert abs(direct.metrics["L1"] - curved.metrics["L1"]) < 1e-8
    assert abs(direct.metrics["PSNR"] - curved.metrics["PSNR"]) < 1e-8


def test_curved_reverse_nfe2_uses_nonzero_intermediate_curvature():
    loader = DataLoader(_CurvedDataset(), batch_size=2, shuffle=False)

    nfe1 = evaluate_sar_curved_reverse(
        _CurvedToy(),
        loader,
        total_steps=1000,
        nfe=1,
        device="cpu",
    )
    nfe2 = evaluate_sar_curved_reverse(
        _CurvedToy(),
        loader,
        total_steps=1000,
        nfe=2,
        device="cpu",
    )

    assert abs(nfe1.metrics["L1"] - nfe2.metrics["L1"]) > 1e-4
