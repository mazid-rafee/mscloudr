"""Regression tests for paired CanonicalBridge fixed-alpha diagnostics."""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mscloudr.cli.eval_canonical_bridge_alpha_grid import (
    _validate_alphas,
    build_parser,
    evaluate_alpha_grid,
)
from mscloudr.evaluation import evaluate_nfe1_endpoint


class _TinyDataset(Dataset):
    def __init__(self, n=3):
        generator = torch.Generator().manual_seed(123)
        self.target = 0.2 + 0.4 * torch.rand(
            n, 13, 8, 8, generator=generator
        )
        self.cloudy = self.target + 0.1
        self.sar = torch.zeros(n, 2, 8, 8)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return {
            "target": self.target[index],
            "cloudy": self.cloudy[index],
            "sar": self.sar[index],
            "sample_id": f"sample-{index}",
        }


class _RecordingIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.conditioning = []

    def forward(self, x_alpha, conditioning, sar):
        self.inputs.append(x_alpha.detach().cpu().clone())
        self.conditioning.append(conditioning.detach().cpu().clone())
        return x_alpha


def test_alpha_one_matches_existing_endpoint_metric_implementation():
    loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)
    model_a = _RecordingIdentity()
    model_b = _RecordingIdentity()

    grid = evaluate_alpha_grid(
        {"a": (model_a, 1000), "b": (model_b, 1000)},
        loader,
        alphas=(1.0,),
        device="cpu",
    )
    endpoint = evaluate_nfe1_endpoint(
        _RecordingIdentity(), loader, total_steps=1000, device="cpu"
    )

    assert grid["num_samples"] == endpoint.num_samples == 3
    assert grid["num_batches"] == endpoint.num_batches == 2
    row = grid["rows"][0]
    for key, expected in endpoint.metrics.items():
        assert row["metrics_a"][key] == pytest.approx(expected, abs=1e-6)
        assert row["metrics_b"][key] == pytest.approx(expected, abs=1e-6)
        assert row["delta_b_minus_a"][key] == pytest.approx(0.0, abs=1e-6)
        assert row["metric_counts_a"][key] == endpoint.metric_counts[key]


def test_intermediate_state_is_shared_and_uses_inverse_mr_conditioning():
    dataset = _TinyDataset(n=2)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model_a = _RecordingIdentity()
    model_b = _RecordingIdentity()

    grid = evaluate_alpha_grid(
        {"a": (model_a, 1000), "b": (model_b, 1000)},
        loader,
        alphas=(0.5, 1.0),
        device="cpu",
    )

    intermediate = 0.5 * dataset.target + 0.5 * dataset.cloudy
    torch.testing.assert_close(model_a.inputs[0], intermediate)
    torch.testing.assert_close(model_b.inputs[0], intermediate)
    torch.testing.assert_close(model_a.inputs[1], dataset.cloudy)
    torch.testing.assert_close(model_b.inputs[1], dataset.cloudy)

    expected_coord = -1000.0 / 3.0 * math.log1p(
        -(1.0 - math.exp(-3.0)) * 0.5
    )
    torch.testing.assert_close(
        model_a.conditioning[0],
        torch.full((2,), expected_coord),
        rtol=1e-6,
        atol=1e-5,
    )
    torch.testing.assert_close(
        model_b.conditioning[1],
        torch.full((2,), 1000.0),
    )
    assert grid["rows"][0]["metrics_a"]["L1"] == pytest.approx(0.05, abs=1e-6)
    assert grid["rows"][1]["metrics_a"]["L1"] == pytest.approx(0.10, abs=1e-6)


def test_pilot_counts_are_sample_weighted():
    loader = DataLoader(_TinyDataset(n=3), batch_size=2, shuffle=False)
    grid = evaluate_alpha_grid(
        {"a": (_RecordingIdentity(), 1000), "b": (_RecordingIdentity(), 1000)},
        loader,
        alphas=(0.25,),
        device="cpu",
        max_batches=1,
    )
    assert grid["num_batches"] == 1
    assert grid["num_samples"] == 2
    assert grid["rows"][0]["metric_counts_a"]["L1"] == 2


@pytest.mark.parametrize(
    "bad",
    [(1.1,), (-0.1,), (float("nan"),), (0.5, 0.5), ()],
)
def test_bad_alpha_grid_rejected(bad):
    with pytest.raises(ValueError):
        _validate_alphas(bad)


def test_cli_explicitly_forbids_test_split():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint-a", "a.pt",
                "--checkpoint-b", "b.pt",
                "--output", "grid.json",
                "--split", "test",
            ]
        )
