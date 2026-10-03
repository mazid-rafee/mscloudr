import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mscloudr.evaluation import evaluate_nfe1_endpoint


class _DictDataset(Dataset):
    def __init__(self, n=5):
        generator = torch.Generator().manual_seed(123)
        self.target = 0.2 + 0.3 * torch.rand(
            n,
            13,
            8,
            8,
            generator=generator,
        )
        self.cloudy = self.target + 0.1
        self.sar = torch.zeros(
            n,
            2,
            8,
            8,
        )

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        return {
            "target": self.target[index],
            "cloudy": self.cloudy[index],
            "sar": self.sar[index],
            "sample_id": f"sample-{index}",
        }


class _EndpointIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_timesteps = []

    def forward(self, x_t, t, sar):
        del sar
        self.seen_timesteps.append(
            t.detach().cpu().clone()
        )
        return x_t


def test_endpoint_evaluation_uses_cloudy_at_t_and_reports_l1_alias():
    model = _EndpointIdentity()
    loader = DataLoader(
        _DictDataset(n=5),
        batch_size=2,
        shuffle=False,
    )

    result = evaluate_nfe1_endpoint(
        model,
        loader,
        total_steps=1000,
        device="cpu",
    )

    assert result.num_samples == 5
    assert result.num_batches == 3
    assert abs(result.metrics["L1"] - 0.1) < 1e-6
    assert abs(result.metrics["MAE"] - 0.1) < 1e-6
    assert abs(result.metrics["RMSE"] - 0.1) < 1e-6
    assert abs(result.metrics["PSNR"] - 20.0) < 1e-4
    assert result.metric_counts["L1"] == 5
    assert result.metric_counts["MAE"] == 5

    timesteps = torch.cat(
        model.seen_timesteps
    )
    assert torch.equal(
        timesteps,
        torch.full((5,), 1000),
    )


def test_endpoint_evaluation_progress_and_max_batches():
    model = _EndpointIdentity()
    loader = DataLoader(
        _DictDataset(n=6),
        batch_size=2,
        shuffle=False,
    )
    events = []

    result = evaluate_nfe1_endpoint(
        model,
        loader,
        total_steps=1000,
        device="cpu",
        max_batches=2,
        progress_every=1,
        progress_callback=events.append,
    )

    assert result.num_samples == 4
    assert result.num_batches == 2
    assert [event.batch for event in events] == [1, 2]
    assert all(
        event.total_batches == 2
        for event in events
    )
    assert all(
        "L1" in event.metrics
        for event in events
    )
