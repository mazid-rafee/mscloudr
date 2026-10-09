import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mscloudr.bridge import linear_alpha
from mscloudr.reproducibility import make_torch_generator
from mscloudr.runner import fit


class _TinyCurvedDataset(Dataset):
    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        value = float(index + 1) / float(self.n + 1)
        target = torch.full((1, 4, 4), 0.4 * value)
        cloudy = target + 0.2
        sar = torch.full((1, 4, 4), 0.1)
        return {
            "target": target,
            "cloudy": cloudy,
            "sar": sar,
        }


class _TinyCurvedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.8))

    def bridge_curvature(self, cloudy, sar):
        del sar
        return torch.zeros_like(cloudy)

    def forward(self, x_t, conditioning, sar):
        del conditioning, sar
        return self.scale * x_t


def _loader(seed, *, shuffle):
    return DataLoader(
        _TinyCurvedDataset(),
        batch_size=2,
        shuffle=shuffle,
        generator=make_torch_generator(seed),
        num_workers=0,
    )


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_fit_saves_nfe_matched_curved_checkpoints(tmp_path):
    model = _TinyCurvedModel()

    history = fit(
        model,
        train_loader=_loader(1, shuffle=True),
        val_loader=_loader(2, shuffle=False),
        schedule=linear_alpha,
        schedule_name="canonical_alpha",
        total_steps=1000,
        sampler_generator=make_torch_generator(3),
        output_dir=tmp_path,
        epochs=2,
        device="cpu",
        conditioning_mode="physical_alpha",
        lr=1e-2,
        max_train_batches=1,
        max_val_batches=1,
        curved_validation_nfes=(2, 3, 5),
    )

    assert len(history) == 2

    for nfe in (2, 3, 5):
        path = tmp_path / "checkpoints" / f"best_curved_nfe{nfe}.pt"
        assert path.is_file()

        payload = _load(path)
        metric_name = f"val_curved_nfe{nfe}_l1"
        assert metric_name in payload["metrics"]
        assert (
            payload["run_metadata"]["checkpoint_selection_metric"]
            == metric_name
        )
        assert payload["run_metadata"]["matched_inference_nfe"] == nfe

    curved_history = tmp_path / "curved_validation_history.json"
    assert curved_history.is_file()
