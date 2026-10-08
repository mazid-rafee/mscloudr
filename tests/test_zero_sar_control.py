import torch
from torch.utils.data import Dataset

from mscloudr.data.ablation import ZeroSARDataset


class _ToyDataset(Dataset):
    def __init__(self):
        self.samples = ("a", "b")

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "cloudy": torch.full((13, 4, 4), float(index + 1)),
            "sar": torch.full((2, 4, 4), float(index + 3)),
            "target": torch.full((13, 4, 4), float(index + 5)),
            "sample_id": f"sample-{index}",
        }


def test_zero_sar_dataset_preserves_non_sar_fields_and_samples():
    base = _ToyDataset()
    wrapped = ZeroSARDataset(base)

    assert wrapped.samples == base.samples
    item = wrapped[1]
    base_item = base[1]

    assert torch.equal(item["cloudy"], base_item["cloudy"])
    assert torch.equal(item["target"], base_item["target"])
    assert item["sample_id"] == base_item["sample_id"]
    assert torch.equal(item["sar"], torch.zeros_like(base_item["sar"]))


def test_zero_sar_dataset_does_not_mutate_base_item():
    base = _ToyDataset()
    before = base[0]["sar"].clone()
    _ = ZeroSARDataset(base)[0]
    after = base[0]["sar"]

    assert torch.equal(before, after)
