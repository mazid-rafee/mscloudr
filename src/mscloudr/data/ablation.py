"""Dataset wrappers for controlled modality ablations."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class ZeroSARDataset(Dataset):
    """Replace SAR with exact zeros while leaving every other field unchanged.

    This wrapper is used for a capacity-matched optical-only control: the
    CanonicalBridgeNet architecture is unchanged, but no scene-specific SAR
    information is available during training or evaluation.
    """

    def __init__(self, base: Dataset):
        self.base = base
        self.samples = getattr(base, "samples", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        sar = item.get("sar")
        if not torch.is_tensor(sar):
            raise TypeError("wrapped dataset item must contain tensor key 'sar'")
        item["sar"] = torch.zeros_like(sar)
        return item
