"""Reference SEN12MS-CR datasets and DataLoaders.

Benchmark membership is the fixed ROI-level split copied from UnCRtainTS.
There is no random train/val/test split in this module.

Randomness is limited to training-order shuffling and is controlled by the
experiment train_seed. Bridge timestep sampling is a separate subsystem and
must use sampler_seed instead.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from torch.utils.data import DataLoader

from mscloudr.reproducibility import make_torch_generator, seed_dataloader_worker

from .sen12mscr import SEN12MSCRDataset, SEN12MSCRSample
from .sen12mscr_splits import (
    UNCRTAINTS_SOURCE_COMMIT,
    UNCRTAINTS_SOURCE_REPO,
    partition_samples_by_uncrtaints_split,
    sample_group_key,
)

REFERENCE_PROTOCOL = "uncrtaints_sen12mscr_roi_split"

FROZEN_DATASET_SAMPLE_COUNT = 122_217
FROZEN_DATASET_SAMPLE_IDS_SHA256 = (
    "4ccf44fe4cb28dddcbb4b1e6deb00bed24a70106910e8ae8a2113c6700c5bae3"
)

FROZEN_SPLIT_AUDIT = {
    "train": {
        "num_samples": 107_142,
        "num_roi_groups": 155,
        "sample_ids_sha256": (
            "2987371bc1f28329c77c38f035eec51c5ff9699e2830a252c670eec776c4019b"
        ),
    },
    "val": {
        "num_samples": 7_176,
        "num_roi_groups": 10,
        "sample_ids_sha256": (
            "35c03ebedb5eb01dd6f4b922ba8ad62788a82c14b3dc32055eb109dd521f8314"
        ),
    },
    "test": {
        "num_samples": 7_899,
        "num_roi_groups": 10,
        "sample_ids_sha256": (
            "97577d0b6f73db09409ed4a7cd3ca1d533cb4b9d10c658a2a37f0d7956d5138b"
        ),
    },
}


@dataclass(frozen=True)
class ReferenceDatasets:
    train: SEN12MSCRDataset
    val: SEN12MSCRDataset
    test: SEN12MSCRDataset

    def as_dict(self) -> dict[str, SEN12MSCRDataset]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }


@dataclass(frozen=True)
class ReferenceDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader

    def as_dict(self) -> dict[str, DataLoader]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }


def _fingerprint_sample_ids(
    samples: Sequence[SEN12MSCRSample],
) -> str:
    ids = sorted(sample.sample_id for sample in samples)
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reference_split_audit(
    partitions: Mapping[str, Sequence[SEN12MSCRSample]],
) -> dict[str, object]:
    """Return stable membership metadata for the fixed ROI split."""

    if set(partitions) != {"train", "val", "test"}:
        raise ValueError("partitions must contain exactly train, val, and test")

    split_audit: dict[str, dict[str, object]] = {}
    all_samples: list[SEN12MSCRSample] = []

    for split in ("train", "val", "test"):
        samples = list(partitions[split])
        all_samples.extend(samples)
        split_audit[split] = {
            "num_samples": len(samples),
            "num_roi_groups": len(
                {sample_group_key(sample) for sample in samples}
            ),
            "sample_ids_sha256": _fingerprint_sample_ids(samples),
        }

    return {
        "protocol": REFERENCE_PROTOCOL,
        "source_repo": UNCRTAINTS_SOURCE_REPO,
        "source_commit": UNCRTAINTS_SOURCE_COMMIT,
        "dataset_num_samples_after_ignored": len(all_samples),
        "dataset_sample_ids_sha256": _fingerprint_sample_ids(all_samples),
        "splits": split_audit,
    }


def validate_frozen_reference_split(
    partitions: Mapping[str, Sequence[SEN12MSCRSample]],
) -> dict[str, object]:
    """Verify membership against the locally audited paper protocol."""

    audit = reference_split_audit(partitions)

    if audit["dataset_num_samples_after_ignored"] != FROZEN_DATASET_SAMPLE_COUNT:
        raise ValueError(
            "reference dataset sample count mismatch: "
            f"expected {FROZEN_DATASET_SAMPLE_COUNT}, "
            f"got {audit['dataset_num_samples_after_ignored']}"
        )

    if audit["dataset_sample_ids_sha256"] != FROZEN_DATASET_SAMPLE_IDS_SHA256:
        raise ValueError("reference dataset sample-ID fingerprint mismatch")

    split_audit = audit["splits"]
    for split, expected in FROZEN_SPLIT_AUDIT.items():
        actual = split_audit[split]
        for field, expected_value in expected.items():
            if actual[field] != expected_value:
                raise ValueError(
                    f"{split} {field} mismatch: "
                    f"expected {expected_value!r}, got {actual[field]!r}"
                )

    return audit


def build_reference_datasets(
    samples: Iterable[SEN12MSCRSample],
    *,
    include_sample_id: bool = True,
    strict_channels: bool = True,
    verify_frozen_membership: bool = False,
) -> ReferenceDatasets:
    """Partition samples by the fixed ROI protocol and wrap each split."""

    partitions = partition_samples_by_uncrtaints_split(samples)

    if verify_frozen_membership:
        validate_frozen_reference_split(partitions)

    return ReferenceDatasets(
        train=SEN12MSCRDataset(
            partitions["train"],
            include_sample_id=include_sample_id,
            strict_channels=strict_channels,
        ),
        val=SEN12MSCRDataset(
            partitions["val"],
            include_sample_id=include_sample_id,
            strict_channels=strict_channels,
        ),
        test=SEN12MSCRDataset(
            partitions["test"],
            include_sample_id=include_sample_id,
            strict_channels=strict_channels,
        ),
    )


def build_reference_dataloaders(
    datasets: ReferenceDatasets,
    *,
    batch_size: int,
    num_workers: int,
    train_seed: int,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
) -> ReferenceDataLoaders:
    """Build deterministic benchmark DataLoaders."""

    batch_size = int(batch_size)
    num_workers = int(num_workers)
    train_seed = int(train_seed)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if train_seed < 0:
        raise ValueError("train_seed must be non-negative")

    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "worker_init_fn": seed_dataloader_worker,
        "persistent_workers": bool(persistent_workers),
    }

    # Independent generator objects prevent iteration of one split from
    # consuming another split RNG stream. They intentionally use the same
    # train_seed because this is one training-randomness subsystem.
    train_generator = make_torch_generator(train_seed)
    val_generator = make_torch_generator(train_seed)
    test_generator = make_torch_generator(train_seed)

    return ReferenceDataLoaders(
        train=DataLoader(
            datasets.train,
            shuffle=True,
            generator=train_generator,
            **common,
        ),
        val=DataLoader(
            datasets.val,
            shuffle=False,
            generator=val_generator,
            **common,
        ),
        test=DataLoader(
            datasets.test,
            shuffle=False,
            generator=test_generator,
            **common,
        ),
    )
