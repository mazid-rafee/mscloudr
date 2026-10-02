"""Stable dataset split manifests.

A research split should be defined by sample identity, not by the current
length/order of a directory listing. These helpers create and validate a JSON
manifest containing the actual sample IDs assigned to each partition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .reproducibility import make_torch_generator


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def validate(self) -> None:
        values = (self.train, self.val, self.test)
        if any(v < 0.0 for v in values):
            raise ValueError("split fractions must be non-negative")
        if abs(sum(values) - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to 1.0")


def _fingerprint(sample_ids: list[str]) -> str:
    payload = "\n".join(sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_split_manifest(
    sample_ids: Iterable[str],
    *,
    split_seed: int,
    fractions: SplitFractions = SplitFractions(),
) -> dict:
    """Create a deterministic split manifest from unique sample IDs.

    Input ordering is canonicalized by sorting, so filesystem enumeration order
    cannot change the resulting split. The manifest stores IDs rather than
    integer indices to remain stable if loader internals later change.
    """

    fractions.validate()

    ids = sorted(str(sample_id) for sample_id in sample_ids)
    if not ids:
        raise ValueError("sample_ids must not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("sample_ids must be unique")

    n = len(ids)
    n_train = int(n * fractions.train)
    n_val = int(n * fractions.val)

    generator = make_torch_generator(split_seed)
    permutation = torch.randperm(n, generator=generator).tolist()

    train_ids = [ids[i] for i in permutation[:n_train]]
    val_ids = [ids[i] for i in permutation[n_train : n_train + n_val]]
    test_ids = [ids[i] for i in permutation[n_train + n_val :]]

    return {
        "format_version": 1,
        "split_seed": split_seed,
        "fractions": {
            "train": fractions.train,
            "val": fractions.val,
            "test": fractions.test,
        },
        "num_samples": n,
        "canonical_sample_ids_sha256": _fingerprint(ids),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "splits": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }


def validate_split_manifest(manifest: dict, available_sample_ids: Iterable[str]) -> None:
    """Raise if a manifest is incompatible with the available dataset."""

    available = sorted(str(sample_id) for sample_id in available_sample_ids)
    expected_list = [
        sample_id
        for split_ids in manifest["splits"].values()
        for sample_id in split_ids
    ]
    expected = set(expected_list)

    if len(expected_list) != manifest["num_samples"] or len(expected) != len(expected_list):
        raise ValueError("manifest contains duplicate or missing split entries")
    if expected != set(available):
        missing = sorted(expected - set(available))
        extra = sorted(set(available) - expected)
        raise ValueError(
            "dataset does not match split manifest: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if _fingerprint(available) != manifest["canonical_sample_ids_sha256"]:
        raise ValueError("dataset sample-ID fingerprint does not match manifest")


def save_split_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_split_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
