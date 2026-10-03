"""Deterministic 10% pilot subset for fast SEN12MS-CR experiments.

The pilot starts from the fixed UnCRtainTS ROI-disjoint train/val/test split,
then downsamples *within each split*.  Sampling is stratified by season-scoped
ROI group so that every retained split preserves broad geographic and seasonal
coverage while using about 10% of the patches.

This module never moves an ROI between train/val/test.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .loaders import ReferenceDatasets, validate_frozen_reference_split
from .sen12mscr import SEN12MSCRDataset, SEN12MSCRSample
from .sen12mscr_splits import (
    partition_samples_by_uncrtaints_split,
    sample_group_key,
)

PILOT_PROTOCOL = "uncrtaints_roi_disjoint_pilot10_roi_season_stratified_v1"
PILOT_FRACTION = 0.10
PILOT_SEED = 42


def _fingerprint_sample_ids(samples: Sequence[SEN12MSCRSample]) -> str:
    ids = sorted(sample.sample_id for sample in samples)
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_rank(sample: SEN12MSCRSample, *, split: str, seed: int) -> tuple[bytes, str]:
    payload = f"{seed}\0{split}\0{sample.sample_id}".encode("utf-8")
    return hashlib.sha256(payload).digest(), sample.sample_id


def _allocate_group_quotas(
    group_sizes: Mapping[str, int],
    *,
    target: int,
) -> dict[str, int]:
    """Allocate exactly ``target`` examples approximately proportional to group size.

    A minimum of one example per ROI group is enforced.  This is deliberate for
    the pilot benchmark: the full reference split already defines geographic
    separation, while retaining every ROI group maximizes geographic coverage.
    """

    if target <= 0:
        raise ValueError("target must be positive")
    if not group_sizes:
        raise ValueError("group_sizes must not be empty")
    if any(size <= 0 for size in group_sizes.values()):
        raise ValueError("group sizes must be positive")

    total = sum(group_sizes.values())
    groups = sorted(group_sizes)
    if target > total:
        raise ValueError("target cannot exceed available samples")
    if target < len(groups):
        raise ValueError(
            "target is too small to retain at least one sample from every ROI group"
        )

    ideals = {
        group: target * group_sizes[group] / total
        for group in groups
    }
    quotas = {
        group: min(group_sizes[group], max(1, math.floor(ideals[group])))
        for group in groups
    }

    current = sum(quotas.values())
    while current < target:
        candidates = [group for group in groups if quotas[group] < group_sizes[group]]
        if not candidates:
            raise RuntimeError("unable to allocate requested pilot target")
        # Most under-allocated relative to its proportional ideal gets the next slot.
        group = max(
            candidates,
            key=lambda key: (ideals[key] - quotas[key], group_sizes[key], key),
        )
        quotas[group] += 1
        current += 1

    while current > target:
        candidates = [group for group in groups if quotas[group] > 1]
        if not candidates:
            raise RuntimeError("unable to reduce pilot quotas to requested target")
        # Most over-allocated relative to its proportional ideal loses one slot.
        group = max(
            candidates,
            key=lambda key: (quotas[key] - ideals[key], -group_sizes[key], key),
        )
        quotas[group] -= 1
        current -= 1

    return quotas


def stratified_subset_partitions(
    partitions: Mapping[str, Sequence[SEN12MSCRSample]],
    *,
    fraction: float = PILOT_FRACTION,
    seed: int = PILOT_SEED,
) -> dict[str, list[SEN12MSCRSample]]:
    """Create deterministic per-split subsets stratified by season-scoped ROI.

    The caller supplies already-separated train/val/test partitions.  Each split
    is sampled independently, so ROI membership can never cross split boundaries.
    Within a split, quotas are allocated per ``sample_group_key`` (season + ROI)
    and samples are chosen by a stable SHA-256 ranking rather than input order.
    """

    if set(partitions) != {"train", "val", "test"}:
        raise ValueError("partitions must contain exactly train, val, and test")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    selected: dict[str, list[SEN12MSCRSample]] = {}

    for split in ("train", "val", "test"):
        samples = list(partitions[split])
        if not samples:
            raise ValueError(f"{split} partition must not be empty")

        by_group: dict[str, list[SEN12MSCRSample]] = defaultdict(list)
        for sample in samples:
            by_group[sample_group_key(sample)].append(sample)

        target = int(round(len(samples) * fraction))
        if target < len(by_group):
            raise ValueError(
                f"{split} target={target} is too small to retain all "
                f"{len(by_group)} ROI groups at fraction={fraction}"
            )

        quotas = _allocate_group_quotas(
            {group: len(group_samples) for group, group_samples in by_group.items()},
            target=target,
        )

        split_selected: list[SEN12MSCRSample] = []
        for group in sorted(by_group):
            ranked = sorted(
                by_group[group],
                key=lambda sample: _stable_rank(sample, split=split, seed=seed),
            )
            split_selected.extend(ranked[: quotas[group]])

        split_selected.sort(key=lambda sample: sample.sample_id)
        selected[split] = split_selected

    return selected


def _counts_by_season(samples: Sequence[SEN12MSCRSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.season_prefix] = counts.get(sample.season_prefix, 0) + 1
    return dict(sorted(counts.items()))


def create_pilot_manifest(
    full_partitions: Mapping[str, Sequence[SEN12MSCRSample]],
    pilot_partitions: Mapping[str, Sequence[SEN12MSCRSample]],
    *,
    fraction: float = PILOT_FRACTION,
    seed: int = PILOT_SEED,
) -> dict[str, object]:
    """Create a reproducibility manifest for one deterministic pilot subset."""

    if set(full_partitions) != {"train", "val", "test"}:
        raise ValueError("full_partitions must contain exactly train, val, and test")
    if set(pilot_partitions) != {"train", "val", "test"}:
        raise ValueError("pilot_partitions must contain exactly train, val, and test")

    splits: dict[str, object] = {}
    all_selected: list[SEN12MSCRSample] = []

    for split in ("train", "val", "test"):
        full = list(full_partitions[split])
        selected = list(pilot_partitions[split])
        full_ids = {sample.sample_id for sample in full}
        selected_ids = [sample.sample_id for sample in selected]

        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"{split} pilot subset contains duplicate sample IDs")
        if not set(selected_ids).issubset(full_ids):
            raise ValueError(f"{split} pilot subset contains samples outside its base split")

        full_groups = {sample_group_key(sample) for sample in full}
        selected_groups = {sample_group_key(sample) for sample in selected}
        if selected_groups != full_groups:
            raise ValueError(
                f"{split} pilot subset does not retain every ROI group "
                f"({len(selected_groups)} vs {len(full_groups)})"
            )

        all_selected.extend(selected)
        splits[split] = {
            "full_num_samples": len(full),
            "num_samples": len(selected),
            "realized_fraction": len(selected) / len(full),
            "num_roi_groups": len(selected_groups),
            "season_counts_full": _counts_by_season(full),
            "season_counts": _counts_by_season(selected),
            "sample_ids_sha256": _fingerprint_sample_ids(selected),
            "sample_ids": sorted(selected_ids),
        }

    return {
        "format_version": 1,
        "protocol": PILOT_PROTOCOL,
        "fraction": fraction,
        "subset_seed": seed,
        "stratification": "season_scoped_roi",
        "selection": "stable_sha256_ranking_with_proportional_roi_quotas",
        "base_split": "UnCRtainTS fixed ROI-disjoint SEN12MS-CR split",
        "num_samples": len(all_selected),
        "sample_ids_sha256": _fingerprint_sample_ids(all_selected),
        "splits": splits,
    }


def pilot_split_audit(manifest: Mapping[str, object]) -> dict[str, object]:
    """Return compact provenance suitable for checkpoint metadata."""

    splits_raw = manifest["splits"]
    if not isinstance(splits_raw, Mapping):
        raise ValueError("manifest.splits must be a mapping")

    compact_splits: dict[str, object] = {}
    for split in ("train", "val", "test"):
        entry = splits_raw[split]
        if not isinstance(entry, Mapping):
            raise ValueError(f"manifest split {split!r} must be a mapping")
        compact_splits[split] = {
            key: entry[key]
            for key in (
                "full_num_samples",
                "num_samples",
                "realized_fraction",
                "num_roi_groups",
                "season_counts_full",
                "season_counts",
                "sample_ids_sha256",
            )
        }

    return {
        "protocol": manifest["protocol"],
        "fraction": manifest["fraction"],
        "subset_seed": manifest["subset_seed"],
        "stratification": manifest["stratification"],
        "selection": manifest["selection"],
        "base_split": manifest["base_split"],
        "num_samples": manifest["num_samples"],
        "sample_ids_sha256": manifest["sample_ids_sha256"],
        "splits": compact_splits,
    }


def build_pilot_datasets(
    samples: Iterable[SEN12MSCRSample],
    *,
    include_sample_id: bool = True,
    strict_channels: bool = True,
    verify_frozen_membership: bool = True,
) -> tuple[ReferenceDatasets, dict[str, object]]:
    """Build the fixed 10% pilot datasets and full sample-ID manifest."""

    full_partitions = partition_samples_by_uncrtaints_split(samples)
    if verify_frozen_membership:
        validate_frozen_reference_split(full_partitions)

    pilot_partitions = stratified_subset_partitions(
        full_partitions,
        fraction=PILOT_FRACTION,
        seed=PILOT_SEED,
    )
    manifest = create_pilot_manifest(
        full_partitions,
        pilot_partitions,
        fraction=PILOT_FRACTION,
        seed=PILOT_SEED,
    )

    datasets = ReferenceDatasets(
        train=SEN12MSCRDataset(
            pilot_partitions["train"],
            include_sample_id=include_sample_id,
            strict_channels=strict_channels,
        ),
        val=SEN12MSCRDataset(
            pilot_partitions["val"],
            include_sample_id=include_sample_id,
            strict_channels=strict_channels,
        ),
        test=SEN12MSCRDataset(
            pilot_partitions["test"],
            include_sample_id=include_sample_id,
            strict_channels=strict_channels,
        ),
    )
    return datasets, manifest


def save_pilot_manifest(manifest: Mapping[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
