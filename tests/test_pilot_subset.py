from pathlib import Path

import pytest

from mscloudr.data.pilot import (
    PILOT_FRACTION,
    PILOT_PROTOCOL,
    PILOT_SEED,
    create_pilot_manifest,
    pilot_split_audit,
    stratified_subset_partitions,
)
from mscloudr.data.sen12mscr import SEN12MSCRSample
from mscloudr.data.sen12mscr_splits import sample_group_key


def _samples(split: str, season: str, roi: int, count: int):
    return [
        SEN12MSCRSample(
            sample_id=f"{split}/{season}/roi{roi}/p{index:04d}.tif",
            season_prefix=season,
            roi_id=str(roi),
            patch_id=f"p{index}",
            cloudy_path=Path("/tmp/cloudy"),
            sar_path=Path("/tmp/sar"),
            target_path=Path("/tmp/target"),
        )
        for index in range(count)
    ]


def _partitions():
    return {
        "train": (
            _samples("train", "ROIs2017_winter", 1, 100)
            + _samples("train", "ROIs1158_spring", 2, 80)
            + _samples("train", "ROIs1868_summer", 3, 120)
            + _samples("train", "ROIs1970_fall", 4, 100)
        ),
        "val": (
            _samples("val", "ROIs2017_winter", 10, 50)
            + _samples("val", "ROIs1158_spring", 11, 50)
            + _samples("val", "ROIs1868_summer", 12, 50)
            + _samples("val", "ROIs1970_fall", 13, 50)
        ),
        "test": (
            _samples("test", "ROIs2017_winter", 20, 70)
            + _samples("test", "ROIs1158_spring", 21, 60)
            + _samples("test", "ROIs1868_summer", 22, 40)
            + _samples("test", "ROIs1970_fall", 23, 30)
        ),
    }


def test_pilot_subset_is_exactly_about_ten_percent_and_keeps_all_rois():
    partitions = _partitions()
    subset = stratified_subset_partitions(partitions)

    assert len(subset["train"]) == 40
    assert len(subset["val"]) == 20
    assert len(subset["test"]) == 20

    for split in ("train", "val", "test"):
        full_groups = {sample_group_key(sample) for sample in partitions[split]}
        pilot_groups = {sample_group_key(sample) for sample in subset[split]}
        assert pilot_groups == full_groups


def test_pilot_subset_preserves_all_four_seasons_in_each_split():
    subset = stratified_subset_partitions(_partitions())

    for split in ("train", "val", "test"):
        seasons = {sample.season_prefix for sample in subset[split]}
        assert seasons == {
            "ROIs2017_winter",
            "ROIs1158_spring",
            "ROIs1868_summer",
            "ROIs1970_fall",
        }


def test_pilot_subset_is_order_invariant():
    partitions = _partitions()
    reversed_partitions = {
        split: list(reversed(samples))
        for split, samples in partitions.items()
    }

    a = stratified_subset_partitions(partitions)
    b = stratified_subset_partitions(reversed_partitions)

    for split in ("train", "val", "test"):
        assert [sample.sample_id for sample in a[split]] == [
            sample.sample_id for sample in b[split]
        ]


def test_pilot_seed_changes_patch_membership_without_changing_roi_coverage():
    partitions = _partitions()
    a = stratified_subset_partitions(partitions, seed=1)
    b = stratified_subset_partitions(partitions, seed=2)

    assert {
        sample.sample_id for sample in a["train"]
    } != {
        sample.sample_id for sample in b["train"]
    }
    assert {
        sample_group_key(sample) for sample in a["train"]
    } == {
        sample_group_key(sample) for sample in b["train"]
    }


def test_manifest_records_protocol_seasons_and_fingerprints():
    partitions = _partitions()
    subset = stratified_subset_partitions(partitions)
    manifest = create_pilot_manifest(partitions, subset)
    audit = pilot_split_audit(manifest)

    assert manifest["protocol"] == PILOT_PROTOCOL
    assert manifest["fraction"] == PILOT_FRACTION
    assert manifest["subset_seed"] == PILOT_SEED
    assert audit["splits"]["train"]["num_roi_groups"] == 4
    assert audit["splits"]["train"]["season_counts"] == {
        "ROIs1158_spring": 8,
        "ROIs1868_summer": 12,
        "ROIs1970_fall": 10,
        "ROIs2017_winter": 10,
    }
    assert len(audit["splits"]["train"]["sample_ids_sha256"]) == 64
    assert "sample_ids" not in audit["splits"]["train"]


def test_fraction_too_small_to_keep_every_roi_is_rejected():
    partitions = {
        "train": _samples("train", "ROIs2017_winter", 1, 1)
        + _samples("train", "ROIs1158_spring", 2, 1),
        "val": _samples("val", "ROIs1868_summer", 3, 1),
        "test": _samples("test", "ROIs1970_fall", 4, 1),
    }

    # The public subset helper deliberately raises rather than silently dropping
    # geographic groups when a requested target cannot represent all ROIs.
    with pytest.raises(ValueError):
        stratified_subset_partitions(partitions, fraction=0.1)
