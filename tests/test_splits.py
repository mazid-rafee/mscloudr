import pytest

from mscloudr.splits import (
    SplitFractions,
    create_split_manifest,
    validate_split_manifest,
)


def test_split_manifest_is_order_invariant():
    ids = [f"sample-{i:03d}" for i in range(20)]

    a = create_split_manifest(ids, split_seed=42)
    b = create_split_manifest(reversed(ids), split_seed=42)

    assert a == b


def test_split_seed_changes_membership():
    ids = [f"sample-{i:03d}" for i in range(100)]

    a = create_split_manifest(ids, split_seed=1)
    b = create_split_manifest(ids, split_seed=2)

    assert a["splits"]["train"] != b["splits"]["train"]


def test_split_counts_cover_all_samples():
    ids = [f"sample-{i:03d}" for i in range(23)]
    manifest = create_split_manifest(
        ids,
        split_seed=42,
        fractions=SplitFractions(train=0.8, val=0.1, test=0.1),
    )

    assert manifest["counts"] == {"train": 18, "val": 2, "test": 3}
    validate_split_manifest(manifest, ids)


def test_manifest_detects_dataset_change():
    ids = [f"sample-{i:03d}" for i in range(10)]
    manifest = create_split_manifest(ids, split_seed=42)

    with pytest.raises(ValueError):
        validate_split_manifest(manifest, ids[:-1])


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        create_split_manifest(["a", "a", "b"], split_seed=42)
