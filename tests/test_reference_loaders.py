from pathlib import Path

from torch.utils.data import RandomSampler, SequentialSampler

from mscloudr.data.loaders import (
    FROZEN_DATASET_SAMPLE_COUNT,
    FROZEN_SPLIT_AUDIT,
    REFERENCE_PROTOCOL,
    build_reference_dataloaders,
    build_reference_datasets,
    reference_split_audit,
    validate_frozen_reference_split,
)
from mscloudr.data.sen12mscr import SEN12MSCRSample


def _sample(season: str, roi: str, patch: int) -> SEN12MSCRSample:
    dummy = Path(f"/tmp/{season}_{roi}_{patch}.tif")
    return SEN12MSCRSample(
        sample_id=(
            f"{season}_s2_cloudy/"
            f"s2_cloudy_{roi}/"
            f"{season}_s2_cloudy_{roi}_p{patch}.tif"
        ),
        season_prefix=season,
        roi_id=roi,
        patch_id=f"p{patch}",
        cloudy_path=dummy,
        sar_path=dummy,
        target_path=dummy,
    )


def _tiny_reference_samples():
    samples = []
    for patch in range(8):
        samples.append(_sample("ROIs2017_winter", "68", patch))
    for patch in range(3):
        samples.append(_sample("ROIs2017_winter", "22", patch))
    for patch in range(4):
        samples.append(_sample("ROIs2017_winter", "108", patch))
    return samples


def test_reference_datasets_use_fixed_roi_membership():
    datasets = build_reference_datasets(_tiny_reference_samples())

    assert len(datasets.train) == 8
    assert len(datasets.val) == 3
    assert len(datasets.test) == 4

    assert {sample.roi_id for sample in datasets.train.samples} == {"68"}
    assert {sample.roi_id for sample in datasets.val.samples} == {"22"}
    assert {sample.roi_id for sample in datasets.test.samples} == {"108"}


def test_reference_loader_sampler_types_are_train_random_val_test_sequential():
    datasets = build_reference_datasets(_tiny_reference_samples())
    loaders = build_reference_dataloaders(
        datasets,
        batch_size=2,
        num_workers=0,
        train_seed=42,
    )

    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.val.sampler, SequentialSampler)
    assert isinstance(loaders.test.sampler, SequentialSampler)


def test_train_shuffle_order_repeats_for_same_train_seed():
    datasets = build_reference_datasets(_tiny_reference_samples())

    first = build_reference_dataloaders(
        datasets,
        batch_size=2,
        num_workers=0,
        train_seed=42,
    )
    second = build_reference_dataloaders(
        datasets,
        batch_size=2,
        num_workers=0,
        train_seed=42,
    )

    assert list(iter(first.train.sampler)) == list(iter(second.train.sampler))


def test_train_shuffle_changes_with_different_train_seed():
    datasets = build_reference_datasets(_tiny_reference_samples())

    first = build_reference_dataloaders(
        datasets,
        batch_size=2,
        num_workers=0,
        train_seed=42,
    )
    second = build_reference_dataloaders(
        datasets,
        batch_size=2,
        num_workers=0,
        train_seed=43,
    )

    assert list(iter(first.train.sampler)) != list(iter(second.train.sampler))


def test_val_and_test_order_follow_stable_dataset_order():
    datasets = build_reference_datasets(_tiny_reference_samples())
    loaders = build_reference_dataloaders(
        datasets,
        batch_size=2,
        num_workers=0,
        train_seed=42,
    )

    assert list(iter(loaders.val.sampler)) == list(range(len(datasets.val)))
    assert list(iter(loaders.test.sampler)) == list(range(len(datasets.test)))


def test_tiny_split_audit_is_membership_based_and_has_protocol_identity():
    datasets = build_reference_datasets(_tiny_reference_samples())
    partitions = {
        split: dataset.samples
        for split, dataset in datasets.as_dict().items()
    }
    audit = reference_split_audit(partitions)

    assert audit["protocol"] == REFERENCE_PROTOCOL
    assert audit["dataset_num_samples_after_ignored"] == 15
    assert audit["splits"]["train"]["num_roi_groups"] == 1
    assert audit["splits"]["val"]["num_roi_groups"] == 1
    assert audit["splits"]["test"]["num_roi_groups"] == 1


def test_frozen_audit_constants_match_verified_local_counts():
    assert FROZEN_DATASET_SAMPLE_COUNT == 122_217
    assert FROZEN_SPLIT_AUDIT["train"]["num_samples"] == 107_142
    assert FROZEN_SPLIT_AUDIT["val"]["num_samples"] == 7_176
    assert FROZEN_SPLIT_AUDIT["test"]["num_samples"] == 7_899


def test_frozen_membership_validation_rejects_tiny_fixture():
    datasets = build_reference_datasets(_tiny_reference_samples())
    partitions = {
        split: dataset.samples
        for split, dataset in datasets.as_dict().items()
    }

    try:
        validate_frozen_reference_split(partitions)
    except ValueError as error:
        assert "sample count mismatch" in str(error)
    else:
        raise AssertionError("expected frozen membership validation failure")
