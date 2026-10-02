from pathlib import Path

from mscloudr.data.sen12mscr import SEN12MSCRSample
from mscloudr.data.sen12mscr_splits import (
    UNCRTAINTS_SPLIT_GROUPS,
    partition_samples_by_uncrtaints_split,
    sample_group_key,
    validate_uncrtaints_split_definition,
)


def _sample(season: str, roi: str) -> SEN12MSCRSample:
    dummy = Path("/tmp/dummy.tif")
    return SEN12MSCRSample(
        sample_id=f"{season}_s2_cloudy/s2_cloudy_{roi}/x.tif",
        season_prefix=season,
        roi_id=roi,
        patch_id="p1",
        cloudy_path=dummy,
        sar_path=dummy,
        target_path=dummy,
    )


def test_reference_split_has_175_disjoint_roi_groups():
    validate_uncrtaints_split_definition()
    assert len(UNCRTAINTS_SPLIT_GROUPS["train"]) == 155
    assert len(UNCRTAINTS_SPLIT_GROUPS["val"]) == 10
    assert len(UNCRTAINTS_SPLIT_GROUPS["test"]) == 10


def test_sample_group_key_matches_reference_format():
    sample = _sample("ROIs2017_winter", "108")
    assert sample_group_key(sample) == "ROIs2017_winter_s1/s1_108"


def test_known_reference_test_roi_is_assigned_to_test():
    sample = _sample("ROIs2017_winter", "108")
    parts = partition_samples_by_uncrtaints_split([sample])
    assert not parts["train"]
    assert not parts["val"]
    assert parts["test"] == [sample]
