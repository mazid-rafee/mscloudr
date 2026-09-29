"""Official ROI-level SEN12MS-CR split used by UnCRtainTS.

The lists below are transcribed programmatically from
PatrickTUM/UnCRtainTS commit 5e1f1b58e993645e765b64e10b6e9c7ff828b36f,
class SEN12MSCR in data/dataLoader.py.

The split is geographic/ROI-level, not a random patch split.
"""

from __future__ import annotations

from typing import Iterable

from .sen12mscr import SEN12MSCRSample

UNCRTAINTS_SOURCE_REPO = "PatrickTUM/UnCRtainTS"
UNCRTAINTS_SOURCE_COMMIT = "5e1f1b58e993645e765b64e10b6e9c7ff828b36f"

UNCRTAINTS_TRAIN_GROUPS = (
    "ROIs1970_fall_s1/s1_3",
    "ROIs1970_fall_s1/s1_22",
    "ROIs1970_fall_s1/s1_148",
    "ROIs1970_fall_s1/s1_107",
    "ROIs1970_fall_s1/s1_1",
    "ROIs1970_fall_s1/s1_114",
    "ROIs1970_fall_s1/s1_135",
    "ROIs1970_fall_s1/s1_40",
    "ROIs1970_fall_s1/s1_42",
    "ROIs1970_fall_s1/s1_31",
    "ROIs1970_fall_s1/s1_149",
    "ROIs1970_fall_s1/s1_64",
    "ROIs1970_fall_s1/s1_28",
    "ROIs1970_fall_s1/s1_144",
    "ROIs1970_fall_s1/s1_57",
    "ROIs1970_fall_s1/s1_35",
    "ROIs1970_fall_s1/s1_133",
    "ROIs1970_fall_s1/s1_30",
    "ROIs1970_fall_s1/s1_134",
    "ROIs1970_fall_s1/s1_141",
    "ROIs1970_fall_s1/s1_112",
    "ROIs1970_fall_s1/s1_116",
    "ROIs1970_fall_s1/s1_37",
    "ROIs1970_fall_s1/s1_26",
    "ROIs1970_fall_s1/s1_77",
    "ROIs1970_fall_s1/s1_100",
    "ROIs1970_fall_s1/s1_83",
    "ROIs1970_fall_s1/s1_71",
    "ROIs1970_fall_s1/s1_93",
    "ROIs1970_fall_s1/s1_119",
    "ROIs1970_fall_s1/s1_104",
    "ROIs1970_fall_s1/s1_136",
    "ROIs1970_fall_s1/s1_6",
    "ROIs1970_fall_s1/s1_41",
    "ROIs1970_fall_s1/s1_125",
    "ROIs1970_fall_s1/s1_91",
    "ROIs1970_fall_s1/s1_131",
    "ROIs1970_fall_s1/s1_120",
    "ROIs1970_fall_s1/s1_110",
    "ROIs1970_fall_s1/s1_19",
    "ROIs1970_fall_s1/s1_14",
    "ROIs1970_fall_s1/s1_81",
    "ROIs1970_fall_s1/s1_39",
    "ROIs1970_fall_s1/s1_109",
    "ROIs1970_fall_s1/s1_33",
    "ROIs1970_fall_s1/s1_88",
    "ROIs1970_fall_s1/s1_11",
    "ROIs1970_fall_s1/s1_128",
    "ROIs1970_fall_s1/s1_142",
    "ROIs1970_fall_s1/s1_122",
    "ROIs1970_fall_s1/s1_4",
    "ROIs1970_fall_s1/s1_27",
    "ROIs1970_fall_s1/s1_147",
    "ROIs1970_fall_s1/s1_85",
    "ROIs1970_fall_s1/s1_82",
    "ROIs1970_fall_s1/s1_105",
    "ROIs1158_spring_s1/s1_9",
    "ROIs1158_spring_s1/s1_1",
    "ROIs1158_spring_s1/s1_124",
    "ROIs1158_spring_s1/s1_40",
    "ROIs1158_spring_s1/s1_101",
    "ROIs1158_spring_s1/s1_21",
    "ROIs1158_spring_s1/s1_134",
    "ROIs1158_spring_s1/s1_145",
    "ROIs1158_spring_s1/s1_141",
    "ROIs1158_spring_s1/s1_66",
    "ROIs1158_spring_s1/s1_8",
    "ROIs1158_spring_s1/s1_26",
    "ROIs1158_spring_s1/s1_77",
    "ROIs1158_spring_s1/s1_113",
    "ROIs1158_spring_s1/s1_100",
    "ROIs1158_spring_s1/s1_117",
    "ROIs1158_spring_s1/s1_119",
    "ROIs1158_spring_s1/s1_6",
    "ROIs1158_spring_s1/s1_58",
    "ROIs1158_spring_s1/s1_120",
    "ROIs1158_spring_s1/s1_110",
    "ROIs1158_spring_s1/s1_126",
    "ROIs1158_spring_s1/s1_115",
    "ROIs1158_spring_s1/s1_121",
    "ROIs1158_spring_s1/s1_39",
    "ROIs1158_spring_s1/s1_109",
    "ROIs1158_spring_s1/s1_63",
    "ROIs1158_spring_s1/s1_75",
    "ROIs1158_spring_s1/s1_132",
    "ROIs1158_spring_s1/s1_128",
    "ROIs1158_spring_s1/s1_142",
    "ROIs1158_spring_s1/s1_15",
    "ROIs1158_spring_s1/s1_45",
    "ROIs1158_spring_s1/s1_97",
    "ROIs1158_spring_s1/s1_147",
    "ROIs1868_summer_s1/s1_90",
    "ROIs1868_summer_s1/s1_87",
    "ROIs1868_summer_s1/s1_25",
    "ROIs1868_summer_s1/s1_124",
    "ROIs1868_summer_s1/s1_114",
    "ROIs1868_summer_s1/s1_135",
    "ROIs1868_summer_s1/s1_40",
    "ROIs1868_summer_s1/s1_101",
    "ROIs1868_summer_s1/s1_42",
    "ROIs1868_summer_s1/s1_31",
    "ROIs1868_summer_s1/s1_36",
    "ROIs1868_summer_s1/s1_139",
    "ROIs1868_summer_s1/s1_56",
    "ROIs1868_summer_s1/s1_133",
    "ROIs1868_summer_s1/s1_55",
    "ROIs1868_summer_s1/s1_43",
    "ROIs1868_summer_s1/s1_113",
    "ROIs1868_summer_s1/s1_76",
    "ROIs1868_summer_s1/s1_123",
    "ROIs1868_summer_s1/s1_143",
    "ROIs1868_summer_s1/s1_93",
    "ROIs1868_summer_s1/s1_125",
    "ROIs1868_summer_s1/s1_89",
    "ROIs1868_summer_s1/s1_120",
    "ROIs1868_summer_s1/s1_126",
    "ROIs1868_summer_s1/s1_72",
    "ROIs1868_summer_s1/s1_115",
    "ROIs1868_summer_s1/s1_121",
    "ROIs1868_summer_s1/s1_146",
    "ROIs1868_summer_s1/s1_140",
    "ROIs1868_summer_s1/s1_95",
    "ROIs1868_summer_s1/s1_102",
    "ROIs1868_summer_s1/s1_7",
    "ROIs1868_summer_s1/s1_11",
    "ROIs1868_summer_s1/s1_132",
    "ROIs1868_summer_s1/s1_15",
    "ROIs1868_summer_s1/s1_137",
    "ROIs1868_summer_s1/s1_4",
    "ROIs1868_summer_s1/s1_27",
    "ROIs1868_summer_s1/s1_147",
    "ROIs1868_summer_s1/s1_86",
    "ROIs1868_summer_s1/s1_47",
    "ROIs2017_winter_s1/s1_68",
    "ROIs2017_winter_s1/s1_25",
    "ROIs2017_winter_s1/s1_62",
    "ROIs2017_winter_s1/s1_135",
    "ROIs2017_winter_s1/s1_42",
    "ROIs2017_winter_s1/s1_64",
    "ROIs2017_winter_s1/s1_21",
    "ROIs2017_winter_s1/s1_55",
    "ROIs2017_winter_s1/s1_112",
    "ROIs2017_winter_s1/s1_116",
    "ROIs2017_winter_s1/s1_8",
    "ROIs2017_winter_s1/s1_59",
    "ROIs2017_winter_s1/s1_49",
    "ROIs2017_winter_s1/s1_104",
    "ROIs2017_winter_s1/s1_81",
    "ROIs2017_winter_s1/s1_146",
    "ROIs2017_winter_s1/s1_75",
    "ROIs2017_winter_s1/s1_94",
    "ROIs2017_winter_s1/s1_102",
    "ROIs2017_winter_s1/s1_61",
    "ROIs2017_winter_s1/s1_47",
    "ROIs1868_summer_s1/s1_100",
)

UNCRTAINTS_VAL_GROUPS = (
    "ROIs2017_winter_s1/s1_22",
    "ROIs1868_summer_s1/s1_19",
    "ROIs1970_fall_s1/s1_65",
    "ROIs1158_spring_s1/s1_17",
    "ROIs2017_winter_s1/s1_107",
    "ROIs1868_summer_s1/s1_80",
    "ROIs1868_summer_s1/s1_127",
    "ROIs2017_winter_s1/s1_130",
    "ROIs1868_summer_s1/s1_17",
    "ROIs2017_winter_s1/s1_84",
)

UNCRTAINTS_TEST_GROUPS = (
    "ROIs1158_spring_s1/s1_106",
    "ROIs1158_spring_s1/s1_123",
    "ROIs1158_spring_s1/s1_140",
    "ROIs1158_spring_s1/s1_31",
    "ROIs1158_spring_s1/s1_44",
    "ROIs1868_summer_s1/s1_119",
    "ROIs1868_summer_s1/s1_73",
    "ROIs1970_fall_s1/s1_139",
    "ROIs2017_winter_s1/s1_108",
    "ROIs2017_winter_s1/s1_63",
)

UNCRTAINTS_SPLIT_GROUPS = {
    "train": frozenset(UNCRTAINTS_TRAIN_GROUPS),
    "val": frozenset(UNCRTAINTS_VAL_GROUPS),
    "test": frozenset(UNCRTAINTS_TEST_GROUPS),
}


def sample_group_key(sample: SEN12MSCRSample) -> str:
    """Return the group key format used in the reference UnCRtainTS loader."""

    return f"{sample.season_prefix}_s1/s1_{sample.roi_id}"


def validate_uncrtaints_split_definition() -> None:
    """Validate that the reference split is disjoint and covers 175 ROIs."""

    train_groups = UNCRTAINTS_SPLIT_GROUPS["train"]
    val_groups = UNCRTAINTS_SPLIT_GROUPS["val"]
    test_groups = UNCRTAINTS_SPLIT_GROUPS["test"]

    if len(train_groups) != 155 or len(val_groups) != 10 or len(test_groups) != 10:
        raise ValueError("unexpected UnCRtainTS ROI split sizes")
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise ValueError("UnCRtainTS ROI splits overlap")
    if len(train_groups | val_groups | test_groups) != 175:
        raise ValueError("UnCRtainTS ROI split does not cover 175 unique ROIs")


def partition_samples_by_uncrtaints_split(
    samples: Iterable[SEN12MSCRSample],
) -> dict[str, list[SEN12MSCRSample]]:
    """Partition discovered samples using the reference ROI split.

    Raises if a discovered sample belongs to an ROI absent from the reference
    split.  This prevents silently leaking a new ROI into train/val/test.
    """

    validate_uncrtaints_split_definition()
    group_to_split = {
        group: split
        for split, groups in UNCRTAINTS_SPLIT_GROUPS.items()
        for group in groups
    }

    partitions: dict[str, list[SEN12MSCRSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    unknown: set[str] = set()

    for sample in samples:
        group = sample_group_key(sample)
        split = group_to_split.get(group)
        if split is None:
            unknown.add(group)
        else:
            partitions[split].append(sample)

    if unknown:
        preview = ", ".join(sorted(unknown)[:10])
        raise ValueError(
            f"{len(unknown)} discovered ROI groups are absent from the "
            f"UnCRtainTS split definition: {preview}"
        )

    return partitions
