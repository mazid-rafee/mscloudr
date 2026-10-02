"""Audit the reference UnCRtainTS ROI split on the local SEN12MS-CR copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mscloudr.data.sen12mscr import discover_sen12mscr, load_ignored_sample_ids
from mscloudr.data.sen12mscr_splits import (
    UNCRTAINTS_SOURCE_COMMIT,
    UNCRTAINTS_SOURCE_REPO,
    UNCRTAINTS_SPLIT_GROUPS,
    partition_samples_by_uncrtaints_split,
    sample_group_key,
)


def _sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--ignore-file",
        default="manifests/known_invalid_samples.txt",
    )
    parser.add_argument(
        "--output",
        default="outputs/official_split_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ignored = load_ignored_sample_ids(args.ignore_file)
    report = discover_sen12mscr(
        args.data_root,
        ignored_sample_ids=ignored,
        strict_counterparts=True,
    )
    partitions = partition_samples_by_uncrtaints_split(report.samples)

    split_summary = {}
    for split, samples in partitions.items():
        ids = [sample.sample_id for sample in samples]
        groups = sorted({sample_group_key(sample) for sample in samples})
        split_summary[split] = {
            "num_samples": len(ids),
            "num_roi_groups": len(groups),
            "sample_ids_sha256": _sha256(ids),
        }

    summary = {
        "protocol": "uncrtaints_sen12mscr_roi_split",
        "source_repo": UNCRTAINTS_SOURCE_REPO,
        "source_commit": UNCRTAINTS_SOURCE_COMMIT,
        "dataset_num_samples_after_ignored": len(report.samples),
        "dataset_sample_ids_sha256": _sha256(
            [sample.sample_id for sample in report.samples]
        ),
        "reference_roi_counts": {
            split: len(groups)
            for split, groups in UNCRTAINTS_SPLIT_GROUPS.items()
        },
        "splits": split_summary,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
