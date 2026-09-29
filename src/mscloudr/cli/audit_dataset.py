"""Audit an external SEN12MS-CR installation without copying the dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mscloudr.data.sen12mscr import (
    discover_sen12mscr,
    load_ignored_sample_ids,
)


def _ids_sha256(sample_ids: list[str]) -> str:
    payload = "\n".join(sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "SEN12MS-CR root or its parent. If omitted, MSCLOUDR_DATA_ROOT "
            "must be set."
        ),
    )
    parser.add_argument(
        "--ignore-file",
        default="manifests/known_invalid_samples.txt",
        help="Text file containing dataset-relative cloudy sample IDs to exclude.",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        default=None,
        help="Optional subset: winter spring summer fall.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON summary path. No imagery is copied.",
    )
    parser.add_argument(
        "--write-sample-ids",
        default=None,
        help="Optional text file containing all discovered stable sample IDs.",
    )
    parser.add_argument(
        "--allow-missing-counterparts",
        action="store_true",
        help="Report rather than fail on missing aligned S1/S2 files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ignored = load_ignored_sample_ids(args.ignore_file)

    report = discover_sen12mscr(
        args.data_root,
        seasons=args.seasons,
        ignored_sample_ids=ignored,
        strict_counterparts=not args.allow_missing_counterparts,
    )
    sample_ids = [sample.sample_id for sample in report.samples]

    summary = {
        "dataset_root": str(report.dataset_root),
        "num_samples": len(sample_ids),
        "counts_by_season": report.counts_by_season(),
        "num_missing_counterparts": len(report.missing_counterparts),
        "num_ignored": len(report.ignored_sample_ids),
        "sample_ids_sha256": _ids_sha256(sample_ids),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.write_sample_ids:
        path = Path(args.write_sample_ids)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sample_ids) + "\n")


if __name__ == "__main__":
    main()
