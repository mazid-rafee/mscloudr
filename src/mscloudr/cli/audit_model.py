"""Print the current DB-CR architecture parameter audit."""

from __future__ import annotations

import argparse
import json

from mscloudr.models.dbcr import AuditedDBCRNet, DBCRArchitectureConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-downsamplers", action="store_true")
    parser.add_argument("--no-time-conditioning", action="store_true")
    parser.add_argument(
        "--output-kernel-size",
        type=int,
        choices=(1, 3),
        default=3,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DBCRArchitectureConfig(
        separate_modality_downsamplers=not args.shared_downsamplers,
        time_conditioning=(
            "none" if args.no_time_conditioning else "legacy_stage_bias"
        ),
        output_kernel_size=args.output_kernel_size,
    )
    audit = AuditedDBCRNet(cfg).architecture_audit()
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
