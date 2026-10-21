"""Audit the source-guided DB-CR architecture without starting training.

Usage:
  python -m mscloudr.cli.audit_source_guided_dbcr
  python -m mscloudr.cli.audit_source_guided_dbcr --time-sharing per_width
  python -m mscloudr.cli.audit_source_guided_dbcr --smoke-forward
"""

from __future__ import annotations

import argparse
import json

from mscloudr.models.source_guided_dbcr import (
    SourceGuidedDBCRConfig,
    SourceGuidedDBCRNet,
)
from mscloudr.models.patent_blocks import PatentTimeEmbeddingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time-sharing",
        choices=("per_block", "per_width"),
        default="per_block",
        help="Non-source-resolved assumption; separate identity from legacy model.",
    )
    parser.add_argument("--positional-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--smoke-forward",
        action="store_true",
        help="Run a CPU-only 16x16 inference shape/finiteness diagnostic.",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    cfg = SourceGuidedDBCRConfig(
        time_sharing=args.time_sharing,
        time_embedding=PatentTimeEmbeddingConfig(
            positional_dim=args.positional_dim,
            hidden_dim=args.hidden_dim,
        ),
    )
    model = SourceGuidedDBCRNet(cfg)
    audit = model.architecture_audit()
    if args.smoke_forward:
        import torch

        model.eval()
        with torch.inference_mode():
            y = model(
                torch.zeros(1, cfg.in_opt, 16, 16),
                torch.tensor([float(cfg.total_steps)]),
                torch.zeros(1, cfg.in_sar, 16, 16),
            )
        audit["smoke_forward"] = {
            "input_shape": [1, cfg.in_opt, 16, 16],
            "output_shape": list(y.shape),
            "finite": bool(torch.isfinite(y).all()),
        }
        if not audit["smoke_forward"]["finite"]:
            raise RuntimeError("source-guided DB-CR forward produced nonfinite values")
    return audit


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
