"""Audit parameter-count implications of the DB-CR patent disclosure.

This command does not claim a unique recovered parameter count because the
patent does not specify time-embedding widths or sharing policy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from mscloudr.models.blocks import NAFBlock, count_trainable_parameters
from mscloudr.models.dbcr import (
    AuditedDBCRNet,
    PUBLISHED_PARAMETER_COUNT,
)
from mscloudr.models.patent_blocks import (
    PatentTimeEmbeddedNAFBlock,
    PatentTimeEmbedding,
    PatentTimeEmbeddingConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positional-dim",
        type=int,
        default=128,
        help=(
            "Patent does not specify this width; 128 is retained only "
            "for continuity with the historical time_dim."
        ),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Patent does not specify this hidden width.",
    )
    return parser.parse_args()


def _block_counts(model: AuditedDBCRNet) -> Counter[int]:
    cfg = model.config
    counts: Counter[int] = Counter()

    for width, num_blocks in zip(
        cfg.widths,
        cfg.encoder_blocks,
    ):
        counts[width] += 2 * num_blocks

    for width, num_blocks in zip(
        reversed(cfg.widths),
        cfg.decoder_blocks,
    ):
        counts[width] += num_blocks

    return counts


def main() -> None:
    args = parse_args()
    time_cfg = PatentTimeEmbeddingConfig(
        positional_dim=args.positional_dim,
        hidden_dim=args.hidden_dim,
    )

    current = AuditedDBCRNet()
    current_total = count_trainable_parameters(current)
    counts = _block_counts(current)

    current_naf_parameters = sum(
        num_blocks
        * count_trainable_parameters(
            NAFBlock(width)
        )
        for width, num_blocks in counts.items()
    )
    legacy_time_parameters = (
        count_trainable_parameters(
            current.time_conditioner
        )
        if current.time_conditioner is not None
        else 0
    )
    non_naf_non_time_parameters = (
        current_total
        - current_naf_parameters
        - legacy_time_parameters
    )

    patent_base_block_parameters = sum(
        num_blocks
        * count_trainable_parameters(
            PatentTimeEmbeddedNAFBlock(
                width,
                time_config=time_cfg,
                own_time_embedding=False,
            )
        )
        for width, num_blocks in counts.items()
    )

    independent_time_parameters = sum(
        num_blocks
        * count_trainable_parameters(
            PatentTimeEmbedding(
                width,
                time_cfg,
            )
        )
        for width, num_blocks in counts.items()
    )

    shared_by_width_time_parameters = sum(
        count_trainable_parameters(
            PatentTimeEmbedding(
                width,
                time_cfg,
            )
        )
        for width in sorted(counts)
    )

    independent_total = (
        non_naf_non_time_parameters
        + patent_base_block_parameters
        + independent_time_parameters
    )
    shared_by_width_total = (
        non_naf_non_time_parameters
        + patent_base_block_parameters
        + shared_by_width_time_parameters
    )

    report = {
        "source": "US20260289738A1",
        "important_limitation": (
            "The patent specifies the time-embedding structure but not its "
            "positional/hidden widths or sharing policy. Scenario totals are "
            "therefore diagnostics, not recovered ground truth."
        ),
        "assumed_time_dimensions": {
            "positional_dim": args.positional_dim,
            "hidden_dim": args.hidden_dim,
        },
        "nafblock_counts_by_width": {
            str(width): counts[width]
            for width in sorted(counts)
        },
        "current_audited_scaffold": {
            "parameters": current_total,
            "legacy_time_parameters": legacy_time_parameters,
            "current_naf_parameters": current_naf_parameters,
            "other_parameters": non_naf_non_time_parameters,
        },
        "patent_disclosed_base_blocks_without_time_mlp": {
            "parameters": patent_base_block_parameters,
        },
        "scenario_independent_time_mlp_per_nafblock": {
            "time_parameters": independent_time_parameters,
            "total_parameters": independent_total,
            "gap_vs_published": (
                PUBLISHED_PARAMETER_COUNT
                - independent_total
            ),
        },
        "scenario_one_time_mlp_per_unique_width": {
            "time_parameters": shared_by_width_time_parameters,
            "total_parameters": shared_by_width_total,
            "gap_vs_published": (
                PUBLISHED_PARAMETER_COUNT
                - shared_by_width_total
            ),
        },
        "published_dbcr_parameters": PUBLISHED_PARAMETER_COUNT,
    }

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
