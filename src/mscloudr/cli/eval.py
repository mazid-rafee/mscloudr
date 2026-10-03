"""Paper-facing evaluation command for trained legacy DB-CR checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from mscloudr.checkpointing import CHECKPOINT_FORMAT_VERSION
from mscloudr.data import (
    REFERENCE_PROTOCOL,
    build_reference_dataloaders,
    build_reference_datasets,
    discover_sen12mscr,
    load_ignored_sample_ids,
    reference_split_audit,
)
from mscloudr.evaluation import (
    EvaluationProgress,
    EvaluationResult,
    evaluate_nfe1_endpoint,
)
from mscloudr.metrics import REFERENCE_COMMIT, REFERENCE_REPO
from mscloudr.models import (
    LEGACY_DBCR_PARAMETER_COUNT,
    LegacyDBCRNet,
    count_legacy_parameters,
)
from mscloudr.reproducibility import seed_everything


DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4
DEFAULT_PROGRESS_EVERY = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a legacy DB-CR checkpoint on the frozen SEN12MS-CR "
            "test split using deterministic NFE=1 endpoint inference."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Training checkpoint to evaluate. Paper tables should use "
            "checkpoints/best_endpoint.pt."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "SEN12MS-CR root or parent. If omitted, MSCLOUDR_DATA_ROOT "
            "must be set."
        ),
    )
    parser.add_argument(
        "--ignore-file",
        default="manifests/known_invalid_samples.txt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Defaults to <run-dir>/eval_metrics.json."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device such as cuda, cuda:0, or cpu. Defaults to cuda "
            "when available."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=(
            "Print running test metrics every N batches. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Diagnostic-only truncation. Any value here marks the result as "
            "non-paper-grade."
        ),
    )
    parser.add_argument(
        "--allow-smoke-checkpoint",
        action="store_true",
        help=(
            "Allow evaluation of a checkpoint whose run metadata is marked "
            "paper_grade=false."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing eval_metrics.json output.",
    )
    return parser


def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    device = torch.device(requested)
    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA device requested but CUDA is not available"
        )
    return device


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive"
        )
    if args.num_workers < 0:
        raise ValueError(
            "num-workers must be non-negative"
        )
    if args.progress_every < 0:
        raise ValueError(
            "progress-every must be non-negative"
        )
    if (
        args.max_batches is not None
        and args.max_batches <= 0
    ):
        raise ValueError(
            "max-batches must be positive when provided"
        )


def _torch_load_payload(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}"
        )
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )
    if not isinstance(payload, dict):
        raise ValueError(
            "checkpoint payload must be a dictionary"
        )
    return payload


def _validate_checkpoint(
    payload: dict[str, Any],
    *,
    allow_smoke_checkpoint: bool,
) -> dict[str, Any]:
    if (
        payload.get("format_version")
        != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported checkpoint format version"
        )
    if "model_state" not in payload:
        raise ValueError(
            "checkpoint is missing model_state"
        )

    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            "checkpoint is missing run_metadata"
        )
    if metadata.get("model_identity") != "legacy_dbcr":
        raise ValueError(
            "evaluation currently supports only model_identity=legacy_dbcr"
        )
    if metadata.get("split_protocol") != REFERENCE_PROTOCOL:
        raise ValueError(
            "checkpoint was not trained with the frozen reference split protocol"
        )
    if metadata.get("schedule_name") not in {
        "original",
        "mr_r3",
    }:
        raise ValueError(
            "unsupported controlled schedule identity in checkpoint metadata"
        )
    if (
        metadata.get("paper_grade") is False
        and not allow_smoke_checkpoint
    ):
        raise ValueError(
            "checkpoint is marked non-paper-grade; pass "
            "--allow-smoke-checkpoint only for diagnostics"
        )

    total_steps = int(
        metadata.get("total_steps", 0)
    )
    if total_steps <= 0:
        raise ValueError(
            "checkpoint metadata has invalid total_steps"
        )

    return metadata


def _train_seed_from_metadata(
    metadata: dict[str, Any],
) -> int:
    try:
        return int(
            metadata["reproducibility"]
            ["seeds"]
            ["train_seed"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "checkpoint metadata is missing reproducibility.seeds.train_seed"
        ) from error


def _deterministic_algorithms_from_metadata(
    metadata: dict[str, Any],
) -> bool:
    reproducibility = metadata.get(
        "reproducibility",
        {},
    )
    if not isinstance(reproducibility, dict):
        return False
    return bool(
        reproducibility.get(
            "deterministic_algorithms",
            False,
        )
    )


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"evaluation output already exists: {path}. "
            "Use --overwrite to replace it."
        )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(
        temporary,
        path,
    )


def _progress(progress: EvaluationProgress) -> None:
    l1 = progress.metrics.get("L1")
    psnr = progress.metrics.get("PSNR")
    l1_text = (
        "nan" if l1 is None else f"{l1:.6f}"
    )
    psnr_text = (
        "nan" if psnr is None else f"{psnr:.4f}"
    )
    print(
        "eval batch={batch}/{total} samples={samples} "
        "L1={l1} PSNR={psnr}".format(
            batch=progress.batch,
            total=progress.total_batches,
            samples=progress.num_samples,
            l1=l1_text,
            psnr=psnr_text,
        ),
        flush=True,
    )


def _default_output_path(
    checkpoint_path: Path,
) -> Path:
    # Expected layout: <run-dir>/checkpoints/<name>.pt
    if checkpoint_path.parent.name == "checkpoints":
        return (
            checkpoint_path.parent.parent
            / "eval_metrics.json"
        )
    return checkpoint_path.with_name(
        checkpoint_path.stem
        + "_eval_metrics.json"
    )


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    validate_cli_args(args)

    checkpoint_path = Path(
        args.checkpoint
    )
    payload = _torch_load_payload(
        checkpoint_path
    )
    metadata = _validate_checkpoint(
        payload,
        allow_smoke_checkpoint=(
            args.allow_smoke_checkpoint
        ),
    )

    device = resolve_device(
        args.device
    )
    train_seed = _train_seed_from_metadata(
        metadata
    )
    seed_everything(
        train_seed,
        deterministic_algorithms=(
            _deterministic_algorithms_from_metadata(
                metadata
            )
        ),
    )

    ignored = load_ignored_sample_ids(
        args.ignore_file
    )
    discovery = discover_sen12mscr(
        args.data_root,
        ignored_sample_ids=ignored,
        strict_counterparts=True,
    )
    datasets = build_reference_datasets(
        discovery.samples,
        include_sample_id=True,
        strict_channels=True,
        verify_frozen_membership=True,
    )
    split_audit = reference_split_audit(
        {
            split: dataset.samples
            for split, dataset
            in datasets.as_dict().items()
        }
    )

    checkpoint_split_audit = metadata.get(
        "split_audit"
    )
    if checkpoint_split_audit != split_audit:
        raise ValueError(
            "current dataset split audit does not match checkpoint provenance"
        )

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=train_seed,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = LegacyDBCRNet()
    trainable_parameters = (
        count_legacy_parameters(model)
    )
    if (
        trainable_parameters
        != LEGACY_DBCR_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "legacy_dbcr parameter count drifted before evaluation"
        )
    model.load_state_dict(
        payload["model_state"],
        strict=True,
    )

    paper_grade = bool(
        metadata.get("paper_grade", False)
    ) and args.max_batches is None

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(
                    payload.get("epoch", 0)
                ),
                "device": str(device),
                "model_identity": "legacy_dbcr",
                "schedule_name": metadata[
                    "schedule_name"
                ],
                "split": "test",
                "test_samples": len(
                    datasets.test
                ),
                "nfe": 1,
                "inference": "endpoint_direct_x0",
                "paper_grade": paper_grade,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    result: EvaluationResult = (
        evaluate_nfe1_endpoint(
            model,
            loaders.test,
            total_steps=int(
                metadata["total_steps"]
            ),
            device=device,
            max_batches=args.max_batches,
            progress_every=(
                args.progress_every
                if args.progress_every > 0
                else None
            ),
            progress_callback=(
                _progress
                if args.progress_every > 0
                else None
            ),
        )
    )

    output_payload: dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_epoch": int(
            payload.get("epoch", 0)
        ),
        "checkpoint_metrics": payload.get(
            "metrics",
            {},
        ),
        "checkpoint_selection_metric": metadata.get(
            "checkpoint_selection_metric"
        ),
        "canonical_checkpoint_filename": (
            checkpoint_path.name
            == "best_endpoint.pt"
        ),
        "model_identity": metadata[
            "model_identity"
        ],
        "schedule_name": metadata[
            "schedule_name"
        ],
        "total_steps": int(
            metadata["total_steps"]
        ),
        "nfe": 1,
        "inference": "endpoint_direct_x0",
        "schedule_used_during_inference": False,
        "split": "test",
        "split_protocol": REFERENCE_PROTOCOL,
        "split_audit": split_audit,
        "num_samples": result.num_samples,
        "num_batches": result.num_batches,
        "batch_size": int(
            args.batch_size
        ),
        "max_batches": args.max_batches,
        "paper_grade": paper_grade,
        "metrics": result.metrics,
        "metric_counts": result.metric_counts,
        "metric_reference_repo": REFERENCE_REPO,
        "metric_reference_commit": REFERENCE_COMMIT,
        "prediction_clipping": False,
    }

    output_path = (
        Path(args.output)
        if args.output is not None
        else _default_output_path(
            checkpoint_path
        )
    )
    _write_json_atomic(
        output_path,
        output_payload,
        overwrite=args.overwrite,
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "num_samples": (
                    result.num_samples
                ),
                "metrics": result.metrics,
                "paper_grade": paper_grade,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    return output_payload


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
