"""Paper-grade training entry point for the controlled DB-CR baseline.

This command intentionally exposes only the experiment identities that have
already been audited:

- model: legacy_dbcr
- schedules: original and mr_r3
- data split: fixed UnCRtainTS ROI split

It validates the full local dataset fingerprint before training starts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from mscloudr.bridge import get_bridge_schedule
from mscloudr.data import (
    REFERENCE_PROTOCOL,
    build_reference_dataloaders,
    build_reference_datasets,
    discover_sen12mscr,
    load_ignored_sample_ids,
    reference_split_audit,
)
from mscloudr.models import (
    LEGACY_DBCR_PARAMETER_COUNT,
    LegacyDBCRNet,
    count_legacy_parameters,
)
from mscloudr.reproducibility import (
    SeedConfig,
    make_torch_generator,
    reproducibility_metadata,
    seed_everything,
)
from mscloudr.runner import (
    HISTORICAL_LEARNING_RATE,
    BatchProgress,
    EpochMetrics,
    fit,
)
from mscloudr.training import (
    CHECKPOINT_SELECTION_METRIC,
    DIAGNOSTIC_RANDOM_T_METRIC,
)


DEFAULT_TOTAL_STEPS = 1000
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4
DEFAULT_TRAIN_SEED = 42
DEFAULT_SAMPLER_SEED = 42
DEFAULT_SMOKE_TRAIN_BATCHES = 2
DEFAULT_SMOKE_VAL_BATCHES = 2
DEFAULT_PROGRESS_EVERY = 500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen legacy DB-CR backbone on the verified "
            "SEN12MS-CR ROI split."
        )
    )
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
        help=(
            "Dataset-relative invalid-sample list. The verified protocol "
            "expects the repository default."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help=(
            "Single directory name under output-root. Existing non-resume "
            "runs are rejected instead of silently overwritten."
        ),
    )
    parser.add_argument(
        "--schedule",
        choices=("original", "mr_r3"),
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
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
        "--lr",
        type=float,
        default=HISTORICAL_LEARNING_RATE,
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=DEFAULT_TRAIN_SEED,
        help=(
            "Controls model initialization, global training RNG, and "
            "DataLoader shuffle/worker RNG."
        ),
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=DEFAULT_SAMPLER_SEED,
        help="Controls bridge timestep sampling only.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device such as cuda, cuda:0, or cpu. Defaults to cuda "
            "when available, otherwise cpu."
        ),
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help=(
            "Request torch deterministic algorithms in addition to deterministic "
            "cuDNN settings. Unsupported operations may raise."
        ),
    )
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help=(
            "Run exactly one truncated train/validation epoch for end-to-end "
            "plumbing verification. Smoke outputs are marked non-paper-grade."
        ),
    )
    parser.add_argument(
        "--smoke-train-batches",
        type=int,
        default=DEFAULT_SMOKE_TRAIN_BATCHES,
        help="Number of training batches used when --smoke-run is enabled.",
    )
    parser.add_argument(
        "--smoke-val-batches",
        type=int,
        default=DEFAULT_SMOKE_VAL_BATCHES,
        help="Number of validation batches used when --smoke-run is enabled.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=(
            "Print within-epoch progress every N processed batches. "
            "Use 0 to disable batch progress logging."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Epoch-boundary checkpoint to resume. Use the same run-name/output "
            "directory that created the checkpoint."
        ),
    )
    return parser


def resolve_device(
    requested: str | None,
) -> torch.device:
    if requested is None:
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        requested
    )
    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA device requested but CUDA is not available"
        )
    return device


def canonical_schedule(
    name: str,
):
    if name == "original":
        return (
            "original",
            get_bridge_schedule(
                "original"
            ),
            None,
        )
    if name == "mr_r3":
        return (
            "mr_r3",
            get_bridge_schedule(
                "mr_r3",
                mean_reversion_rate=3.0,
            ),
            3.0,
        )
    raise ValueError(
        f"unsupported controlled schedule: {name!r}"
    )


def validate_cli_args(
    args: argparse.Namespace,
) -> None:
    if (
        Path(args.run_name).name
        != args.run_name
        or args.run_name in {".", ".."}
    ):
        raise ValueError(
            "run-name must be a single directory name"
        )
    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )
    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive"
        )
    if args.num_workers < 0:
        raise ValueError(
            "num-workers must be non-negative"
        )
    if args.lr <= 0.0:
        raise ValueError(
            "lr must be positive"
        )
    if args.total_steps <= 0:
        raise ValueError(
            "total-steps must be positive"
        )
    if args.train_seed < 0:
        raise ValueError(
            "train-seed must be non-negative"
        )
    if args.sampler_seed < 0:
        raise ValueError(
            "sampler-seed must be non-negative"
        )
    if args.smoke_train_batches <= 0:
        raise ValueError(
            "smoke-train-batches must be positive"
        )
    if args.smoke_val_batches <= 0:
        raise ValueError(
            "smoke-val-batches must be positive"
        )
    if args.progress_every < 0:
        raise ValueError(
            "progress-every must be non-negative"
        )
    if args.smoke_run:
        if args.epochs != 1:
            raise ValueError(
                "smoke-run requires --epochs 1"
            )
        if args.resume is not None:
            raise ValueError(
                "smoke-run does not support --resume"
            )


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
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


def _prepare_output_dir(
    args: argparse.Namespace,
) -> Path:
    run_dir = (
        Path(args.output_root)
        / args.run_name
    )

    if args.resume is None:
        if (
            run_dir.exists()
            and any(run_dir.iterdir())
        ):
            raise FileExistsError(
                f"run directory is not empty: {run_dir}. "
                "Choose a new --run-name or use --resume."
            )
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    elif not run_dir.is_dir():
        raise FileNotFoundError(
            f"resume run directory does not exist: {run_dir}"
        )

    return run_dir


def _run_metadata(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    split_audit: dict[str, Any],
    trainable_parameters: int,
    schedule_name: str,
    mean_reversion_rate: float | None,
    device: torch.device,
) -> dict[str, Any]:
    # The official ROI split is fixed and consumes no RNG. SeedConfig currently
    # uses integer fields, so its split entry is explicitly replaced by null in
    # the persisted paper protocol.
    seed_metadata = reproducibility_metadata(
        SeedConfig(
            split_seed=0,
            train_seed=args.train_seed,
            sampler_seed=args.sampler_seed,
        ),
        deterministic_algorithms=args.deterministic_algorithms,
    )
    seed_metadata["seeds"][
        "split_seed"
    ] = None

    return {
        "format_version": 1,
        "model_identity": "legacy_dbcr",
        "trainable_parameters": (
            trainable_parameters
        ),
        "expected_trainable_parameters": (
            LEGACY_DBCR_PARAMETER_COUNT
        ),
        "conditioning_identity": (
            "legacy_raw_t_stage_bias"
        ),
        "schedule_name": schedule_name,
        "mean_reversion_rate": (
            mean_reversion_rate
        ),
        "total_steps": args.total_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.lr,
        "checkpoint_selection_metric": (
            CHECKPOINT_SELECTION_METRIC
        ),
        "diagnostic_random_t_metric": (
            DIAGNOSTIC_RANDOM_T_METRIC
        ),
        "split_protocol": (
            REFERENCE_PROTOCOL
        ),
        "split_seed": None,
        "split_seed_note": (
            "fixed ROI membership; no split RNG is consumed"
        ),
        "split_audit": split_audit,
        "data_root": str(
            dataset_root
        ),
        "ignore_file": str(
            args.ignore_file
        ),
        "device": str(device),
        "run_kind": (
            "smoke" if args.smoke_run else "full"
        ),
        "paper_grade": (
            not args.smoke_run
        ),
        "max_train_batches": (
            args.smoke_train_batches
            if args.smoke_run
            else None
        ),
        "max_val_batches": (
            args.smoke_val_batches
            if args.smoke_run
            else None
        ),
        "progress_every": (
            args.progress_every
        ),
        "reproducibility": seed_metadata,
    }


def _resume_identity(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_identity": metadata[
            "model_identity"
        ],
        "trainable_parameters": metadata[
            "trainable_parameters"
        ],
        "conditioning_identity": metadata[
            "conditioning_identity"
        ],
        "schedule_name": metadata[
            "schedule_name"
        ],
        "mean_reversion_rate": metadata[
            "mean_reversion_rate"
        ],
        "total_steps": metadata[
            "total_steps"
        ],
        "batch_size": metadata[
            "batch_size"
        ],
        "num_workers": metadata[
            "num_workers"
        ],
        "learning_rate": metadata[
            "learning_rate"
        ],
        "split_protocol": metadata[
            "split_protocol"
        ],
        "split_audit": metadata[
            "split_audit"
        ],
        "train_seed": metadata[
            "reproducibility"
        ]["seeds"]["train_seed"],
        "sampler_seed": metadata[
            "reproducibility"
        ]["seeds"]["sampler_seed"],
        "deterministic_algorithms": metadata[
            "reproducibility"
        ]["deterministic_algorithms"],
    }


def validate_resume_metadata(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Reject a resume if experiment-defining settings changed."""

    previous_identity = _resume_identity(
        previous
    )
    current_identity = _resume_identity(
        current
    )
    if (
        previous_identity
        != current_identity
    ):
        mismatches = [
            key
            for key in previous_identity
            if previous_identity[key]
            != current_identity[key]
        ]
        raise ValueError(
            "resume configuration mismatch for: "
            + ", ".join(mismatches)
        )


def _epoch_progress(
    metrics: EpochMetrics,
) -> None:
    print(
        "epoch={epoch} train_l1={train:.6f} "
        "val_random_t_l1={random:.6f} "
        "val_endpoint_l1={endpoint:.6f}".format(
            epoch=metrics.epoch,
            train=metrics.train_l1,
            random=(
                metrics.val_random_t_l1
            ),
            endpoint=(
                metrics.val_endpoint_l1
            ),
        ),
        flush=True,
    )


def _batch_progress(
    progress: BatchProgress,
) -> None:
    """Print compact, grep-friendly within-epoch progress."""

    if progress.phase == "train":
        print(
            "epoch={epoch} phase=train batch={batch}/{total} "
            "train_l1_running={l1:.6f}".format(
                epoch=progress.epoch,
                batch=progress.batch,
                total=progress.total_batches,
                l1=progress.metrics[
                    "train_l1_running"
                ],
            ),
            flush=True,
        )
        return

    if progress.phase == "val":
        print(
            "epoch={epoch} phase=val batch={batch}/{total} "
            "val_random_t_l1_running={random:.6f} "
            "val_endpoint_l1_running={endpoint:.6f}".format(
                epoch=progress.epoch,
                batch=progress.batch,
                total=progress.total_batches,
                random=progress.metrics[
                    "val_random_t_l1_running"
                ],
                endpoint=progress.metrics[
                    "val_endpoint_l1_running"
                ],
            ),
            flush=True,
        )
        return

    raise ValueError(
        f"unknown progress phase: {progress.phase!r}"
    )


def run(
    args: argparse.Namespace,
) -> list[EpochMetrics]:
    validate_cli_args(
        args
    )
    run_dir = _prepare_output_dir(
        args
    )
    device = resolve_device(
        args.device
    )
    (
        schedule_name,
        schedule,
        mr_rate,
    ) = canonical_schedule(
        args.schedule
    )

    # Seed before model construction and DataLoader creation.
    seed_everything(
        args.train_seed,
        deterministic_algorithms=(
            args.deterministic_algorithms
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

    # Full paper runs must match the exact locally audited 122,217-sample
    # installation after the documented invalid triplet is excluded.
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

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=args.train_seed,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = LegacyDBCRNet()
    trainable_parameters = (
        count_legacy_parameters(
            model
        )
    )
    if (
        trainable_parameters
        != LEGACY_DBCR_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "legacy_dbcr parameter count drifted: "
            f"expected {LEGACY_DBCR_PARAMETER_COUNT}, "
            f"got {trainable_parameters}"
        )

    metadata = _run_metadata(
        args=args,
        dataset_root=(
            discovery.dataset_root
        ),
        split_audit=split_audit,
        trainable_parameters=(
            trainable_parameters
        ),
        schedule_name=schedule_name,
        mean_reversion_rate=mr_rate,
        device=device,
    )

    config_path = (
        run_dir
        / "run_config.json"
    )
    if args.resume is None:
        _write_json_atomic(
            config_path,
            metadata,
        )
    else:
        if not config_path.is_file():
            raise FileNotFoundError(
                "resume requires existing "
                f"run_config.json: {config_path}"
            )
        previous_metadata = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )
        validate_resume_metadata(
            previous_metadata,
            metadata,
        )
        # epochs and device may intentionally change on resume; persist the
        # current target/runtime after invariant settings have been checked.
        _write_json_atomic(
            config_path,
            metadata,
        )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "device": str(device),
                "model_identity": (
                    "legacy_dbcr"
                ),
                "trainable_parameters": (
                    trainable_parameters
                ),
                "schedule_name": (
                    schedule_name
                ),
                "train_samples": (
                    len(datasets.train)
                ),
                "val_samples": (
                    len(datasets.val)
                ),
                "test_samples": (
                    len(datasets.test)
                ),
                "checkpoint_selection_metric": (
                    CHECKPOINT_SELECTION_METRIC
                ),
                "run_kind": (
                    "smoke"
                    if args.smoke_run
                    else "full"
                ),
                "progress_every": (
                    args.progress_every
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    sampler_generator = (
        make_torch_generator(
            args.sampler_seed,
            device="cpu",
        )
    )

    history = fit(
        model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        schedule=schedule,
        schedule_name=schedule_name,
        total_steps=args.total_steps,
        sampler_generator=(
            sampler_generator
        ),
        output_dir=run_dir,
        epochs=args.epochs,
        device=device,
        model_identity="legacy_dbcr",
        lr=args.lr,
        run_metadata=metadata,
        resume_from=args.resume,
        epoch_callback=_epoch_progress,
        batch_progress_callback=(
            _batch_progress
            if args.progress_every > 0
            else None
        ),
        progress_every=(
            args.progress_every
            if args.progress_every > 0
            else None
        ),
        max_train_batches=(
            args.smoke_train_batches
            if args.smoke_run
            else None
        ),
        max_val_batches=(
            args.smoke_val_batches
            if args.smoke_run
            else None
        ),
    )

    return history


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )
    history = run(
        args
    )
    last = history[-1]
    print(
        json.dumps(
            {
                "completed_epoch": (
                    last.epoch
                ),
                "train_l1": (
                    last.train_l1
                ),
                "val_random_t_l1": (
                    last.val_random_t_l1
                ),
                "val_endpoint_l1": (
                    last.val_endpoint_l1
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
