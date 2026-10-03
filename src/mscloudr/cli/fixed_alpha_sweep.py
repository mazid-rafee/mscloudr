"""Fixed-physical-alpha diagnostic for DB-CR checkpoints.

This diagnostic evaluates a trained checkpoint at the same physical bridge
states x_alpha = (1-alpha) * clean + alpha * cloudy across a user-specified
alpha grid. The bridge state is therefore schedule-independent. The model's
time-conditioning value is recovered by inverting the schedule that was used
for training, then rounding to the nearest discrete training timestep.

Both audited full-data checkpoints and fixed 10% pilot checkpoints are
supported. The default split is validation, not test, because this command is
intended for method exploration rather than final reporting.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from mscloudr.bridge import get_bridge_schedule, make_bridge_state
from mscloudr.cli import eval as eval_base
from mscloudr.cli import pilot_eval
from mscloudr.data import (
    REFERENCE_PROTOCOL,
    build_reference_dataloaders,
    build_reference_datasets,
    discover_sen12mscr,
    load_ignored_sample_ids,
    reference_split_audit,
)
from mscloudr.data.pilot import (
    PILOT_PROTOCOL,
    build_pilot_datasets,
    pilot_split_audit,
)
from mscloudr.metrics import REFERENCE_COMMIT, REFERENCE_REPO, ReferenceMetricAccumulator
from mscloudr.models import (
    LEGACY_DBCR_PARAMETER_COUNT,
    LegacyDBCRNet,
    count_legacy_parameters,
)
from mscloudr.reproducibility import seed_everything


DEFAULT_ALPHAS = tuple(i / 10.0 for i in range(11))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an audited full-data or pilot DB-CR checkpoint at fixed "
            "physical bridge corruption levels alpha."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--ignore-file",
        default="manifests/known_invalid_samples.txt",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help=(
            "Dataset split for the diagnostic. Validation is the default so "
            "the held-out test split is not repeatedly used for method design."
        ),
    )
    parser.add_argument(
        "--alphas",
        default=",".join(f"{value:.1f}" for value in DEFAULT_ALPHAS),
        help="Comma-separated physical alpha values in [0,1].",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N batches for each alpha; 0 disables.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional diagnostic truncation for quick smoke checks.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_alphas(text: str) -> list[float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("at least one alpha is required")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("all alpha values must lie in [0,1]")
    return values


def normalized_time_from_alpha(alpha: float, schedule_name: str) -> float:
    """Invert the training parameterization to obtain s=t/T."""

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")

    if schedule_name == "canonical_alpha":
        return alpha

    if schedule_name == "original":
        return (2.0 / math.pi) * math.asin(alpha)

    if schedule_name == "mr_r3":
        rate = 3.0
        one_minus_exp = 1.0 - math.exp(-rate)
        return -math.log(1.0 - one_minus_exp * alpha) / rate

    raise ValueError(f"unsupported schedule: {schedule_name!r}")


def conditioning_for_alpha(
    alpha: float,
    *,
    schedule_name: str,
    total_steps: int,
) -> dict[str, float | int]:
    """Return continuous/rounded timestep and realized conditioning alpha."""

    s = normalized_time_from_alpha(alpha, schedule_name)
    t_continuous = s * float(total_steps)
    t_rounded = int(round(t_continuous))
    t_rounded = max(0, min(int(total_steps), t_rounded))

    schedule = get_bridge_schedule(
        schedule_name,
        mean_reversion_rate=3.0,
    )
    realized = float(
        schedule(
            torch.tensor([float(t_rounded)], dtype=torch.float32),
            total_steps,
        )[0].item()
    )
    return {
        "normalized_time": float(s),
        "t_continuous": float(t_continuous),
        "t_rounded": int(t_rounded),
        "conditioning_alpha": realized,
        "conditioning_alpha_error": realized - float(alpha),
    }


def _default_output_path(checkpoint: Path, split: str) -> Path:
    filename = f"fixed_alpha_sweep_{split}.json"
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent / filename
    return checkpoint.with_name(checkpoint.stem + f"_{filename}")


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = (
            value.to(device=device, non_blocking=True)
            if torch.is_tensor(value)
            else value
        )
    return moved


def _metrics_with_l1(accumulator: ReferenceMetricAccumulator) -> tuple[dict, dict]:
    metrics = accumulator.compute()
    counts = accumulator.metric_counts()
    if "MAE" in metrics:
        metrics["L1"] = metrics["MAE"]
        counts["L1"] = counts.get("MAE", 0)
    return metrics, counts


def _validate_and_build_datasets(payload, discovery):
    """Dispatch to the dataset protocol recorded in checkpoint provenance."""

    raw_metadata = payload.get("run_metadata")
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint is missing run_metadata")
    protocol = raw_metadata.get("split_protocol")

    if protocol == PILOT_PROTOCOL:
        metadata = pilot_eval._validate_pilot_checkpoint(payload)
        datasets, pilot_manifest = build_pilot_datasets(
            discovery.samples,
            include_sample_id=True,
            strict_channels=True,
            verify_frozen_membership=True,
        )
        split_audit = pilot_split_audit(pilot_manifest)
        if metadata.get("split_audit") != split_audit:
            raise ValueError(
                "current pilot split audit does not match checkpoint provenance"
            )
        if (
            metadata.get("pilot_manifest_sample_ids_sha256")
            != pilot_manifest["sample_ids_sha256"]
        ):
            raise ValueError(
                "pilot sample-ID fingerprint does not match checkpoint provenance"
            )
        membership = {
            "kind": "pilot10",
            "sample_ids_sha256": pilot_manifest["sample_ids_sha256"],
        }
        return metadata, datasets, split_audit, membership

    if protocol == REFERENCE_PROTOCOL:
        metadata = eval_base._validate_checkpoint(
            payload,
            allow_smoke_checkpoint=False,
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
                for split, dataset in datasets.as_dict().items()
            }
        )
        if metadata.get("split_audit") != split_audit:
            raise ValueError(
                "current full-data split audit does not match checkpoint provenance"
            )
        membership = {
            "kind": "full_reference",
            "dataset_sample_ids_sha256": split_audit[
                "dataset_sample_ids_sha256"
            ],
        }
        return metadata, datasets, split_audit, membership

    raise ValueError(
        "unsupported checkpoint split_protocol for fixed-alpha sweep: "
        f"{protocol!r}"
    )


def run(args: argparse.Namespace) -> dict:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.progress_every < 0:
        raise ValueError("progress-every must be non-negative")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max-batches must be positive when provided")

    alphas = parse_alphas(args.alphas)
    checkpoint_path = Path(args.checkpoint)
    payload = eval_base._torch_load_payload(checkpoint_path)

    ignored = load_ignored_sample_ids(args.ignore_file)
    discovery = discover_sen12mscr(
        args.data_root,
        ignored_sample_ids=ignored,
        strict_counterparts=True,
    )
    metadata, datasets, split_audit, membership = _validate_and_build_datasets(
        payload,
        discovery,
    )

    device = eval_base.resolve_device(args.device)
    train_seed = eval_base._train_seed_from_metadata(metadata)
    seed_everything(
        train_seed,
        deterministic_algorithms=eval_base._deterministic_algorithms_from_metadata(
            metadata
        ),
    )

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=train_seed,
        pin_memory=(device.type == "cuda"),
    )
    loader = loaders.as_dict()[args.split]

    model = LegacyDBCRNet()
    if count_legacy_parameters(model) != LEGACY_DBCR_PARAMETER_COUNT:
        raise RuntimeError("legacy_dbcr parameter count drifted")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()

    schedule_name = metadata["schedule_name"]
    total_steps = int(metadata["total_steps"])
    split_size = len(datasets.as_dict()[args.split])
    total_batches = len(loader)
    if args.max_batches is not None:
        total_batches = min(total_batches, int(args.max_batches))

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(payload.get("epoch", 0)),
                "schedule_name": schedule_name,
                "split_protocol": metadata["split_protocol"],
                "dataset_profile": membership["kind"],
                "split": args.split,
                "split_samples": split_size,
                "alphas": alphas,
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    results = []
    with torch.inference_mode():
        for alpha in alphas:
            conditioning = conditioning_for_alpha(
                alpha,
                schedule_name=schedule_name,
                total_steps=total_steps,
            )
            t_value = int(conditioning["t_rounded"])
            accumulator = ReferenceMetricAccumulator()
            num_samples = 0
            num_batches = 0

            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break

                batch = _move_batch(batch, device)
                target = batch["target"]
                cloudy = batch["cloudy"]
                sar = batch["sar"]

                bridge_state = make_bridge_state(
                    target,
                    cloudy,
                    torch.tensor(alpha, device=device, dtype=target.dtype),
                )
                timesteps = torch.full(
                    (target.shape[0],),
                    t_value,
                    device=device,
                    dtype=torch.long,
                )
                prediction = model(bridge_state, timesteps, sar)
                accumulator.update_batch(target, prediction)

                num_samples += int(target.shape[0])
                num_batches += 1
                batch_number = batch_index + 1
                if (
                    args.progress_every > 0
                    and (
                        batch_number % args.progress_every == 0
                        or batch_number == total_batches
                    )
                ):
                    running, _ = _metrics_with_l1(accumulator)
                    print(
                        "alpha={alpha:.2f} t={t} batch={batch}/{total} "
                        "samples={samples} L1={l1:.6f} PSNR={psnr:.4f}".format(
                            alpha=alpha,
                            t=t_value,
                            batch=batch_number,
                            total=total_batches,
                            samples=num_samples,
                            l1=running.get("L1", float("nan")),
                            psnr=running.get("PSNR", float("nan")),
                        ),
                        flush=True,
                    )

            metrics, metric_counts = _metrics_with_l1(accumulator)
            results.append(
                {
                    "requested_alpha": float(alpha),
                    **conditioning,
                    "num_samples": num_samples,
                    "num_batches": num_batches,
                    "metrics": metrics,
                    "metric_counts": metric_counts,
                }
            )

    output_payload = {
        "format_version": 1,
        "diagnostic": "fixed_physical_alpha_sweep",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "model_identity": metadata["model_identity"],
        "schedule_name": schedule_name,
        "total_steps": total_steps,
        "split": args.split,
        "split_protocol": metadata["split_protocol"],
        "split_audit": split_audit,
        "dataset_membership": membership,
        "batch_size": int(args.batch_size),
        "max_batches": args.max_batches,
        "bridge_state_definition": "x_alpha=(1-alpha)*target+alpha*cloudy",
        "time_conditioning": (
            "inverse_training_schedule_then_nearest_integer_timestep"
        ),
        "metric_reference_repo": REFERENCE_REPO,
        "metric_reference_commit": REFERENCE_COMMIT,
        "paper_grade": False,
        "results": results,
    }

    output_path = (
        Path(args.output)
        if args.output is not None
        else _default_output_path(checkpoint_path, args.split)
    )
    eval_base._write_json_atomic(
        output_path,
        output_payload,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output": str(output_path)}, indent=2), flush=True)
    return output_payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
