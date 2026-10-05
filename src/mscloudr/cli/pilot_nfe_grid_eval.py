"""Compare equal-t and equal-alpha reverse grids on pilot checkpoints.

This diagnostic is intentionally restricted to physical-alpha-conditioned,
straight-geometry pilot checkpoints. It isolates inference discretization from
training parameterization by evaluating the same trained model with the same
number of function evaluations (NFE) but different reverse grids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from mscloudr.bridge import deterministic_reverse_step, get_bridge_schedule
from mscloudr.cli import eval as base
from mscloudr.cli.pilot_eval import _validate_pilot_checkpoint
from mscloudr.data import (
    build_reference_dataloaders,
    discover_sen12mscr,
    load_ignored_sample_ids,
)
from mscloudr.data.pilot import PILOT_PROTOCOL, build_pilot_datasets, pilot_split_audit
from mscloudr.metrics import REFERENCE_COMMIT, REFERENCE_REPO, ReferenceMetricAccumulator
from mscloudr.models import LEGACY_DBCR_PARAMETER_COUNT, LegacyDBCRNet, count_legacy_parameters
from mscloudr.reproducibility import seed_everything
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING


SUPPORTED_GRIDS = ("equal_t", "equal_alpha")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one physical-alpha-conditioned pilot checkpoint at multiple "
            "NFEs using equal-t and equal-alpha reverse grids."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ignore-file", default="manifests/known_invalid_samples.txt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--grid",
        choices=SUPPORTED_GRIDS,
        nargs="+",
        default=list(SUPPORTED_GRIDS),
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.progress_every < 0:
        raise ValueError("progress-every must be non-negative")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max-batches must be positive when provided")
    if not args.nfe or any(int(v) <= 0 for v in args.nfe):
        raise ValueError("all NFE values must be positive")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def reverse_alpha_grid(
    *,
    total_steps: int,
    nfe: int,
    grid: str,
    schedule,
    device: torch.device,
) -> torch.Tensor:
    """Return descending alpha values including 1 and 0."""

    total_steps = int(total_steps)
    nfe = int(nfe)
    if total_steps <= 0 or nfe <= 0:
        raise ValueError("total_steps and nfe must be positive")
    if grid == "equal_alpha":
        alpha = torch.linspace(1.0, 0.0, nfe + 1, device=device)
    elif grid == "equal_t":
        t = torch.linspace(float(total_steps), 0.0, nfe + 1, device=device)
        alpha = schedule(t, total_steps).to(device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported reverse grid: {grid!r}")

    alpha = alpha.clone()
    alpha[0] = 1.0
    alpha[-1] = 0.0
    if torch.any(alpha[1:] > alpha[:-1] + 1e-7):
        raise ValueError("reverse alpha grid must be non-increasing")
    return alpha


def _metric_payload(accumulator: ReferenceMetricAccumulator) -> tuple[dict[str, float], dict[str, int]]:
    metrics = dict(accumulator.compute())
    counts = dict(accumulator.metric_counts())
    if "MAE" in metrics:
        metrics["L1"] = metrics["MAE"]
        counts["L1"] = counts.get("MAE", 0)
    return metrics, counts


def evaluate_one(
    model: LegacyDBCRNet,
    loader,
    *,
    schedule,
    total_steps: int,
    nfe: int,
    grid: str,
    device: torch.device,
    max_batches: int | None,
    progress_every: int,
) -> dict[str, Any]:
    alpha_grid = reverse_alpha_grid(
        total_steps=total_steps,
        nfe=nfe,
        grid=grid,
        schedule=schedule,
        device=device,
    )

    model.eval()
    accumulator = ReferenceMetricAccumulator()
    num_samples = 0
    num_batches = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch(batch, device)
            cloudy = batch["cloudy"]
            sar = batch["sar"]
            target = batch["target"]
            x = cloudy

            for step in range(nfe):
                alpha_t = alpha_grid[step]
                alpha_s = alpha_grid[step + 1]
                conditioning = torch.full(
                    (x.shape[0],),
                    float(total_steps) * float(alpha_t.item()),
                    device=device,
                    dtype=torch.float32,
                )
                x0_hat = model(x, conditioning, sar)
                x = deterministic_reverse_step(
                    x,
                    x0_hat,
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                )

            accumulator.update_batch(target, x)
            num_batches += 1
            num_samples += int(target.shape[0])

            if progress_every > 0 and (
                num_batches % progress_every == 0 or num_batches == len(loader)
            ):
                running, _ = _metric_payload(accumulator)
                print(
                    json.dumps(
                        {
                            "grid": grid,
                            "nfe": nfe,
                            "batch": num_batches,
                            "num_samples": num_samples,
                            "metrics": running,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if num_samples == 0:
        raise ValueError("evaluation loader produced no samples")
    metrics, counts = _metric_payload(accumulator)
    return {
        "nfe": int(nfe),
        "grid": grid,
        "alpha_grid": [float(v) for v in alpha_grid.detach().cpu().tolist()],
        "num_samples": num_samples,
        "num_batches": num_batches,
        "metrics": metrics,
        "metric_counts": counts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    checkpoint_path = Path(args.checkpoint)
    payload = base._torch_load_payload(checkpoint_path)
    metadata = _validate_pilot_checkpoint(payload)

    if metadata.get("conditioning_mode") != PHYSICAL_ALPHA_CONDITIONING:
        raise ValueError(
            "NFE grid diagnostic requires conditioning_mode=physical_alpha"
        )
    if metadata.get("bridge_geometry", "straight_linear") not in {
        "straight_linear",
        None,
    }:
        raise ValueError("NFE grid diagnostic currently supports straight geometry only")

    schedule_name = str(metadata["schedule_name"])
    if schedule_name not in {"original", "mr_r3", "canonical_alpha"}:
        raise ValueError(f"unsupported schedule for NFE diagnostic: {schedule_name!r}")
    schedule = get_bridge_schedule(
        schedule_name,
        mean_reversion_rate=float(metadata.get("mean_reversion_rate") or 3.0),
    )

    device = base.resolve_device(args.device)
    train_seed = base._train_seed_from_metadata(metadata)
    seed_everything(
        train_seed,
        deterministic_algorithms=base._deterministic_algorithms_from_metadata(metadata),
    )

    ignored = load_ignored_sample_ids(args.ignore_file)
    discovery = discover_sen12mscr(
        args.data_root,
        ignored_sample_ids=ignored,
        strict_counterparts=True,
    )
    datasets, pilot_manifest = build_pilot_datasets(
        discovery.samples,
        include_sample_id=True,
        strict_channels=True,
        verify_frozen_membership=True,
    )
    split_audit = pilot_split_audit(pilot_manifest)
    if metadata.get("split_protocol") != PILOT_PROTOCOL:
        raise ValueError("checkpoint does not use the fixed pilot split protocol")
    if metadata.get("split_audit") != split_audit:
        raise ValueError("current pilot split audit does not match checkpoint provenance")
    if metadata.get("pilot_manifest_sample_ids_sha256") != pilot_manifest["sample_ids_sha256"]:
        raise ValueError("pilot sample-ID fingerprint mismatch")

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=train_seed,
        pin_memory=(device.type == "cuda"),
    )

    model = LegacyDBCRNet()
    if count_legacy_parameters(model) != LEGACY_DBCR_PARAMETER_COUNT:
        raise RuntimeError("legacy DB-CR parameter count drifted")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)

    total_steps = int(metadata["total_steps"])
    results = []
    for nfe in sorted(set(int(v) for v in args.nfe)):
        for grid in args.grid:
            results.append(
                evaluate_one(
                    model,
                    loaders.test,
                    schedule=schedule,
                    total_steps=total_steps,
                    nfe=nfe,
                    grid=grid,
                    device=device,
                    max_batches=args.max_batches,
                    progress_every=args.progress_every,
                )
            )

    output_payload = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "checkpoint_metrics": payload.get("metrics", {}),
        "schedule_name": schedule_name,
        "conditioning_mode": metadata["conditioning_mode"],
        "diagnostic": "reverse_grid_parameterization",
        "split": "test",
        "split_protocol": PILOT_PROTOCOL,
        "pilot_manifest_sample_ids_sha256": pilot_manifest["sample_ids_sha256"],
        "batch_size": int(args.batch_size),
        "max_batches": args.max_batches,
        "paper_grade": False,
        "metric_reference_repo": REFERENCE_REPO,
        "metric_reference_commit": REFERENCE_COMMIT,
        "results": results,
    }

    output_path = Path(args.output) if args.output else checkpoint_path.parent.parent / "pilot_nfe_grid_metrics.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)

    print(json.dumps({"output": str(output_path), "results": results}, indent=2, sort_keys=True), flush=True)
    return output_payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
