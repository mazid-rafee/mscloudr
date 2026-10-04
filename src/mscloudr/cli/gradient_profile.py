"""Fixed-alpha optimization profile for the CanonicalAlpha pilot checkpoint.

This diagnostic asks how strongly different physical bridge states drive the
network during optimization.  For each requested alpha it evaluates the same
ordered pilot-training batches, computes the direct-x0 L1 loss, backpropagates
without taking an optimizer step, and records the full-model batch gradient
L2 norm

    G_B(alpha) = || grad_theta L_B(alpha) ||_2.

The output reports the mean and batch-to-batch variance of that scalar norm.
This is deliberately a *gradient-norm variability* diagnostic; it is not the
full covariance/variance of the gradient vector itself.

Only the pilot TRAIN split is used so held-out validation/test information is
not involved in later training-measure design.  No model parameters or running
statistics are updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch.utils.data import DataLoader

from mscloudr.bridge import make_bridge_state
from mscloudr.cli import eval as eval_base
from mscloudr.cli import fixed_alpha_sweep, pilot_eval
from mscloudr.data import discover_sen12mscr, load_ignored_sample_ids
from mscloudr.data.pilot import (
    PILOT_PROTOCOL,
    build_pilot_datasets,
    pilot_split_audit,
)
from mscloudr.models import (
    LEGACY_DBCR_PARAMETER_COUNT,
    LegacyDBCRNet,
    count_legacy_parameters,
)
from mscloudr.reproducibility import seed_dataloader_worker, seed_everything
from mscloudr.training import (
    PHYSICAL_ALPHA_CONDITIONING,
    RAW_T_CONDITIONING,
)


DEFAULT_ALPHAS = tuple(i / 10.0 for i in range(11))
DEFAULT_MAX_BATCHES = 256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fixed-alpha L1 and full-model batch gradient norms on "
            "the deterministic 10% pilot TRAIN split."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--ignore-file",
        default="manifests/known_invalid_samples.txt",
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
        "--max-batches",
        type=int,
        default=DEFAULT_MAX_BATCHES,
        help=(
            "Number of equal-size train batches evaluated at every alpha. "
            "Default 256 (1024 samples for batch-size 4)."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N batches at each alpha; 0 disables.",
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


def validate_checkpoint_metadata(metadata: dict) -> str:
    """Require the canonical bridge coordinate used by this diagnostic."""

    if metadata.get("split_protocol") != PILOT_PROTOCOL:
        raise ValueError("gradient profile requires the fixed pilot10 checkpoint")
    if metadata.get("schedule_name") != "canonical_alpha":
        raise ValueError("gradient profile requires a canonical_alpha checkpoint")

    mode = fixed_alpha_sweep.conditioning_mode_from_metadata(metadata)
    if mode not in {RAW_T_CONDITIONING, PHYSICAL_ALPHA_CONDITIONING}:
        raise ValueError(f"unsupported canonical conditioning mode: {mode!r}")

    # For canonical_alpha, legacy raw_t and physical-alpha conditioning are
    # numerically the same coordinate on the training grid: t = T*alpha.
    return mode


def full_model_gradient_l2(model: torch.nn.Module) -> float:
    """Return sqrt(sum_p ||grad_p||_2^2) over all parameters with gradients."""

    device = next(model.parameters()).device
    sum_squares = torch.zeros((), dtype=torch.float64, device=device)
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        gradient = parameter.grad.detach()
        sum_squares = sum_squares + torch.sum(
            gradient.to(dtype=torch.float64) ** 2
        )
    if not found:
        raise ValueError("model has no parameter gradients")
    return math.sqrt(float(sum_squares.item()))


def summarize_batch_values(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    avg = float(mean(values))
    std = float(pstdev(values))
    variance = std * std
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    return {
        "mean": avg,
        "std_population": std,
        "variance_population": variance,
        "rms": float(rms),
        "min": float(min(values)),
        "max": float(max(values)),
        "coefficient_of_variation": (
            float(std / avg) if avg != 0.0 else 0.0
        ),
    }


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: (
            value.to(device=device, non_blocking=True)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def _sample_ids_from_batch(batch: dict) -> list[str]:
    values = batch.get("sample_id")
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _fingerprint_ids(sample_ids: list[str]) -> str:
    payload = "\n".join(sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_output(checkpoint: Path) -> Path:
    filename = "gradient_profile_train.json"
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent / filename
    return checkpoint.with_name(filename)


def run(args: argparse.Namespace) -> dict:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.max_batches <= 0:
        raise ValueError("max-batches must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every must be non-negative")

    alphas = parse_alphas(args.alphas)
    checkpoint_path = Path(args.checkpoint)
    payload = eval_base._torch_load_payload(checkpoint_path)
    metadata = pilot_eval._validate_pilot_checkpoint(payload)
    conditioning_mode = validate_checkpoint_metadata(metadata)

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
    if metadata.get("split_audit") != split_audit:
        raise ValueError("current pilot split audit does not match checkpoint provenance")
    if (
        metadata.get("pilot_manifest_sample_ids_sha256")
        != pilot_manifest["sample_ids_sha256"]
    ):
        raise ValueError("pilot sample-ID fingerprint does not match checkpoint provenance")

    device = eval_base.resolve_device(args.device)
    train_seed = eval_base._train_seed_from_metadata(metadata)
    seed_everything(
        train_seed,
        deterministic_algorithms=eval_base._deterministic_algorithms_from_metadata(
            metadata
        ),
    )

    # Sequential, drop-last loader ensures that every alpha sees exactly the
    # same equal-size batches, making batch gradient norms directly comparable.
    loader = DataLoader(
        datasets.train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=seed_dataloader_worker,
        persistent_workers=(args.num_workers > 0),
    )
    total_batches = min(len(loader), int(args.max_batches))

    model = LegacyDBCRNet()
    if count_legacy_parameters(model) != LEGACY_DBCR_PARAMETER_COUNT:
        raise RuntimeError("legacy_dbcr parameter count drifted")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()

    total_steps = int(metadata["total_steps"])
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(payload.get("epoch", 0)),
                "schedule_name": metadata["schedule_name"],
                "conditioning_mode": conditioning_mode,
                "split": "train",
                "split_samples": len(datasets.train),
                "batch_size": args.batch_size,
                "max_batches": args.max_batches,
                "evaluated_batches_per_alpha": total_batches,
                "evaluated_samples_per_alpha": total_batches * args.batch_size,
                "alphas": alphas,
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    results: list[dict] = []
    expected_sample_fingerprint: str | None = None

    for alpha in alphas:
        conditioning = fixed_alpha_sweep.conditioning_for_alpha(
            alpha,
            schedule_name="canonical_alpha",
            total_steps=total_steps,
            conditioning_mode=conditioning_mode,
        )
        model_conditioning_value = float(
            conditioning["model_conditioning_value"]
        )

        batch_losses: list[float] = []
        batch_gradient_norms: list[float] = []
        sample_ids: list[str] = []
        num_samples = 0

        for batch_index, batch in enumerate(loader):
            if batch_index >= total_batches:
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
            model_conditioning = torch.full(
                (target.shape[0],),
                model_conditioning_value,
                device=device,
                dtype=torch.float32,
            )

            model.zero_grad(set_to_none=True)
            prediction = model(bridge_state, model_conditioning, sar)
            loss = torch.mean(torch.abs(prediction - target))
            loss.backward()
            grad_l2 = full_model_gradient_l2(model)

            batch_losses.append(float(loss.detach().item()))
            batch_gradient_norms.append(float(grad_l2))
            sample_ids.extend(_sample_ids_from_batch(batch))
            num_samples += int(target.shape[0])

            batch_number = batch_index + 1
            if (
                args.progress_every > 0
                and (
                    batch_number % args.progress_every == 0
                    or batch_number == total_batches
                )
            ):
                print(
                    "alpha={alpha:.2f} batch={batch}/{total} samples={samples} "
                    "L1={loss:.6f} grad_l2={grad:.6f}".format(
                        alpha=alpha,
                        batch=batch_number,
                        total=total_batches,
                        samples=num_samples,
                        loss=batch_losses[-1],
                        grad=batch_gradient_norms[-1],
                    ),
                    flush=True,
                )

        model.zero_grad(set_to_none=True)
        sample_fingerprint = _fingerprint_ids(sample_ids)
        if expected_sample_fingerprint is None:
            expected_sample_fingerprint = sample_fingerprint
        elif sample_fingerprint != expected_sample_fingerprint:
            raise RuntimeError("different training samples were evaluated across alpha levels")

        results.append(
            {
                "requested_alpha": float(alpha),
                **conditioning,
                "num_batches": len(batch_losses),
                "num_samples": num_samples,
                "sample_ids_sha256": sample_fingerprint,
                "batch_l1": summarize_batch_values(batch_losses),
                "batch_gradient_l2": summarize_batch_values(
                    batch_gradient_norms
                ),
            }
        )

    output_payload = {
        "format_version": 1,
        "diagnostic": "fixed_physical_alpha_gradient_profile",
        "analysis_role": "optimization_profile_for_bridge_measure_design",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "model_identity": metadata["model_identity"],
        "schedule_name": metadata["schedule_name"],
        "conditioning_mode": conditioning_mode,
        "canonical_coordinate_equivalence": (
            "for canonical_alpha, raw_t=t and physical c=T*alpha are identical "
            "on the discrete training grid"
        ),
        "split": "train",
        "split_protocol": metadata["split_protocol"],
        "held_out_data_used": False,
        "pilot_manifest_sample_ids_sha256": pilot_manifest[
            "sample_ids_sha256"
        ],
        "evaluated_sample_ids_sha256": expected_sample_fingerprint,
        "same_batches_across_alpha": True,
        "loader_shuffle": False,
        "drop_last": True,
        "batch_size": int(args.batch_size),
        "max_batches": int(args.max_batches),
        "gradient_definition": (
            "G_B(alpha)=sqrt(sum_p ||d L1_B(alpha)/d theta_p||_2^2)"
        ),
        "gradient_variance_definition": (
            "population variance across equal-size batch gradient L2 norms; "
            "not gradient-vector covariance"
        ),
        "parameter_updates_performed": False,
        "model_mode": "eval",
        "results": results,
    }

    output_path = (
        Path(args.output)
        if args.output is not None
        else _default_output(checkpoint_path)
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
