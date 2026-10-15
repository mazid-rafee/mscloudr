"""Paired, fixed-alpha validation diagnostic for two CanonicalBridge checkpoints.

This is an ORACLE bridge-state diagnostic: at alpha < 1 the input contains
the clean validation target. It is not an executable multi-step sampler.
The existing training and endpoint evaluation commands are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from mscloudr.bridge import make_bridge_state
from mscloudr.cli import eval as base
from mscloudr.cli.eval_canonical_bridge import (
    _checkpoint_widths,
    _checkpoint_backbone,
    _validate_checkpoint,
)
from mscloudr.data import (
    REFERENCE_PROTOCOL,
    build_reference_dataloaders,
    build_reference_datasets,
    discover_sen12mscr,
    load_ignored_sample_ids,
    reference_split_audit,
)
from mscloudr.metrics import REFERENCE_COMMIT, REFERENCE_REPO, ReferenceMetricAccumulator
from mscloudr.models import CanonicalBridgeNet, count_canonical_bridge_parameters
from mscloudr.reproducibility import seed_everything
from mscloudr.training import MR_INVERSE_CONDITIONING, model_conditioning_coordinate


DEFAULT_ALPHAS = (0.25, 0.50, 0.75, 0.90, 0.95, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two MR-inverse CanonicalBridge checkpoints at fixed "
            "oracle bridge positions on the frozen validation split."
        )
    )
    parser.add_argument("--checkpoint-a", required=True, help="Baseline checkpoint.")
    parser.add_argument("--checkpoint-b", required=True, help="Capacity-scaled checkpoint.")
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS)
    )
    parser.add_argument(
        "--split", choices=("val",), default="val",
        help="Validation only; the test split is intentionally unsupported.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--ignore-file", default="manifests/known_invalid_samples.txt"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Optional first-N-batches pilot; omitted for all validation samples.",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting the specified diagnostic JSON file.",
    )
    return parser


def _validate_alphas(alphas: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(float(alpha) for alpha in alphas)
    if not values:
        raise ValueError("at least one alpha is required")
    if any(not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0 for alpha in values):
        raise ValueError("alphas must be finite numbers in [0, 1]")
    if len(values) != len(set(values)):
        raise ValueError("duplicate alpha values are not allowed")
    return values


def _load_model(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[nn.Module, int]:
    widths = _checkpoint_widths(metadata)
    total_steps = int(metadata["total_steps"])
    model = CanonicalBridgeNet(
        widths=widths,
        total_steps=total_steps,
        backbone_variant=_checkpoint_backbone(metadata),
    )
    actual_parameters = count_canonical_bridge_parameters(model)
    if actual_parameters != int(metadata["trainable_parameters"]):
        raise RuntimeError(
            "checkpoint trainable_parameters disagrees with reconstructed model"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), total_steps


def evaluate_alpha_grid(
    models: Mapping[str, tuple[nn.Module, int]],
    loader: DataLoader,
    *,
    alphas: tuple[float, ...],
    device: str | torch.device,
    max_batches: int | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """One deterministic loader pass, identical x_alpha and SAR for both models."""
    alphas = _validate_alphas(alphas)
    if set(models) != {"a", "b"}:
        raise ValueError("models must have exactly the labels 'a' and 'b'")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")

    device = torch.device(device)
    for model, steps in models.values():
        if int(steps) <= 0:
            raise ValueError("model total_steps must be positive")
        model.to(device).eval()

    meters = {
        label: {alpha: ReferenceMetricAccumulator() for alpha in alphas}
        for label in ("a", "b")
    }
    num_samples = 0
    num_batches = 0
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            target = batch["target"].to(device, non_blocking=True)
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            batch_size = int(target.shape[0])

            for alpha in alphas:
                alpha_vec = target.new_full((batch_size,), alpha)
                x_alpha = make_bridge_state(target, cloudy, alpha_vec)
                for label, (model, total_steps) in models.items():
                    timesteps = torch.full(
                        (batch_size,), total_steps,
                        dtype=torch.long, device=device,
                    )
                    conditioning = model_conditioning_coordinate(
                        timesteps,
                        alpha_vec,
                        total_steps=total_steps,
                        conditioning_mode=MR_INVERSE_CONDITIONING,
                    )
                    prediction = model(x_alpha, conditioning, sar)
                    if prediction.shape != target.shape:
                        raise ValueError(
                            f"model {label!r} produced an unexpected output shape"
                        )
                    meters[label][alpha].update_batch(target, prediction)

            num_samples += batch_size
            num_batches += 1
            if progress_every and (
                num_batches % progress_every == 0
                or num_batches == total_batches
            ):
                print(
                    f"alpha-grid batch={num_batches}/{total_batches} "
                    f"samples={num_samples}",
                    flush=True,
                )

    if num_samples == 0:
        raise ValueError("validation loader produced no samples")

    rows = []
    for alpha in alphas:
        a_metrics = dict(meters["a"][alpha].compute())
        b_metrics = dict(meters["b"][alpha].compute())
        a_metrics["L1"] = a_metrics["MAE"]
        b_metrics["L1"] = b_metrics["MAE"]
        a_counts = meters["a"][alpha].metric_counts()
        b_counts = meters["b"][alpha].metric_counts()
        a_counts["L1"] = a_counts["MAE"]
        b_counts["L1"] = b_counts["MAE"]
        rows.append(
            {
                "alpha": alpha,
                "metrics_a": a_metrics,
                "metrics_b": b_metrics,
                "delta_b_minus_a": {
                    key: b_metrics[key] - a_metrics[key]
                    for key in sorted(set(a_metrics) & set(b_metrics))
                },
                "metric_counts_a": a_counts,
                "metric_counts_b": b_counts,
            }
        )

    return {
        "num_samples": num_samples,
        "num_batches": num_batches,
        "max_batches": max_batches,
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0 or args.num_workers < 0 or args.progress_every < 0:
        raise ValueError("invalid batch-size, num-workers, or progress-every")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    alphas = _validate_alphas(args.alphas)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"diagnostic output already exists: {output_path}; use --overwrite"
        )

    checkpoint_paths = {
        "a": Path(args.checkpoint_a),
        "b": Path(args.checkpoint_b),
    }
    payloads = {}
    metadata = {}
    for label, path in checkpoint_paths.items():
        payload = base._torch_load_payload(path)
        meta = _validate_checkpoint(payload, allow_smoke_checkpoint=False)
        if meta["conditioning_mode"] != MR_INVERSE_CONDITIONING:
            raise ValueError(
                f"checkpoint {label} is not MR-inverse conditioned"
            )
        payloads[label] = payload
        metadata[label] = meta

    if metadata["a"].get("split_audit") != metadata["b"].get("split_audit"):
        raise ValueError("the two checkpoint split audits do not match")
    if metadata["a"].get("split_protocol") != REFERENCE_PROTOCOL:
        raise ValueError("checkpoints do not use the reference split protocol")

    device = base.resolve_device(args.device)
    train_seed = base._train_seed_from_metadata(metadata["a"])
    seed_everything(
        train_seed,
        deterministic_algorithms=base._deterministic_algorithms_from_metadata(
            metadata["a"]
        ),
    )
    ignored = load_ignored_sample_ids(args.ignore_file)
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
        {split: dataset.samples for split, dataset in datasets.as_dict().items()}
    )
    if any(meta.get("split_audit") != split_audit for meta in metadata.values()):
        raise ValueError(
            "dataset split audit does not match both checkpoint provenances"
        )

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=train_seed,
        pin_memory=(device.type == "cuda"),
    )
    models = {
        label: _load_model(payloads[label], metadata[label], device=device)
        for label in ("a", "b")
    }

    print(
        json.dumps(
            {
                "split": "val",
                "validation_samples": len(datasets.val),
                "alphas": alphas,
                "checkpoints": {key: str(path) for key, path in checkpoint_paths.items()},
                "parameters": {
                    key: metadata[key]["trainable_parameters"] for key in ("a", "b")
                },
                "device": str(device),
                "max_batches": args.max_batches,
            },
            indent=2,
        ),
        flush=True,
    )
    results = evaluate_alpha_grid(
        models,
        loaders.val,
        alphas=alphas,
        device=device,
        max_batches=args.max_batches,
        progress_every=args.progress_every,
    )
    output = {
        "format_version": 1,
        "experiment": "canonical_bridge_capacity_fixed_alpha",
        "diagnostic_only": True,
        "oracle_clean_target_in_bridge_state": True,
        "bridge_state": "x_alpha=(1-alpha)*target+alpha*cloudy",
        "conditioning": "mr_inverse_r3",
        "split": "val",
        "split_protocol": REFERENCE_PROTOCOL,
        "split_audit": split_audit,
        "full_validation": args.max_batches is None,
        "validation_samples_available": len(datasets.val),
        "checkpoint_a": {
            "path": str(checkpoint_paths["a"]),
            "epoch": int(payloads["a"].get("epoch", 0)),
            "widths": list(_checkpoint_widths(metadata["a"])),
            "backbone_variant": _checkpoint_backbone(metadata["a"]),
            "trainable_parameters": int(metadata["a"]["trainable_parameters"]),
            "total_steps": models["a"][1],
        },
        "checkpoint_b": {
            "path": str(checkpoint_paths["b"]),
            "epoch": int(payloads["b"].get("epoch", 0)),
            "widths": list(_checkpoint_widths(metadata["b"])),
            "backbone_variant": _checkpoint_backbone(metadata["b"]),
            "trainable_parameters": int(metadata["b"]["trainable_parameters"]),
            "total_steps": models["b"][1],
        },
        "metric_reference_repo": REFERENCE_REPO,
        "metric_reference_commit": REFERENCE_COMMIT,
        "prediction_clipping": False,
        "batch_size": args.batch_size,
        **results,
    }
    base._write_json_atomic(output_path, output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "num_samples": results["num_samples"],
                "rows": [
                    {
                        "alpha": row["alpha"],
                        "L1_a": row["metrics_a"]["L1"],
                        "L1_b": row["metrics_b"]["L1"],
                        "delta_L1_b_minus_a": row["delta_b_minus_a"]["L1"],
                    }
                    for row in results["rows"]
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
