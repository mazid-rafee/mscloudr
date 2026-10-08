"""Full-data CanonicalBridgeNet training with controlled bridge measures.

This entry point keeps the architecture and straight physical bridge fixed while
allowing two independent experimental axes:

1. training measure q(alpha): uniform or Beta(a, 1);
2. conditioning coordinate: physical alpha, inverse-sine, or inverse-MR(r=3).

All runs use the frozen ROI-disjoint SEN12MS-CR reference split. The bridge
state is always x_alpha = (1-alpha)*x0 + alpha*y; changing the conditioning
coordinate never changes alpha or x_alpha.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mscloudr.bridge import get_bridge_schedule
from mscloudr.cli import train as base
from mscloudr.data import (
    REFERENCE_PROTOCOL,
    build_reference_dataloaders,
    build_reference_datasets,
    discover_sen12mscr,
    load_ignored_sample_ids,
    reference_split_audit,
)
from mscloudr.models import (
    CANONICAL_BRIDGE_MODEL_IDENTITY,
    CanonicalBridgeNet,
    count_canonical_bridge_parameters,
)
from mscloudr.reproducibility import (
    SeedConfig,
    make_torch_generator,
    reproducibility_metadata,
    seed_everything,
)
from mscloudr.runner import fit
from mscloudr.training import (
    MR_INVERSE_CONDITIONING,
    PHYSICAL_ALPHA_CONDITIONING,
    SINE_INVERSE_CONDITIONING,
)


DEFAULT_EPOCHS = 50
SUPPORTED_CANONICAL_CONDITIONING = (
    PHYSICAL_ALPHA_CONDITIONING,
    SINE_INVERSE_CONDITIONING,
    MR_INVERSE_CONDITIONING,
)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = (
        "Train CanonicalBridgeNet on the full frozen ROI-disjoint SEN12MS-CR "
        "split with controlled q(alpha) and conditioning coordinates."
    )

    schedule_action = parser._option_string_actions["--schedule"]
    schedule_action.required = False
    schedule_action.choices = ("canonical_alpha",)
    schedule_action.default = "canonical_alpha"

    parser.set_defaults(epochs=DEFAULT_EPOCHS)
    parser.add_argument(
        "--beta-a",
        type=float,
        default=1.0,
        help=(
            "Beta(a,1) physical-alpha measure. a=1 is the uniform canonical "
            "measure; a>1 shifts training mass toward the cloudy endpoint."
        ),
    )
    parser.add_argument(
        "--conditioning-coordinate",
        choices=SUPPORTED_CANONICAL_CONDITIONING,
        default=PHYSICAL_ALPHA_CONDITIONING,
        help=(
            "Scalar coordinate supplied to CanonicalBridgeNet while holding "
            "the physical bridge state fixed."
        ),
    )
    return parser


def _validate_experiment(args: argparse.Namespace) -> float:
    if args.schedule != "canonical_alpha":
        raise ValueError("CanonicalBridgeNet full runs require canonical_alpha")
    beta_a = float(args.beta_a)
    if not math.isfinite(beta_a) or beta_a <= 0.0:
        raise ValueError("--beta-a must be finite and positive")
    if args.conditioning_coordinate not in SUPPORTED_CANONICAL_CONDITIONING:
        raise ValueError(
            f"unsupported conditioning coordinate: {args.conditioning_coordinate!r}"
        )
    return beta_a


def _beta_a1_schedule(beta_a: float):
    """Map uniform discrete u=t/T to alpha with Beta(a,1) continuous limit."""

    beta_a = float(beta_a)
    if not math.isfinite(beta_a) or beta_a <= 0.0:
        raise ValueError("beta-a must be finite and positive")

    if math.isclose(beta_a, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return get_bridge_schedule("canonical_alpha")

    inverse_a = 1.0 / beta_a

    def schedule(t, total_steps: int):
        import torch

        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        t_float = t if torch.is_tensor(t) else torch.as_tensor(t)
        if not t_float.is_floating_point():
            t_float = t_float.float()
        u = torch.clamp(t_float / float(total_steps), 0.0, 1.0)
        alpha = torch.pow(u, inverse_a)
        return torch.where(
            u <= 0.0,
            torch.zeros_like(alpha),
            torch.where(u >= 1.0, torch.ones_like(alpha), alpha),
        )

    return schedule


def _measure_metadata(beta_a: float) -> dict[str, Any]:
    beta_a = float(beta_a)
    if math.isclose(beta_a, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return {
            "training_bridge_measure": "uniform_discrete_alpha_grid",
            "training_bridge_measure_definition": (
                "t~Uniform{0,...,T}; u=t/T; alpha=u"
            ),
            "alpha_sampling_identity": "canonical_uniform_discrete_alpha",
            "beta_a": 1.0,
            "beta_b": 1.0,
            "continuous_mean_alpha": 0.5,
        }

    return {
        "training_bridge_measure": "beta_a1_physical_alpha",
        "training_bridge_measure_definition": (
            "t~Uniform{0,...,T}; u=t/T; alpha=u^(1/a); "
            "continuous-limit alpha~Beta(a,1)"
        ),
        "training_bridge_measure_density": "q(alpha)=a*alpha^(a-1)",
        "alpha_sampling_identity": "beta_a1_quantile_transform",
        "beta_a": beta_a,
        "beta_b": 1.0,
        "continuous_mean_alpha": beta_a / (beta_a + 1.0),
    }


def _conditioning_metadata(mode: str) -> dict[str, Any]:
    definitions = {
        PHYSICAL_ALPHA_CONDITIONING: {
            "conditioning_identity": "canonical_physical_alpha",
            "conditioning_definition": "u=alpha; c=T*u",
        },
        SINE_INVERSE_CONDITIONING: {
            "conditioning_identity": "canonical_inverse_sine_coordinate",
            "conditioning_definition": "u=(2/pi)*asin(alpha); c=T*u",
        },
        MR_INVERSE_CONDITIONING: {
            "conditioning_identity": "canonical_inverse_mr_r3_coordinate",
            "conditioning_definition": (
                "u=-(1/3)*log(1-(1-exp(-3))*alpha); c=T*u"
            ),
        },
    }
    return {
        "conditioning_mode": mode,
        "conditioning_coordinate": mode,
        **definitions[mode],
        "conditioning_range": "[0,T]",
        "conditioning_changes_bridge_state": False,
    }


def _run_metadata(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    split_audit: dict[str, Any],
    trainable_parameters: int,
    beta_a: float,
    device,
) -> dict[str, Any]:
    seed_metadata = reproducibility_metadata(
        SeedConfig(
            split_seed=0,
            train_seed=args.train_seed,
            sampler_seed=args.sampler_seed,
        ),
        deterministic_algorithms=args.deterministic_algorithms,
    )
    seed_metadata["seeds"]["split_seed"] = None

    metadata: dict[str, Any] = {
        "format_version": 1,
        "model_identity": CANONICAL_BRIDGE_MODEL_IDENTITY,
        "model_family": "dual_stream_residual_unet",
        "trainable_parameters": int(trainable_parameters),
        "dbcr_specific_blocks": False,
        "naf_blocks": False,
        "sf_blocks": False,
        "architecture_widths": [32, 64, 128, 256],
        "architecture_mid_blocks": 2,
        "sar_fusion": "gated_additive_multiscale",
        "upsampling": "bilinear_plus_3x3_conv",
        "prediction_target": "direct_x0",
        "bridge_geometry": "straight_linear",
        "bridge_coordinate": "physical_alpha",
        "bridge_state_definition": "x_alpha=(1-alpha)*x0+alpha*y",
        "schedule_name": "canonical_alpha",
        "total_steps": int(args.total_steps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "learning_rate": float(args.lr),
        "checkpoint_selection_metric": "val_endpoint_l1",
        "diagnostic_random_t_metric": "val_random_t_l1",
        "split_protocol": REFERENCE_PROTOCOL,
        "split_seed": None,
        "split_seed_note": "fixed ROI membership; no split RNG is consumed",
        "split_audit": split_audit,
        "data_root": str(dataset_root),
        "ignore_file": str(args.ignore_file),
        "device": str(device),
        "run_kind": "smoke" if args.smoke_run else "full",
        "paper_grade": not args.smoke_run,
        "max_train_batches": (
            int(args.smoke_train_batches) if args.smoke_run else None
        ),
        "max_val_batches": (
            int(args.smoke_val_batches) if args.smoke_run else None
        ),
        "progress_every": int(args.progress_every),
        "reproducibility": seed_metadata,
    }
    metadata.update(_measure_metadata(beta_a))
    metadata.update(_conditioning_metadata(args.conditioning_coordinate))
    return metadata


def _resume_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_identity": metadata["model_identity"],
        "trainable_parameters": metadata["trainable_parameters"],
        "bridge_geometry": metadata["bridge_geometry"],
        "schedule_name": metadata["schedule_name"],
        "training_bridge_measure": metadata["training_bridge_measure"],
        "beta_a": metadata["beta_a"],
        "beta_b": metadata["beta_b"],
        "conditioning_mode": metadata["conditioning_mode"],
        "conditioning_definition": metadata["conditioning_definition"],
        "total_steps": metadata["total_steps"],
        "batch_size": metadata["batch_size"],
        "num_workers": metadata["num_workers"],
        "learning_rate": metadata["learning_rate"],
        "split_protocol": metadata["split_protocol"],
        "split_audit": metadata["split_audit"],
        "train_seed": metadata["reproducibility"]["seeds"]["train_seed"],
        "sampler_seed": metadata["reproducibility"]["seeds"]["sampler_seed"],
        "deterministic_algorithms": metadata["reproducibility"][
            "deterministic_algorithms"
        ],
    }


def _validate_resume_metadata(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> None:
    previous_identity = _resume_identity(previous)
    current_identity = _resume_identity(current)
    if previous_identity != current_identity:
        mismatches = [
            key
            for key in previous_identity
            if previous_identity[key] != current_identity[key]
        ]
        raise ValueError(
            "resume configuration mismatch for: " + ", ".join(mismatches)
        )


def run(args: argparse.Namespace):
    base.validate_cli_args(args)
    beta_a = _validate_experiment(args)
    run_dir = base._prepare_output_dir(args)
    device = base.resolve_device(args.device)

    seed_everything(
        args.train_seed,
        deterministic_algorithms=args.deterministic_algorithms,
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
        {
            split: dataset.samples
            for split, dataset in datasets.as_dict().items()
        }
    )
    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=args.train_seed,
        pin_memory=(device.type == "cuda"),
    )

    model = CanonicalBridgeNet(total_steps=args.total_steps)
    trainable_parameters = count_canonical_bridge_parameters(model)
    schedule = _beta_a1_schedule(beta_a)

    metadata = _run_metadata(
        args=args,
        dataset_root=discovery.dataset_root,
        split_audit=split_audit,
        trainable_parameters=trainable_parameters,
        beta_a=beta_a,
        device=device,
    )

    config_path = run_dir / "run_config.json"
    if args.resume is None:
        base._write_json_atomic(config_path, metadata)
    else:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"resume requires existing run_config.json: {config_path}"
            )
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        _validate_resume_metadata(previous, metadata)
        base._write_json_atomic(config_path, metadata)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "device": str(device),
                "model_identity": CANONICAL_BRIDGE_MODEL_IDENTITY,
                "trainable_parameters": trainable_parameters,
                "training_bridge_measure": metadata["training_bridge_measure"],
                "beta_a": beta_a,
                "conditioning_coordinate": args.conditioning_coordinate,
                "bridge_geometry": "straight_linear",
                "epochs": args.epochs,
                "train_samples": len(datasets.train),
                "val_samples": len(datasets.val),
                "test_samples": len(datasets.test),
                "checkpoint_selection_metric": "val_endpoint_l1",
                "deterministic_algorithms": args.deterministic_algorithms,
                "paper_grade": not args.smoke_run,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    sampler_generator = make_torch_generator(args.sampler_seed, device="cpu")
    return fit(
        model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        schedule=schedule,
        schedule_name="canonical_alpha",
        total_steps=args.total_steps,
        sampler_generator=sampler_generator,
        output_dir=run_dir,
        epochs=args.epochs,
        device=device,
        model_identity=CANONICAL_BRIDGE_MODEL_IDENTITY,
        conditioning_mode=args.conditioning_coordinate,
        lr=args.lr,
        run_metadata=metadata,
        resume_from=args.resume,
        epoch_callback=base._epoch_progress,
        batch_progress_callback=(
            base._batch_progress if args.progress_every > 0 else None
        ),
        progress_every=(
            args.progress_every if args.progress_every > 0 else None
        ),
        max_train_batches=(
            args.smoke_train_batches if args.smoke_run else None
        ),
        max_val_batches=(
            args.smoke_val_batches if args.smoke_run else None
        ),
    )


def main() -> None:
    args = build_parser().parse_args()
    history = run(args)
    last = history[-1]
    print(
        json.dumps(
            {
                "completed_epoch": last.epoch,
                "train_l1": last.train_l1,
                "val_random_t_l1": last.val_random_t_l1,
                "val_endpoint_l1": last.val_endpoint_l1,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
