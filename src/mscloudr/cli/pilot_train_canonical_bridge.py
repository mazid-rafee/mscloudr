"""Pilot training for the architecture-independent CanonicalBridgeNet.

This command intentionally fixes bridge geometry to the straight physical-alpha
path and conditioning to c=T*alpha.  The first controlled experiment changes
only q(alpha): uniform physical-alpha versus Beta(a,1).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from mscloudr.cli import pilot_train as legacy_pilot
from mscloudr.cli import train as base
from mscloudr.data import (
    build_reference_dataloaders,
    discover_sen12mscr,
    load_ignored_sample_ids,
)
from mscloudr.data.pilot import (
    PILOT_FRACTION,
    PILOT_PROTOCOL,
    PILOT_SEED,
    build_pilot_datasets,
    pilot_split_audit,
)
from mscloudr.models import (
    CANONICAL_BRIDGE_MODEL_IDENTITY,
    CANONICAL_DUAL_ROLE_SAR_BRIDGE_MODEL_IDENTITY,
    CANONICAL_SAR_RESIDUAL_COORDINATE_MODEL_IDENTITY,
    CanonicalBridgeNet,
    CanonicalDualRoleSARBridgeNet,
    CanonicalSARResidualCoordinateBridgeNet,
    count_canonical_bridge_parameters,
    count_canonical_dual_role_sar_bridge_parameters,
    count_canonical_sar_residual_coordinate_parameters,
)
from mscloudr.reproducibility import make_torch_generator, seed_everything
from mscloudr.runner import fit
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING


DEFAULT_PILOT_EPOCHS = 25


def build_parser():
    parser = legacy_pilot.build_parser()
    parser.description = (
        "Train CanonicalBridgeNet on the fixed 10% SEN12MS-CR pilot split "
        "with straight geometry, physical-alpha conditioning, and a controlled "
        "uniform/Beta(a,1) training measure."
    )
    parser.set_defaults(
        epochs=DEFAULT_PILOT_EPOCHS,
        schedule="canonical_alpha",
        conditioning=PHYSICAL_ALPHA_CONDITIONING,
        endpoint_prob=0.0,
    )
    parser.add_argument(
        "--sar-bridge-kappa",
        type=float,
        default=None,
        help=(
            "Enable dual-role SAR-aware bridge curvature with the given "
            "positive maximum optical-space amplitude. Omit for the straight "
            "CanonicalBridge baseline."
        ),
    )
    parser.add_argument(
        "--sar-residual-coordinate-kappa",
        type=float,
        default=None,
        help=(
            "Enable SAR-gated residual-coordinate bridge geometry. SAR predicts "
            "a one-channel spatial gate that modulates the effective alpha "
            "along the clean-to-cloudy residual direction. Requires 0<kappa<0.25."
        ),
    )
    parser.add_argument(
        "--curved-val-nfes",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Optional matched curved-reverse validation NFEs. For example "
            "--curved-val-nfes 2 3 5 saves best_curved_nfe2.pt, "
            "best_curved_nfe3.pt, and best_curved_nfe5.pt."
        ),
    )
    return parser


def _validate_experiment(args) -> float:
    if args.schedule != "canonical_alpha":
        raise ValueError(
            "CanonicalBridgeNet pilot requires --schedule canonical_alpha"
        )
    if args.conditioning != PHYSICAL_ALPHA_CONDITIONING:
        raise ValueError(
            "CanonicalBridgeNet pilot requires --conditioning physical_alpha"
        )
    if not math.isclose(float(args.endpoint_prob), 0.0, abs_tol=1e-12):
        raise ValueError(
            "CanonicalBridgeNet first controlled experiment does not use endpoint mixture"
        )
    beta_a = float(args.beta_a)
    if beta_a <= 0.0:
        raise ValueError("--beta-a must be positive")
    if args.sar_bridge_kappa is not None and float(args.sar_bridge_kappa) <= 0.0:
        raise ValueError("--sar-bridge-kappa must be positive when provided")
    if args.sar_residual_coordinate_kappa is not None:
        kappa = float(args.sar_residual_coordinate_kappa)
        if not (0.0 < kappa < 0.25):
            raise ValueError(
                "--sar-residual-coordinate-kappa must satisfy 0 < kappa < 0.25"
            )
    if (
        args.sar_bridge_kappa is not None
        and args.sar_residual_coordinate_kappa is not None
    ):
        raise ValueError(
            "--sar-bridge-kappa and --sar-residual-coordinate-kappa "
            "are mutually exclusive"
        )
    curved_val_nfes = tuple(sorted(set(int(nfe) for nfe in args.curved_val_nfes)))
    if any(nfe < 2 for nfe in curved_val_nfes):
        raise ValueError("--curved-val-nfes values must be >= 2")
    if curved_val_nfes and args.sar_bridge_kappa is None:
        raise ValueError(
            "--curved-val-nfes currently requires --sar-bridge-kappa; "
            "residual-coordinate matched reverse validation is a separate step"
        )
    args.curved_val_nfes = curved_val_nfes
    return beta_a


def run(args):
    base.validate_cli_args(args)
    beta_a = _validate_experiment(args)
    run_dir = base._prepare_output_dir(args)
    device = base.resolve_device(args.device)

    schedule_name = "canonical_alpha"
    beta_active = not math.isclose(beta_a, 1.0, rel_tol=0.0, abs_tol=1e-12)
    if beta_active:
        schedule = legacy_pilot._beta_a1_schedule(beta_a)
    else:
        _, schedule, _ = legacy_pilot._resolve_pilot_schedule(schedule_name)

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
    datasets, pilot_manifest = build_pilot_datasets(
        discovery.samples,
        include_sample_id=True,
        strict_channels=True,
        verify_frozen_membership=True,
    )
    split_audit = pilot_split_audit(pilot_manifest)

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=args.train_seed,
        pin_memory=(device.type == "cuda"),
    )

    sar_bridge_active = args.sar_bridge_kappa is not None
    residual_coordinate_active = args.sar_residual_coordinate_kappa is not None
    if residual_coordinate_active:
        model = CanonicalSARResidualCoordinateBridgeNet(
            total_steps=args.total_steps,
            residual_coordinate_kappa=float(
                args.sar_residual_coordinate_kappa
            ),
        )
        model_identity = CANONICAL_SAR_RESIDUAL_COORDINATE_MODEL_IDENTITY
        trainable_parameters = count_canonical_sar_residual_coordinate_parameters(
            model
        )
        bridge_geometry = "sar_residual_coordinate"
    elif sar_bridge_active:
        model = CanonicalDualRoleSARBridgeNet(
            total_steps=args.total_steps,
            sar_bridge_kappa=float(args.sar_bridge_kappa),
        )
        model_identity = CANONICAL_DUAL_ROLE_SAR_BRIDGE_MODEL_IDENTITY
        trainable_parameters = count_canonical_dual_role_sar_bridge_parameters(
            model
        )
        bridge_geometry = "sar_curved_dual_role"
    else:
        model = CanonicalBridgeNet(total_steps=args.total_steps)
        model_identity = CANONICAL_BRIDGE_MODEL_IDENTITY
        trainable_parameters = count_canonical_bridge_parameters(model)
        bridge_geometry = "straight_linear"

    metadata = base._run_metadata(
        args=args,
        dataset_root=discovery.dataset_root,
        split_audit=split_audit,
        trainable_parameters=trainable_parameters,
        schedule_name=schedule_name,
        mean_reversion_rate=None,
        device=device,
    )
    metadata["reproducibility"]["seeds"]["split_seed"] = PILOT_SEED
    metadata.update(
        {
            "model_identity": model_identity,
            "model_family": "dual_stream_residual_unet",
            "dbcr_specific_blocks": False,
            "naf_blocks": False,
            "sf_blocks": False,
            "architecture_widths": [32, 64, 128, 256],
            "architecture_mid_blocks": 2,
            "sar_fusion": "gated_additive_multiscale",
            "upsampling": "bilinear_plus_3x3_conv",
            "prediction_target": "direct_x0",
            "bridge_geometry": bridge_geometry,
            "sar_bridge_active": sar_bridge_active,
            "sar_bridge_kappa": (
                float(args.sar_bridge_kappa) if sar_bridge_active else None
            ),
            "sar_residual_coordinate_active": residual_coordinate_active,
            "sar_residual_coordinate_kappa": (
                float(args.sar_residual_coordinate_kappa)
                if residual_coordinate_active
                else None
            ),
            "sar_residual_coordinate_gate": (
                "signed_spatial_1ch_tanh_from_cloudy_s2_plus_s1"
                if residual_coordinate_active
                else None
            ),
            "sar_residual_coordinate_definition": (
                "lambda=alpha+kappa*4*alpha*(1-alpha)*g(y,z)"
                if residual_coordinate_active
                else None
            ),
            "sar_residual_coordinate_direction": (
                "clean_to_cloudy_residual_y_minus_x0"
                if residual_coordinate_active
                else None
            ),
            "sar_bridge_envelope": (
                "4*alpha*(1-alpha)" if sar_bridge_active else None
            ),
            "sar_bridge_curvature_input": (
                "cloudy_s2_13ch_plus_paired_s1_vv_vh_2ch"
                if sar_bridge_active
                else None
            ),
            "sar_bridge_curvature_output": (
                "bounded_13ch_optical_space" if sar_bridge_active else None
            ),
            "sar_bridge_target_access": False if sar_bridge_active else None,
            "sar_bridge_endpoint_preserving": True if sar_bridge_active else None,
            "sar_role": (
                "residual_coordinate_geometry_plus_gated_multiscale_feature_fusion"
                if residual_coordinate_active
                else (
                    "bridge_geometry_plus_gated_multiscale_feature_fusion"
                    if sar_bridge_active
                    else "gated_multiscale_feature_fusion"
                )
            ),
            "bridge_coordinate": "physical_alpha",
            "bridge_parameterization": (
                "alpha=t/T" if not beta_active else "u=t/T; alpha=u^(1/a)"
            ),
            "conditioning_mode": PHYSICAL_ALPHA_CONDITIONING,
            "conditioning_identity": "canonical_physical_alpha_film",
            "conditioning_coordinate": "physical_alpha_scaled_by_T",
            "conditioning_definition": "c=T*alpha; network internally uses alpha=c/T",
            "coordinate_invariant_conditioning": True,
            "split_protocol": PILOT_PROTOCOL,
            "split_seed": PILOT_SEED,
            "split_seed_note": (
                "fixed pilot subset seed; official ROI-disjoint split preserved"
            ),
            "data_profile": "pilot10_roi_season_stratified",
            "pilot_fraction": PILOT_FRACTION,
            "pilot_subset_seed": PILOT_SEED,
            "pilot_manifest_filename": "pilot_subset_manifest.json",
            "pilot_manifest_sample_ids_sha256": pilot_manifest[
                "sample_ids_sha256"
            ],
            "run_kind": (
                "pilot10_sar_residual_coordinate_bridge"
                if residual_coordinate_active
                else (
                    "pilot10_dual_role_sar_bridge"
                    if sar_bridge_active
                    else "pilot10_architecture_independence"
                )
            ),
            "paper_grade": False,
            "endpoint_probability": 0.0,
            "beta_a": beta_a,
            "beta_b": 1.0,
            "curved_validation_nfes": list(args.curved_val_nfes),
        }
    )

    if beta_active:
        metadata.update(legacy_pilot._beta_a1_measure_metadata(beta_a=beta_a))
        metadata["canonical_alpha_direct"] = False
    else:
        metadata.update(
            {
                "training_bridge_measure": "uniform_discrete_alpha_grid",
                "training_bridge_measure_definition": (
                    "t~Uniform{0,...,T}; alpha=t/T; "
                    "q(alpha)=Uniform{0,1/T,...,1}"
                ),
                "alpha_sampling_identity": "canonical_uniform_discrete_alpha",
                "base_measure": "uniform_physical_alpha",
                "continuous_mean_alpha": 0.5,
                "smooth_endpoint_biased_measure": False,
                "deployment_aware_measure": False,
                "canonical_alpha_direct": True,
            }
        )

    config_path = run_dir / "run_config.json"
    if args.resume is None:
        base._write_json_atomic(config_path, metadata)
    else:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"resume requires existing run_config.json: {config_path}"
            )
        previous_metadata = json.loads(config_path.read_text(encoding="utf-8"))
        base.validate_resume_metadata(previous_metadata, metadata)
        base._write_json_atomic(config_path, metadata)

    legacy_pilot._write_or_validate_manifest(
        run_dir,
        pilot_manifest,
        resume=args.resume is not None,
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "device": str(device),
                "model_identity": model_identity,
                "trainable_parameters": trainable_parameters,
                "schedule_name": schedule_name,
                "bridge_geometry": bridge_geometry,
                "sar_bridge_kappa": (
                    float(args.sar_bridge_kappa)
                    if sar_bridge_active
                    else None
                ),
                "sar_residual_coordinate_kappa": (
                    float(args.sar_residual_coordinate_kappa)
                    if residual_coordinate_active
                    else None
                ),
                "conditioning_mode": PHYSICAL_ALPHA_CONDITIONING,
                "training_bridge_measure": metadata["training_bridge_measure"],
                "beta_a": beta_a,
                "epochs": args.epochs,
                "curved_validation_nfes": list(args.curved_val_nfes),
                "train_samples": len(datasets.train),
                "val_samples": len(datasets.val),
                "test_samples": len(datasets.test),
                "paper_grade": False,
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
        schedule_name=schedule_name,
        total_steps=args.total_steps,
        sampler_generator=sampler_generator,
        output_dir=run_dir,
        epochs=args.epochs,
        device=device,
        model_identity=model_identity,
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
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
        curved_validation_nfes=tuple(args.curved_val_nfes),
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
