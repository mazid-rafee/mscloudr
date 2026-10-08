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
    ReferenceDatasets,
    build_reference_dataloaders,
    discover_sen12mscr,
    load_ignored_sample_ids,
)
from mscloudr.data.ablation import ZeroSARDataset
from mscloudr.data.pilot import (
    PILOT_FRACTION,
    PILOT_PROTOCOL,
    PILOT_SEED,
    build_pilot_datasets,
    pilot_split_audit,
)
from mscloudr.models import (
    CANONICAL_BRIDGE_MODEL_IDENTITY,
    CANONICAL_BRIDGE_OPTICAL_ONLY_MODEL_IDENTITY,
    CanonicalBridgeNet,
    CanonicalBridgeOpticalOnlyNet,
    count_canonical_bridge_optical_only_parameters,
    count_canonical_bridge_parameters,
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
        "--sar-input-mode",
        choices=("paired", "zero", "optical_only_13"),
        default="paired",
        help=(
            "paired uses aligned Sentinel-1; zero replaces SAR with exact "
            "zeros for a capacity-matched no-SAR-information control; "
            "optical_only_13 removes the SAR pathway entirely."
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

    if args.sar_input_mode == "zero":
        datasets = ReferenceDatasets(
            train=ZeroSARDataset(datasets.train),
            val=ZeroSARDataset(datasets.val),
            test=ZeroSARDataset(datasets.test),
        )

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=args.train_seed,
        pin_memory=(device.type == "cuda"),
    )

    if args.sar_input_mode == "optical_only_13":
        model = CanonicalBridgeOpticalOnlyNet(total_steps=args.total_steps)
        model_identity = CANONICAL_BRIDGE_OPTICAL_ONLY_MODEL_IDENTITY
        trainable_parameters = count_canonical_bridge_optical_only_parameters(
            model
        )
    else:
        model = CanonicalBridgeNet(total_steps=args.total_steps)
        model_identity = CANONICAL_BRIDGE_MODEL_IDENTITY
        trainable_parameters = count_canonical_bridge_parameters(model)

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
            "model_family": (
                "single_stream_residual_unet"
                if args.sar_input_mode == "optical_only_13"
                else "dual_stream_residual_unet"
            ),
            "dbcr_specific_blocks": False,
            "naf_blocks": False,
            "sf_blocks": False,
            "architecture_widths": [32, 64, 128, 256],
            "architecture_mid_blocks": 2,
            "sar_fusion": (
                "none"
                if args.sar_input_mode == "optical_only_13"
                else "gated_additive_multiscale"
            ),
            "upsampling": "bilinear_plus_3x3_conv",
            "prediction_target": "direct_x0",
            "bridge_geometry": "straight_linear",
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
                "pilot10_true_optical_only_13ch"
                if args.sar_input_mode == "optical_only_13"
                else (
                    "pilot10_no_sar_information_capacity_matched"
                    if args.sar_input_mode == "zero"
                    else "pilot10_architecture_independence"
                )
            ),
            "sar_input_mode": args.sar_input_mode,
            "sar_information_available": args.sar_input_mode == "paired",
            "optical_only_control": args.sar_input_mode in {
                "zero",
                "optical_only_13",
            },
            "true_optical_only_13ch": (
                args.sar_input_mode == "optical_only_13"
            ),
            "capacity_matched_to_multimodal": args.sar_input_mode != "optical_only_13",
            "input_modalities": (
                ["sentinel2_bridge_state_13ch"]
                if args.sar_input_mode == "optical_only_13"
                else ["sentinel2_bridge_state_13ch", "sentinel1_sar_2ch"]
            ),
            "paper_grade": False,
            "endpoint_probability": 0.0,
            "beta_a": beta_a,
            "beta_b": 1.0,
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
                "bridge_geometry": "straight_linear",
                "conditioning_mode": PHYSICAL_ALPHA_CONDITIONING,
                "training_bridge_measure": metadata["training_bridge_measure"],
                "sar_input_mode": args.sar_input_mode,
                "beta_a": beta_a,
                "epochs": args.epochs,
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
