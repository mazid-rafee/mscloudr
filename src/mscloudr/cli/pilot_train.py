"""Fast 10%-data / 25-epoch DB-CR pilot training command.

This command leaves the audited full-data trainer untouched. It reuses the
same model, optimizer, checkpointing, validation logic, and deterministic
pilot subset while allowing controlled bridge-parameterization experiments.

The canonical_alpha option uses alpha=t/T with the historical discrete
uniform timestep sampler. Therefore t~Uniform{0,...,T} induces a uniform
training measure over the physical corruption grid {0,1/T,...,1}.

The ``--conditioning physical_alpha`` option keeps a schedule's bridge-state
sampling measure unchanged but replaces schedule-dependent raw-t conditioning
with the canonical physical coordinate c=T*alpha.

The ``--endpoint-prob`` pilot control deliberately changes only the training
measure in physical-alpha space. With canonical_alpha + physical_alpha, a
uniform base variable u=t/T is transformed by the quantile map

    alpha = u/(1-lambda),  u < 1-lambda
            1,             u >= 1-lambda

which realizes q_lambda=(1-lambda)U(0,1)+lambda*delta_1 up to the finite
{0,...,T} grid.

The ``--beta-a`` control provides a smooth endpoint-biased alternative. With
canonical_alpha + physical_alpha and beta_b fixed to 1, it uses

    alpha = u^(1/a),  u=t/T,

which converges to alpha~Beta(a,1) with density q(alpha)=a*alpha^(a-1).
The first intended experiment uses a=2, giving q(alpha)=2*alpha.

Endpoint checkpoint selection remains val_endpoint_l1 for every measure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from mscloudr.bridge import get_bridge_schedule
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
    save_pilot_manifest,
)
from mscloudr.models import (
    LEGACY_DBCR_PARAMETER_COUNT,
    LegacyDBCRNet,
    count_legacy_parameters,
)
from mscloudr.reproducibility import make_torch_generator, seed_everything
from mscloudr.runner import fit
from mscloudr.training import (
    PHYSICAL_ALPHA_CONDITIONING,
    RAW_T_CONDITIONING,
    SUPPORTED_CONDITIONING_MODES,
)


DEFAULT_PILOT_EPOCHS = 25


def build_parser():
    parser = base.build_parser()
    parser.description = (
        "Train the frozen legacy DB-CR backbone on the deterministic 10% "
        "ROI-disjoint, season/ROI-stratified SEN12MS-CR pilot subset."
    )
    parser.set_defaults(epochs=DEFAULT_PILOT_EPOCHS)

    # The audited full-data CLI intentionally exposes only its historical
    # schedules. The pilot branch adds canonical_alpha without modifying that
    # paper-grade command.
    schedule_action = parser._option_string_actions["--schedule"]
    schedule_action.choices = (
        "original",
        "mr_r3",
        "canonical_alpha",
    )
    parser.add_argument(
        "--conditioning",
        choices=SUPPORTED_CONDITIONING_MODES,
        default=RAW_T_CONDITIONING,
        help=(
            "Coordinate passed to the legacy time embedding. raw_t reproduces "
            "historical DB-CR. physical_alpha passes c=T*alpha while leaving "
            "the selected schedule's bridge-state sampling measure unchanged."
        ),
    )
    parser.add_argument(
        "--endpoint-prob",
        type=float,
        default=0.0,
        help=(
            "Endpoint-mixture weight lambda. A positive value requires "
            "--schedule canonical_alpha --conditioning physical_alpha and "
            "uses q_lambda=(1-lambda)U(0,1)+lambda*delta_1."
        ),
    )
    parser.add_argument(
        "--beta-a",
        type=float,
        default=1.0,
        help=(
            "Shape a for the smooth Beta(a,1) physical-alpha measure. "
            "a=1 reproduces uniform alpha. a!=1 requires --schedule "
            "canonical_alpha --conditioning physical_alpha and is mutually "
            "exclusive with --endpoint-prob > 0."
        ),
    )
    return parser


def _write_or_validate_manifest(run_dir: Path, manifest: dict, *, resume: bool) -> None:
    path = run_dir / "pilot_subset_manifest.json"
    if resume:
        if not path.is_file():
            raise FileNotFoundError(
                f"resume requires existing pilot subset manifest: {path}"
            )
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                "current pilot subset manifest does not match the resumed run"
            )
        return
    save_pilot_manifest(manifest, path)


def _resolve_pilot_schedule(name: str):
    if name == "canonical_alpha":
        return (
            "canonical_alpha",
            get_bridge_schedule("canonical_alpha"),
            None,
        )
    return base.canonical_schedule(name)


def _conditioning_metadata(mode: str) -> dict:
    if mode == RAW_T_CONDITIONING:
        return {
            "conditioning_mode": RAW_T_CONDITIONING,
            "conditioning_identity": "legacy_raw_t_stage_bias",
            "conditioning_coordinate": "schedule_parameter_t",
            "conditioning_definition": "c=t",
            "coordinate_invariant_conditioning": False,
        }
    if mode == PHYSICAL_ALPHA_CONDITIONING:
        return {
            "conditioning_mode": PHYSICAL_ALPHA_CONDITIONING,
            "conditioning_identity": "physical_alpha_scaled_T_stage_bias",
            "conditioning_coordinate": "physical_alpha_scaled_by_T",
            "conditioning_definition": "c=T*alpha",
            "coordinate_invariant_conditioning": True,
        }
    raise ValueError(f"unsupported conditioning mode: {mode!r}")


def _validate_endpoint_measure_args(
    schedule_name: str,
    conditioning_mode: str,
    endpoint_probability: float,
) -> float:
    """Backward-compatible validator for the endpoint-mixture control."""

    endpoint_probability = float(endpoint_probability)
    if not 0.0 <= endpoint_probability < 1.0:
        raise ValueError("endpoint-prob must satisfy 0 <= lambda < 1")
    if endpoint_probability > 0.0:
        if schedule_name != "canonical_alpha":
            raise ValueError(
                "endpoint-prob > 0 requires --schedule canonical_alpha"
            )
        if conditioning_mode != PHYSICAL_ALPHA_CONDITIONING:
            raise ValueError(
                "endpoint-prob > 0 requires --conditioning physical_alpha"
            )
    return endpoint_probability


def _validate_measure_args(
    schedule_name: str,
    conditioning_mode: str,
    *,
    endpoint_probability: float,
    beta_a: float,
) -> tuple[float, float]:
    endpoint_probability = _validate_endpoint_measure_args(
        schedule_name,
        conditioning_mode,
        endpoint_probability,
    )
    beta_a = float(beta_a)
    if beta_a <= 0.0:
        raise ValueError("beta-a must be positive")

    beta_active = not math.isclose(beta_a, 1.0, rel_tol=0.0, abs_tol=1e-12)
    if endpoint_probability > 0.0 and beta_active:
        raise ValueError(
            "endpoint-prob and beta-a != 1 are mutually exclusive measure controls"
        )
    if beta_active:
        if schedule_name != "canonical_alpha":
            raise ValueError(
                "beta-a != 1 requires --schedule canonical_alpha"
            )
        if conditioning_mode != PHYSICAL_ALPHA_CONDITIONING:
            raise ValueError(
                "beta-a != 1 requires --conditioning physical_alpha"
            )

    return endpoint_probability, beta_a


def _endpoint_mixture_schedule(endpoint_probability: float):
    """Quantile map from uniform discrete t to an endpoint-aware alpha measure."""

    endpoint_probability = float(endpoint_probability)
    if not 0.0 < endpoint_probability < 1.0:
        raise ValueError("endpoint mixture schedule requires 0 < lambda < 1")
    cutoff = 1.0 - endpoint_probability

    def schedule(t, total_steps: int):
        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        t_float = t if torch.is_tensor(t) else torch.as_tensor(t)
        if not t_float.is_floating_point():
            t_float = t_float.float()
        u = t_float / float(total_steps)
        return torch.where(
            u < cutoff,
            u / cutoff,
            torch.ones_like(u),
        )

    return schedule


def _beta_a1_schedule(beta_a: float):
    """Quantile map from uniform discrete t to a Beta(a,1) alpha measure."""

    beta_a = float(beta_a)
    if beta_a <= 0.0:
        raise ValueError("beta-a must be positive")
    inverse_a = 1.0 / beta_a

    def schedule(t, total_steps: int):
        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        t_float = t if torch.is_tensor(t) else torch.as_tensor(t)
        if not t_float.is_floating_point():
            t_float = t_float.float()
        u = torch.clamp(t_float / float(total_steps), 0.0, 1.0)
        return torch.pow(u, inverse_a)

    return schedule


def _endpoint_measure_metadata(
    *,
    endpoint_probability: float,
    total_steps: int,
) -> dict:
    endpoint_probability = float(endpoint_probability)
    total_steps = int(total_steps)
    if endpoint_probability <= 0.0:
        return {}

    first_endpoint_t = int(math.ceil((1.0 - endpoint_probability) * total_steps))
    endpoint_grid_count = total_steps - first_endpoint_t + 1
    realized_endpoint_mass = endpoint_grid_count / float(total_steps + 1)

    return {
        "training_bridge_measure": "endpoint_mixture_uniform_alpha",
        "training_bridge_measure_definition": (
            "q_lambda(alpha)=(1-lambda)U(0,1)+lambda*delta(alpha=1); "
            "implemented by quantile transform of uniform discrete t"
        ),
        "alpha_sampling_identity": "endpoint_mixture_quantile_transform",
        "endpoint_probability": endpoint_probability,
        "endpoint_probability_requested": endpoint_probability,
        "endpoint_probability_realized_discrete": realized_endpoint_mass,
        "endpoint_first_discrete_t": first_endpoint_t,
        "endpoint_grid_count": endpoint_grid_count,
        "base_measure": "uniform_physical_alpha",
        "deployment_aware_measure": True,
    }


def _beta_a1_measure_metadata(*, beta_a: float) -> dict:
    beta_a = float(beta_a)
    if beta_a <= 0.0:
        raise ValueError("beta-a must be positive")

    return {
        "training_bridge_measure": "beta_a1_physical_alpha",
        "training_bridge_measure_definition": (
            "u=t/T with t~Uniform{0,...,T}; alpha=u^(1/a); "
            "continuous-limit alpha~Beta(a,1)"
        ),
        "training_bridge_measure_density": "q(alpha)=a*alpha^(a-1)",
        "alpha_sampling_identity": "beta_a1_quantile_transform",
        "beta_a": beta_a,
        "beta_b": 1.0,
        "base_measure": "uniform_physical_alpha",
        "smooth_endpoint_biased_measure": beta_a > 1.0,
        "deployment_aware_measure": beta_a > 1.0,
        "continuous_mean_alpha": beta_a / (beta_a + 1.0),
    }


def run(args):
    base.validate_cli_args(args)
    run_dir = base._prepare_output_dir(args)
    device = base.resolve_device(args.device)
    schedule_name, schedule, mr_rate = _resolve_pilot_schedule(args.schedule)
    endpoint_probability, beta_a = _validate_measure_args(
        schedule_name,
        args.conditioning,
        endpoint_probability=args.endpoint_prob,
        beta_a=args.beta_a,
    )
    beta_active = not math.isclose(beta_a, 1.0, rel_tol=0.0, abs_tol=1e-12)
    if endpoint_probability > 0.0:
        schedule = _endpoint_mixture_schedule(endpoint_probability)
    elif beta_active:
        schedule = _beta_a1_schedule(beta_a)

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

    model = LegacyDBCRNet()
    trainable_parameters = count_legacy_parameters(model)
    if trainable_parameters != LEGACY_DBCR_PARAMETER_COUNT:
        raise RuntimeError(
            "legacy_dbcr parameter count drifted: "
            f"expected {LEGACY_DBCR_PARAMETER_COUNT}, got {trainable_parameters}"
        )

    metadata = base._run_metadata(
        args=args,
        dataset_root=discovery.dataset_root,
        split_audit=split_audit,
        trainable_parameters=trainable_parameters,
        schedule_name=schedule_name,
        mean_reversion_rate=mr_rate,
        device=device,
    )
    metadata["reproducibility"]["seeds"]["split_seed"] = PILOT_SEED
    metadata.update(
        {
            "split_protocol": PILOT_PROTOCOL,
            "split_seed": PILOT_SEED,
            "split_seed_note": (
                "fixed pilot subset seed; train/val/test ROI membership remains "
                "the official UnCRtainTS ROI-disjoint split"
            ),
            "data_profile": "pilot10_roi_season_stratified",
            "pilot_fraction": PILOT_FRACTION,
            "pilot_subset_seed": PILOT_SEED,
            "pilot_manifest_filename": "pilot_subset_manifest.json",
            "pilot_manifest_sample_ids_sha256": pilot_manifest[
                "sample_ids_sha256"
            ],
            "run_kind": "pilot10",
            "paper_grade": False,
        }
    )
    metadata.update(_conditioning_metadata(args.conditioning))

    if schedule_name == "canonical_alpha":
        metadata.update(
            {
                "bridge_coordinate": "physical_alpha",
                "bridge_parameterization": "alpha=t/T",
                "training_bridge_measure": "uniform_discrete_alpha_grid",
                "training_bridge_measure_definition": (
                    "t~Uniform{0,...,T}; alpha=t/T; "
                    "q(alpha)=Uniform{0,1/T,...,1}"
                ),
                "canonical_alpha_direct": True,
            }
        )
    else:
        metadata.update(
            {
                "bridge_coordinate": "schedule_induced_alpha",
                "training_bridge_measure": "uniform_discrete_t",
                "training_bridge_measure_definition": (
                    "t~Uniform{0,...,T}; alpha=schedule(t)"
                ),
                "canonical_alpha_direct": False,
            }
        )

    if endpoint_probability > 0.0:
        metadata.update(
            {
                "bridge_coordinate": "physical_alpha",
                "bridge_parameterization": (
                    "u=t/T; alpha=u/(1-lambda) if u<1-lambda else 1"
                ),
                "canonical_alpha_direct": False,
                "beta_a": 1.0,
                "beta_b": 1.0,
                "smooth_endpoint_biased_measure": False,
            }
        )
        metadata.update(
            _endpoint_measure_metadata(
                endpoint_probability=endpoint_probability,
                total_steps=args.total_steps,
            )
        )
    elif beta_active:
        metadata.update(
            {
                "bridge_coordinate": "physical_alpha",
                "bridge_parameterization": "u=t/T; alpha=u^(1/a)",
                "canonical_alpha_direct": False,
                "endpoint_probability": 0.0,
                "endpoint_probability_realized_discrete": None,
            }
        )
        metadata.update(_beta_a1_measure_metadata(beta_a=beta_a))
    else:
        metadata.update(
            {
                "alpha_sampling_identity": "schedule_default_uniform_discrete_t",
                "endpoint_probability": 0.0,
                "endpoint_probability_realized_discrete": None,
                "beta_a": 1.0,
                "beta_b": 1.0,
                "smooth_endpoint_biased_measure": False,
                "deployment_aware_measure": False,
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

    _write_or_validate_manifest(
        run_dir,
        pilot_manifest,
        resume=args.resume is not None,
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "device": str(device),
                "model_identity": "legacy_dbcr",
                "trainable_parameters": trainable_parameters,
                "schedule_name": schedule_name,
                "conditioning_mode": args.conditioning,
                "conditioning_coordinate": metadata["conditioning_coordinate"],
                "bridge_coordinate": metadata["bridge_coordinate"],
                "training_bridge_measure": metadata[
                    "training_bridge_measure"
                ],
                "alpha_sampling_identity": metadata["alpha_sampling_identity"],
                "endpoint_probability": metadata["endpoint_probability"],
                "endpoint_probability_realized_discrete": metadata.get(
                    "endpoint_probability_realized_discrete"
                ),
                "beta_a": metadata.get("beta_a"),
                "beta_b": metadata.get("beta_b"),
                "continuous_mean_alpha": metadata.get("continuous_mean_alpha"),
                "data_profile": metadata["data_profile"],
                "pilot_fraction": PILOT_FRACTION,
                "pilot_subset_seed": PILOT_SEED,
                "epochs": args.epochs,
                "train_samples": len(datasets.train),
                "val_samples": len(datasets.val),
                "test_samples": len(datasets.test),
                "season_counts": {
                    split: split_audit["splits"][split]["season_counts"]
                    for split in ("train", "val", "test")
                },
                "roi_groups": {
                    split: split_audit["splits"][split]["num_roi_groups"]
                    for split in ("train", "val", "test")
                },
                "paper_grade": False,
                "progress_every": args.progress_every,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    sampler_generator = make_torch_generator(
        args.sampler_seed,
        device="cpu",
    )

    history = fit(
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
        model_identity="legacy_dbcr",
        conditioning_mode=args.conditioning,
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
    return history


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
