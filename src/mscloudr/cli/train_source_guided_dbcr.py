"""Train source-guided DB-CR with the audited ROI-disjoint SEN12MS-CR protocol.

New model identity; NEVER replaces the frozen legacy DB-CR baseline.
Training uses the original sine bridge, random-t L1, and selects the
best checkpoint by validation endpoint L1 for NFE=1 evaluation.
"""

from __future__ import annotations

import argparse
import json

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
from mscloudr.models.source_guided_dbcr import (
    SOURCE_GUIDED_DBCR_IDENTITY,
    SourceGuidedDBCRConfig,
    SourceGuidedDBCRNet,
    SOURCE_GUIDED_SCALE_SHIFT_MODES,
)
from mscloudr.models.patent_blocks import PatentTimeEmbeddingConfig
from mscloudr.reproducibility import (
    SeedConfig,
    make_torch_generator,
    reproducibility_metadata,
    seed_everything,
)
from mscloudr.runner import fit
from mscloudr.training import RAW_T_CONDITIONING


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = (
        "Train source-guided DB-CR (not authors' official code) "
        "on frozen UnCRtainTS ROI-disjoint SEN12MS-CR."
    )
    action = parser._option_string_actions["--schedule"]
    action.required = False
    action.choices = ("original",)
    action.default = "original"
    parser.add_argument(
        "--time-sharing",
        choices=("per_block", "per_width"),
        default="per_block",
        help="Explicit assumption; not specified by DB-CR paper/patent.",
    )
    parser.add_argument(
        "--scale-shift-mode",
        choices=SOURCE_GUIDED_SCALE_SHIFT_MODES,
        default="patent",
        help=(
            "patent: x*scale+shift (source-guided default); "
            "identity_centered: x*(1+scale)+shift everywhere; "
            "encoder_only_identity_centered: identity-centered ONLY in optical "
            "and SAR encoders; patent modulation in bottleneck and decoders"
        ),
    )
    parser.add_argument("--positional-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    return parser


def _architecture(args) -> SourceGuidedDBCRConfig:
    cfg = SourceGuidedDBCRConfig(
        total_steps=args.total_steps,
        time_sharing=args.time_sharing,
        scale_shift_mode=args.scale_shift_mode,
        time_embedding=PatentTimeEmbeddingConfig(
            positional_dim=args.positional_dim,
            hidden_dim=args.hidden_dim,
        ),
    )
    cfg.validate()
    return cfg


def _resume_identity(metadata: dict) -> dict:
    fields = (
        "model_identity",
        "trainable_parameters",
        "architecture_config",
        "schedule_name",
        "total_steps",
        "batch_size",
        "num_workers",
        "learning_rate",
        "split_protocol",
        "split_audit",
        "ignore_file",
    )
    result = {key: metadata[key] for key in fields}
    result["seeds"] = metadata["reproducibility"]["seeds"]
    result["deterministic_algorithms"] = metadata["reproducibility"][
        "deterministic_algorithms"
    ]
    return result


def run(args: argparse.Namespace):
    base.validate_cli_args(args)
    if args.schedule != "original":
        raise ValueError("source-guided DB-CR baseline uses the original sine bridge")
    cfg = _architecture(args)
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
    split_audit = reference_split_audit({
        split: dataset.samples
        for split, dataset in datasets.as_dict().items()
    })
    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=args.train_seed,
        pin_memory=(device.type == "cuda"),
    )
    model = SourceGuidedDBCRNet(cfg)
    architecture_audit = model.architecture_audit()

    seed_metadata = reproducibility_metadata(
        SeedConfig(split_seed=0, train_seed=args.train_seed, sampler_seed=args.sampler_seed),
        deterministic_algorithms=args.deterministic_algorithms,
    )
    seed_metadata["seeds"]["split_seed"] = None
    metadata = {
        "format_version": 1,
        "model_identity": SOURCE_GUIDED_DBCR_IDENTITY,
        "reproduction_status": "source_guided_not_authors_official",
        "trainable_parameters": architecture_audit["trainable_parameters"],
        "published_parameter_count": architecture_audit["published_dbcr_parameters"],
        "architecture_config": cfg.audit_config(),
        "architecture_audit": architecture_audit,
        "schedule_name": "original",
        "conditioning_mode": RAW_T_CONDITIONING,
        "conditioning_identity": {
            "patent": "patent_alpha_sine_scale_shift_per_nafblock",
            "identity_centered": (
                "experimental_identity_centered_alpha_sine_scale_shift_per_nafblock"
            ),
            "encoder_only_identity_centered": (
                "experimental_encoder_only_identity_centered_alpha_sine_scale_shift"
            ),
        }[cfg.scale_shift_mode],
        "experiment_variant": cfg.scale_shift_mode,
        "bridge_geometry": "straight_linear",
        "bridge_state_definition": "x_t=(1-alpha_t)*x0+alpha_t*y",
        "bridge_alpha_definition": "alpha_t=sin(pi*t/(2*T))",
        "prediction_target": "direct_x0",
        "total_steps": args.total_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.lr,
        "checkpoint_selection_metric": "val_endpoint_l1",
        "split_protocol": REFERENCE_PROTOCOL,
        "split_audit": split_audit,
        "split_seed": None,
        "ignore_file": args.ignore_file,
        "data_root": str(discovery.dataset_root),
        "device": str(device),
        "run_kind": "smoke" if args.smoke_run else "full",
        "paper_grade": not args.smoke_run,
        "reproducibility": seed_metadata,
    }
    config_path = run_dir / "run_config.json"
    if args.resume is not None:
        if not config_path.is_file():
            raise FileNotFoundError("cannot resume without existing run_config.json")
        original = json.loads(config_path.read_text(encoding="utf-8"))
        if _resume_identity(original) != _resume_identity(metadata):
            raise ValueError("resume settings differ from source-guided DB-CR checkpoint")
    base._write_json_atomic(config_path, metadata)

    print(json.dumps({
        "run_dir": str(run_dir),
        "model_identity": SOURCE_GUIDED_DBCR_IDENTITY,
        "trainable_parameters": architecture_audit["trainable_parameters"],
        "paper_reported_parameters": architecture_audit["published_dbcr_parameters"],
        "architecture_config": cfg.audit_config(),
        "train_samples": len(datasets.train),
        "validation_samples": len(datasets.val),
        "test_samples": len(datasets.test),
        "split": REFERENCE_PROTOCOL,
        "run_kind": metadata["run_kind"],
    }, indent=2, sort_keys=True), flush=True)

    return fit(
        model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        schedule=get_bridge_schedule("original"),
        schedule_name="original",
        total_steps=args.total_steps,
        sampler_generator=make_torch_generator(args.sampler_seed, device="cpu"),
        output_dir=run_dir,
        epochs=args.epochs,
        device=device,
        model_identity=SOURCE_GUIDED_DBCR_IDENTITY,
        conditioning_mode=RAW_T_CONDITIONING,
        lr=args.lr,
        run_metadata=metadata,
        resume_from=args.resume,
        epoch_callback=base._epoch_progress,
        batch_progress_callback=base._batch_progress if args.progress_every else None,
        progress_every=args.progress_every if args.progress_every else None,
        max_train_batches=args.smoke_train_batches if args.smoke_run else None,
        max_val_batches=args.smoke_val_batches if args.smoke_run else None,
    )


def main() -> None:
    args = build_parser().parse_args()
    history = run(args)
    print(json.dumps(history[-1].to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
