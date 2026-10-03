"""Fast 10%-data / 25-epoch DB-CR pilot training command.

This command leaves the audited full-data trainer untouched.  It reuses the
same model, optimizer, bridge schedules, checkpointing, and validation logic,
but swaps the full SEN12MS-CR datasets for the deterministic pilot subset from
``mscloudr.data.pilot``.
"""

from __future__ import annotations

import json
from pathlib import Path

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


DEFAULT_PILOT_EPOCHS = 25


def build_parser():
    parser = base.build_parser()
    parser.description = (
        "Train the frozen legacy DB-CR backbone on the deterministic 10% "
        "ROI-disjoint, season/ROI-stratified SEN12MS-CR pilot subset."
    )
    parser.set_defaults(epochs=DEFAULT_PILOT_EPOCHS)
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


def run(args):
    base.validate_cli_args(args)
    run_dir = base._prepare_output_dir(args)
    device = base.resolve_device(args.device)
    schedule_name, schedule, mr_rate = base.canonical_schedule(args.schedule)

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
