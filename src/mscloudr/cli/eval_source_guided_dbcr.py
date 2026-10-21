"""NFE=1 full-test evaluation for source-guided DB-CR checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from mscloudr.checkpointing import CHECKPOINT_FORMAT_VERSION
from mscloudr.cli import eval as base
from mscloudr.data import (
    REFERENCE_PROTOCOL,
    build_reference_dataloaders,
    build_reference_datasets,
    discover_sen12mscr,
    load_ignored_sample_ids,
    reference_split_audit,
)
from mscloudr.evaluation import evaluate_nfe1_endpoint
from mscloudr.metrics import REFERENCE_COMMIT, REFERENCE_REPO
from mscloudr.models.source_guided_dbcr import (
    SOURCE_GUIDED_DBCR_IDENTITY,
    SourceGuidedDBCRConfig,
    SourceGuidedDBCRNet,
)
from mscloudr.models.patent_blocks import PatentTimeEmbeddingConfig
from mscloudr.reproducibility import seed_everything


def build_parser():
    parser = base.build_parser()
    parser.description = "Evaluate source-guided DB-CR at NFE=1 on full ROI-disjoint test."
    return parser


def _checkpoint_config(metadata: dict) -> SourceGuidedDBCRConfig:
    if metadata.get("model_identity") != SOURCE_GUIDED_DBCR_IDENTITY:
        raise ValueError("checkpoint is not source-guided DB-CR")
    if metadata.get("split_protocol") != REFERENCE_PROTOCOL:
        raise ValueError("checkpoint split is not frozen ROI-disjoint reference")
    if metadata.get("schedule_name") != "original":
        raise ValueError("source-guided DB-CR checkpoint must use original schedule")
    if metadata.get("conditioning_mode") != "raw_t":
        raise ValueError("source-guided DB-CR checkpoint must use raw_t")
    raw = metadata["architecture_config"]
    time = raw["time_embedding"]
    cfg = SourceGuidedDBCRConfig(
        in_opt=int(raw["in_opt"]),
        in_sar=int(raw["in_sar"]),
        widths=tuple(int(x) for x in raw["widths"]),
        encoder_blocks=tuple(int(x) for x in raw["encoder_blocks"]),
        decoder_blocks=tuple(int(x) for x in raw["decoder_blocks"]),
        sf_heads=tuple(int(x) for x in raw["sf_heads"]),
        total_steps=int(raw["total_steps"]),
        time_embedding=PatentTimeEmbeddingConfig(
            positional_dim=int(time["positional_dim"]),
            hidden_dim=int(time["hidden_dim"]),
        ),
        time_sharing=raw["time_sharing"],
        separate_modality_downsamplers=bool(raw["separate_modality_downsamplers"]),
        stem_kernel_size=int(raw["stem_kernel_size"]),
        output_kernel_size=int(raw["output_kernel_size"]),
    )
    cfg.validate()
    if int(metadata["total_steps"]) != cfg.total_steps:
        raise ValueError("checkpoint total_steps and model configuration disagree")
    return cfg


def run(args):
    base.validate_cli_args(args)
    checkpoint = Path(args.checkpoint)
    payload = base._torch_load_payload(checkpoint)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format version not supported")
    if "model_state" not in payload:
        raise ValueError("checkpoint missing model_state")
    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint missing run_metadata")
    cfg = _checkpoint_config(metadata)
    if metadata.get("paper_grade") is False and not args.allow_smoke_checkpoint:
        raise ValueError("smoke checkpoint requires --allow-smoke-checkpoint")

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
    datasets = build_reference_datasets(
        discovery.samples,
        include_sample_id=True,
        strict_channels=True,
        verify_frozen_membership=True,
    )
    split_audit = reference_split_audit({
        split: dataset.samples for split, dataset in datasets.as_dict().items()
    })
    if metadata.get("split_audit") != split_audit:
        raise ValueError("test dataset split fingerprint differs from checkpoint")

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=train_seed,
        pin_memory=(device.type == "cuda"),
    )

    model = SourceGuidedDBCRNet(cfg)
    count = model.architecture_audit()["trainable_parameters"]
    if count != int(metadata.get("trainable_parameters", -1)):
        raise ValueError("checkpoint parameter count differs from reconstructed architecture")
    model.load_state_dict(payload["model_state"], strict=True)

    paper_grade = bool(metadata.get("paper_grade", False)) and args.max_batches is None
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "model_identity": SOURCE_GUIDED_DBCR_IDENTITY,
        "reproduction_status": metadata.get("reproduction_status"),
        "checkpoint_epoch": payload.get("epoch"),
        "trainable_parameters": count,
        "config": cfg.audit_config(),
        "test_samples": len(datasets.test),
        "nfe": 1,
        "paper_grade": paper_grade,
    }, indent=2, sort_keys=True), flush=True)

    result = evaluate_nfe1_endpoint(
        model,
        loaders.test,
        total_steps=cfg.total_steps,
        device=device,
        max_batches=args.max_batches,
        progress_every=args.progress_every if args.progress_every else None,
        progress_callback=base._progress if args.progress_every else None,
    )
    output = {
        "format_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "checkpoint_metrics": payload.get("metrics", {}),
        "checkpoint_selection_metric": metadata.get("checkpoint_selection_metric"),
        "canonical_checkpoint_filename": checkpoint.name == "best_endpoint.pt",
        "model_identity": SOURCE_GUIDED_DBCR_IDENTITY,
        "reproduction_status": metadata.get("reproduction_status"),
        "trainable_parameters": count,
        "architecture_config": cfg.audit_config(),
        "schedule_name": "original",
        "nfe": 1,
        "inference": "endpoint_direct_x0",
        "endpoint_conditioning_value": cfg.total_steps,
        "schedule_used_during_inference": False,
        "split": "test",
        "split_protocol": REFERENCE_PROTOCOL,
        "split_audit": split_audit,
        "num_samples": result.num_samples,
        "num_batches": result.num_batches,
        "batch_size": int(args.batch_size),
        "max_batches": args.max_batches,
        "paper_grade": paper_grade,
        "metrics": result.metrics,
        "metric_counts": result.metric_counts,
        "metric_reference_repo": REFERENCE_REPO,
        "metric_reference_commit": REFERENCE_COMMIT,
        "prediction_clipping": False,
        "reproducibility": metadata.get("reproducibility"),
    }
    outfile = Path(args.output) if args.output else base._default_output_path(checkpoint)
    base._write_json_atomic(outfile, output, overwrite=args.overwrite)
    print(json.dumps({
        "output": str(outfile),
        "num_samples": result.num_samples,
        "metrics": result.metrics,
    }, indent=2, sort_keys=True), flush=True)
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
