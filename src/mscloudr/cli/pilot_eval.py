"""Evaluation command for checkpoints trained with the fixed 10% pilot subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mscloudr.checkpointing import CHECKPOINT_FORMAT_VERSION
from mscloudr.cli import eval as base
from mscloudr.data import (
    build_reference_dataloaders,
    discover_sen12mscr,
    load_ignored_sample_ids,
)
from mscloudr.data.pilot import (
    PILOT_PROTOCOL,
    build_pilot_datasets,
    pilot_split_audit,
)
from mscloudr.evaluation import EvaluationResult, evaluate_nfe1_endpoint
from mscloudr.metrics import REFERENCE_COMMIT, REFERENCE_REPO
from mscloudr.models import (
    LEGACY_DBCR_PARAMETER_COUNT,
    LegacyDBCRNet,
    count_legacy_parameters,
)
from mscloudr.reproducibility import seed_everything
from mscloudr.training import RAW_T_CONDITIONING, SUPPORTED_CONDITIONING_MODES


def build_parser():
    parser = base.build_parser()
    parser.description = (
        "Evaluate a legacy DB-CR checkpoint on the deterministic 10% "
        "ROI-disjoint, season/ROI-stratified SEN12MS-CR pilot test subset."
    )
    return parser


def _conditioning_mode(metadata: dict[str, Any]) -> str:
    mode = metadata.get("conditioning_mode", RAW_T_CONDITIONING)
    if mode not in SUPPORTED_CONDITIONING_MODES:
        raise ValueError(f"unsupported conditioning mode in checkpoint: {mode!r}")
    return str(mode)


def _geometry_metadata(metadata: dict[str, Any]) -> tuple[str, float]:
    geometry = str(metadata.get("bridge_geometry", "straight_linear"))
    kappa = float(metadata.get("geometry_kappa", 0.0))
    return geometry, kappa


def _validate_pilot_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if "model_state" not in payload:
        raise ValueError("checkpoint is missing model_state")

    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing run_metadata")
    if metadata.get("model_identity") != "legacy_dbcr":
        raise ValueError(
            "pilot evaluation currently supports only model_identity=legacy_dbcr"
        )
    if metadata.get("split_protocol") != PILOT_PROTOCOL:
        raise ValueError(
            "checkpoint was not trained with the fixed pilot10 split protocol"
        )
    if metadata.get("schedule_name") not in {
        "original",
        "mr_r3",
        "canonical_alpha",
    }:
        raise ValueError(
            "unsupported controlled schedule identity in checkpoint metadata"
        )
    _conditioning_mode(metadata)
    _geometry_metadata(metadata)
    if int(metadata.get("total_steps", 0)) <= 0:
        raise ValueError("checkpoint metadata has invalid total_steps")
    return metadata


def _default_output_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "pilot_eval_metrics.json"
    return checkpoint_path.with_name(
        checkpoint_path.stem + "_pilot_eval_metrics.json"
    )


def run(args) -> dict[str, Any]:
    base.validate_cli_args(args)

    checkpoint_path = Path(args.checkpoint)
    payload = base._torch_load_payload(checkpoint_path)
    metadata = _validate_pilot_checkpoint(payload)
    conditioning_mode = _conditioning_mode(metadata)
    bridge_geometry, geometry_kappa = _geometry_metadata(metadata)

    device = base.resolve_device(args.device)
    train_seed = base._train_seed_from_metadata(metadata)
    seed_everything(
        train_seed,
        deterministic_algorithms=base._deterministic_algorithms_from_metadata(
            metadata
        ),
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

    if metadata.get("split_audit") != split_audit:
        raise ValueError(
            "current pilot subset audit does not match checkpoint provenance"
        )
    if (
        metadata.get("pilot_manifest_sample_ids_sha256")
        != pilot_manifest["sample_ids_sha256"]
    ):
        raise ValueError(
            "current pilot sample-ID fingerprint does not match checkpoint provenance"
        )

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=train_seed,
        pin_memory=(device.type == "cuda"),
    )

    model = LegacyDBCRNet()
    trainable_parameters = count_legacy_parameters(model)
    if trainable_parameters != LEGACY_DBCR_PARAMETER_COUNT:
        raise RuntimeError(
            "legacy_dbcr parameter count drifted before pilot evaluation"
        )
    model.load_state_dict(payload["model_state"], strict=True)

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(payload.get("epoch", 0)),
                "device": str(device),
                "model_identity": "legacy_dbcr",
                "schedule_name": metadata["schedule_name"],
                "conditioning_mode": conditioning_mode,
                "bridge_geometry": bridge_geometry,
                "geometry_kappa": geometry_kappa,
                "split": "test",
                "data_profile": metadata.get("data_profile"),
                "test_samples": len(datasets.test),
                "nfe": 1,
                "inference": "endpoint_direct_x0",
                "endpoint_conditioning_value": int(metadata["total_steps"]),
                "paper_grade": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    # At alpha=1, raw-t and physical-alpha conditioning both equal T, and the
    # controlled curvature envelope is exactly zero. The canonical endpoint
    # evaluator therefore remains exactly valid for straight and curved runs.
    result: EvaluationResult = evaluate_nfe1_endpoint(
        model,
        loaders.test,
        total_steps=int(metadata["total_steps"]),
        device=device,
        max_batches=args.max_batches,
        progress_every=(
            args.progress_every if args.progress_every > 0 else None
        ),
        progress_callback=(
            base._progress if args.progress_every > 0 else None
        ),
    )

    output_payload: dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "checkpoint_metrics": payload.get("metrics", {}),
        "checkpoint_selection_metric": metadata.get(
            "checkpoint_selection_metric"
        ),
        "canonical_checkpoint_filename": (
            checkpoint_path.name == "best_endpoint.pt"
        ),
        "model_identity": metadata["model_identity"],
        "schedule_name": metadata["schedule_name"],
        "conditioning_mode": conditioning_mode,
        "conditioning_identity": metadata.get("conditioning_identity"),
        "bridge_geometry": bridge_geometry,
        "geometry_kappa": geometry_kappa,
        "geometry_definition": metadata.get("geometry_definition"),
        "total_steps": int(metadata["total_steps"]),
        "nfe": 1,
        "inference": "endpoint_direct_x0",
        "endpoint_conditioning_value": int(metadata["total_steps"]),
        "schedule_used_during_inference": False,
        "curved_geometry_used_during_inference": False,
        "split": "test",
        "split_protocol": PILOT_PROTOCOL,
        "split_audit": split_audit,
        "pilot_manifest_sample_ids_sha256": pilot_manifest[
            "sample_ids_sha256"
        ],
        "num_samples": result.num_samples,
        "num_batches": result.num_batches,
        "batch_size": int(args.batch_size),
        "max_batches": args.max_batches,
        "paper_grade": False,
        "metrics": result.metrics,
        "metric_counts": result.metric_counts,
        "metric_reference_repo": REFERENCE_REPO,
        "metric_reference_commit": REFERENCE_COMMIT,
        "prediction_clipping": False,
    }

    output_path = (
        Path(args.output)
        if args.output is not None
        else _default_output_path(checkpoint_path)
    )
    base._write_json_atomic(
        output_path,
        output_payload,
        overwrite=args.overwrite,
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "num_samples": result.num_samples,
                "metrics": result.metrics,
                "paper_grade": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    return output_payload


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
