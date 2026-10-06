"""NFE=1 pilot evaluation for CanonicalBridgeNet checkpoints."""

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
    CANONICAL_BRIDGE_MODEL_IDENTITY,
    CanonicalBridgeNet,
    count_canonical_bridge_parameters,
)
from mscloudr.reproducibility import seed_everything
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING


def build_parser():
    parser = base.build_parser()
    parser.description = (
        "Evaluate a CanonicalBridgeNet checkpoint on the fixed 10% "
        "ROI-disjoint SEN12MS-CR pilot test subset."
    )
    return parser


def _validate_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if "model_state" not in payload:
        raise ValueError("checkpoint is missing model_state")

    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing run_metadata")
    if metadata.get("model_identity") != CANONICAL_BRIDGE_MODEL_IDENTITY:
        raise ValueError(
            "expected model_identity=canonical_bridge_net"
        )
    if metadata.get("split_protocol") != PILOT_PROTOCOL:
        raise ValueError("checkpoint does not use the fixed pilot10 split")
    if metadata.get("schedule_name") != "canonical_alpha":
        raise ValueError("canonical bridge pilot requires canonical_alpha")
    if metadata.get("conditioning_mode") != PHYSICAL_ALPHA_CONDITIONING:
        raise ValueError("canonical bridge pilot requires physical_alpha conditioning")
    if metadata.get("bridge_geometry") != "straight_linear":
        raise ValueError("canonical bridge pilot expects straight geometry")
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
    metadata = _validate_checkpoint(payload)

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
    datasets, pilot_manifest = build_pilot_datasets(
        discovery.samples,
        include_sample_id=True,
        strict_channels=True,
        verify_frozen_membership=True,
    )
    split_audit = pilot_split_audit(pilot_manifest)

    if metadata.get("split_audit") != split_audit:
        raise ValueError("current pilot split audit does not match checkpoint")
    if (
        metadata.get("pilot_manifest_sample_ids_sha256")
        != pilot_manifest["sample_ids_sha256"]
    ):
        raise ValueError("pilot sample-ID fingerprint mismatch")

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=train_seed,
        pin_memory=(device.type == "cuda"),
    )

    model = CanonicalBridgeNet(total_steps=int(metadata["total_steps"]))
    trainable_parameters = count_canonical_bridge_parameters(model)
    if trainable_parameters != int(metadata["trainable_parameters"]):
        raise RuntimeError(
            "canonical bridge parameter count does not match checkpoint metadata"
        )
    model.load_state_dict(payload["model_state"], strict=True)

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
        "model_identity": CANONICAL_BRIDGE_MODEL_IDENTITY,
        "model_family": metadata.get("model_family"),
        "trainable_parameters": trainable_parameters,
        "dbcr_specific_blocks": metadata.get("dbcr_specific_blocks"),
        "schedule_name": metadata["schedule_name"],
        "training_bridge_measure": metadata.get("training_bridge_measure"),
        "beta_a": metadata.get("beta_a"),
        "beta_b": metadata.get("beta_b"),
        "conditioning_mode": metadata["conditioning_mode"],
        "conditioning_identity": metadata.get("conditioning_identity"),
        "bridge_geometry": metadata.get("bridge_geometry"),
        "total_steps": int(metadata["total_steps"]),
        "nfe": 1,
        "inference": "endpoint_direct_x0",
        "endpoint_conditioning_value": int(metadata["total_steps"]),
        "schedule_used_during_inference": False,
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
                "model_identity": CANONICAL_BRIDGE_MODEL_IDENTITY,
                "beta_a": metadata.get("beta_a"),
                "num_samples": result.num_samples,
                "metrics": result.metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return output_payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
