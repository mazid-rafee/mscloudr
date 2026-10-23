"""Full ROI-disjoint NFE=1 test of component-supervised CanonicalBridgeNet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mscloudr.checkpointing import CHECKPOINT_FORMAT_VERSION
from mscloudr.cli import eval as base
from mscloudr.data import (
    REFERENCE_PROTOCOL, build_reference_dataloaders, build_reference_datasets,
    discover_sen12mscr, load_ignored_sample_ids, reference_split_audit,
)
from mscloudr.evaluation import evaluate_nfe1_endpoint
from mscloudr.metrics import REFERENCE_COMMIT, REFERENCE_REPO
from mscloudr.models.component_bridge import (
    COMPONENT_MODEL_IDENTITY, CanonicalComponentBridgeNet,
)
from mscloudr.reproducibility import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    return parser


def run(args):
    base.validate_cli_args(args)
    checkpoint = Path(args.checkpoint)
    payload = base._torch_load_payload(checkpoint)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    metadata = payload["run_metadata"]
    if metadata.get("model_identity") != COMPONENT_MODEL_IDENTITY:
        raise ValueError("wrong model identity")
    if metadata.get("split_protocol") != REFERENCE_PROTOCOL:
        raise ValueError("wrong dataset split")
    if metadata.get("schedule_name") != "canonical_alpha":
        raise ValueError("wrong bridge schedule")
    if metadata.get("conditioning_mode") != "mr_inverse":
        raise ValueError("wrong conditioning")
    if metadata.get("component_decomposition") != "orthogonal_2x2_block_low_high":
        raise ValueError("wrong decomposition identity")
    if float(metadata.get("beta_a", -1)) != 1:
        raise ValueError("wrong alpha sampling measure")
    if metadata.get("paper_grade") is False and not args.allow_smoke_checkpoint:
        raise ValueError("smoke checkpoint requires --allow-smoke-checkpoint")
    if metadata.get("architecture_widths") != [32, 64, 128, 256]:
        raise ValueError("unexpected architecture widths")
    device = base.resolve_device(args.device)
    seed = base._train_seed_from_metadata(metadata)
    seed_everything(seed,
        deterministic_algorithms=base._deterministic_algorithms_from_metadata(metadata))
    ignored = load_ignored_sample_ids(args.ignore_file)
    discovered = discover_sen12mscr(
        args.data_root, ignored_sample_ids=ignored, strict_counterparts=True
    )
    datasets = build_reference_datasets(
        discovered.samples, include_sample_id=True, strict_channels=True,
        verify_frozen_membership=True,
    )
    audit = reference_split_audit({
        key: dataset.samples for key, dataset in datasets.as_dict().items()
    })
    if audit != metadata.get("split_audit"):
        raise ValueError("checkpoint split fingerprint mismatch")
    loaders = build_reference_dataloaders(
        datasets, batch_size=args.batch_size,
        num_workers=args.num_workers, train_seed=seed,
        pin_memory=(device.type == "cuda"),
    )
    model = CanonicalComponentBridgeNet(total_steps=int(metadata["total_steps"]))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != int(metadata["trainable_parameters"]):
        raise ValueError("parameter count mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    result = evaluate_nfe1_endpoint(
        model, loaders.test, total_steps=int(metadata["total_steps"]),
        device=device, max_batches=args.max_batches,
        progress_every=args.progress_every if args.progress_every else None,
        progress_callback=base._progress if args.progress_every else None,
    )
    paper_grade = bool(metadata.get("paper_grade", False)) and args.max_batches is None
    output = {
        "format_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "model_identity": COMPONENT_MODEL_IDENTITY,
        "trainable_parameters": count,
        "component_weight": metadata["component_weight"],
        "component_decomposition": metadata["component_decomposition"],
        "checkpoint_selection_metric": metadata["checkpoint_selection_metric"],
        "checkpoint_metrics": payload.get("metrics", {}),
        "nfe": 1, "inference": "direct_x0_endpoint",
        "schedule_name": "canonical_alpha",
        "conditioning_mode": "mr_inverse",
        "split_protocol": REFERENCE_PROTOCOL,
        "split_audit": audit,
        "num_samples": result.num_samples,
        "num_batches": result.num_batches,
        "batch_size": args.batch_size, "max_batches": args.max_batches,
        "paper_grade": paper_grade,
        "metrics": result.metrics,
        "metric_counts": result.metric_counts,
        "metric_reference_repo": REFERENCE_REPO,
        "metric_reference_commit": REFERENCE_COMMIT,
        "reproducibility": metadata.get("reproducibility"),
    }
    dest = Path(args.output) if args.output else base._default_output_path(checkpoint)
    base._write_json_atomic(dest, output, overwrite=args.overwrite)
    print(json.dumps({
        "output": str(dest), "num_samples": result.num_samples,
        "metrics": result.metrics,
    }, indent=2), flush=True)
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
