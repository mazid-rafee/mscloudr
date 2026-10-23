"""Read-only full-ROI CanonicalBridgeNet endpoint-gradient analysis.

Only accepts the original CNN backbone, uniform physical-alpha training
measure, physical-alpha conditioning and zero endpoint mixture. This prevents
architecture/measure/conditioning differences from confounding the diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mscloudr.bridge import get_bridge_schedule
from mscloudr.cli import eval as base_eval
from mscloudr.cli.eval_canonical_bridge import (
    _checkpoint_backbone,
    _checkpoint_widths,
    _validate_checkpoint,
)
from mscloudr.data import (
    REFERENCE_PROTOCOL,
    build_reference_dataloaders,
    build_reference_datasets,
    discover_sen12mscr,
    load_ignored_sample_ids,
    reference_split_audit,
)
from mscloudr.endpoint_alignment import profile_endpoint_alignment
from mscloudr.models.canonical_bridge import (
    CanonicalBridgeNet,
    count_canonical_bridge_parameters,
)
from mscloudr.reproducibility import seed_everything
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ignore-file", default="manifests/known_invalid_samples.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_canonical_control(metadata: dict) -> None:
    """Accept only the frozen CNN/uniform/physical-alpha experimental control."""
    if _checkpoint_backbone(metadata) != "cnn":
        raise ValueError("diagnostic requires CanonicalBridgeNet CNN backbone")
    if tuple(_checkpoint_widths(metadata)) != (32, 64, 128, 256):
        raise ValueError("diagnostic requires default 7.65M CNN widths")
    if metadata.get("conditioning_mode") != PHYSICAL_ALPHA_CONDITIONING:
        raise ValueError("diagnostic requires physical_alpha conditioning")
    if abs(float(metadata.get("beta_a", -1.0)) - 1.0) > 1e-12:
        raise ValueError("diagnostic requires uniform physical-alpha measure")
    if float(metadata.get("endpoint_probability", 0.0)) != 0.0:
        raise ValueError("diagnostic requires zero endpoint-mixture probability")


def run(args: argparse.Namespace) -> dict:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid dataloader settings")
    checkpoint_path = Path(args.checkpoint)
    target = Path(args.output_dir) / "alignment.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(target)

    payload = base_eval._torch_load_payload(checkpoint_path)
    metadata = _validate_checkpoint(
        payload, allow_smoke_checkpoint=args.allow_smoke_checkpoint
    )
    validate_canonical_control(metadata)
    device = base_eval.resolve_device(args.device)
    train_seed = base_eval._train_seed_from_metadata(metadata)
    seed_everything(
        train_seed,
        deterministic_algorithms=base_eval._deterministic_algorithms_from_metadata(
            metadata
        ),
    )

    ignored = load_ignored_sample_ids(args.ignore_file)
    discovery = discover_sen12mscr(
        args.data_root, ignored_sample_ids=ignored, strict_counterparts=True
    )
    datasets = build_reference_datasets(
        discovery.samples,
        include_sample_id=True,
        strict_channels=True,
        verify_frozen_membership=True,
    )
    audit = reference_split_audit({
        name: dataset.samples for name, dataset in datasets.as_dict().items()
    })
    if metadata.get("split_protocol") != REFERENCE_PROTOCOL:
        raise ValueError("checkpoint does not use reference split")
    if metadata.get("split_audit") != audit:
        raise ValueError("checkpoint split fingerprint differs from current dataset")

    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=train_seed,
        pin_memory=(device.type == "cuda"),
    )
    model = CanonicalBridgeNet(
        widths=_checkpoint_widths(metadata),
        total_steps=int(metadata["total_steps"]),
        backbone_variant="cnn",
    ).to(device)
    if count_canonical_bridge_parameters(model) != int(metadata["trainable_parameters"]):
        raise ValueError("canonical model parameter count mismatch")
    model.load_state_dict(payload["model_state"], strict=True)

    result = profile_endpoint_alignment(
        model,
        loaders.train,
        schedule=get_bridge_schedule("canonical_alpha"),
        total_steps=int(metadata["total_steps"]),
        device=device,
        max_batches=args.max_batches,
        num_bins=args.num_bins,
        sampler_seed=args.sampler_seed,
        conditioning_mode=PHYSICAL_ALPHA_CONDITIONING,
    )
    result.update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "model_identity": metadata["model_identity"],
        "backbone_variant": "cnn",
        "training_bridge_measure": "uniform_discrete_alpha_grid",
        "beta_a": 1.0,
        "split_protocol": REFERENCE_PROTOCOL,
        "split_audit": audit,
        "train_seed": train_seed,
        "architecture_widths": list(_checkpoint_widths(metadata)),
        "paper_grade": False,
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({
        "output": str(target),
        "cosines": [round(b["cosine"]["mean"], 5) for b in result["bins"]],
    }, indent=2))
    return result


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
