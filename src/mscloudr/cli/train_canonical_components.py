"""Controlled independent low/high component training on CanonicalBridgeNet.

Source architecture: original CNN 32/64/128/256, 7.65M baseline.
Fixed conditions: uniform physical alpha, MR-inverse r=3, straight bridge,
direct x0 prediction, 50 epochs seed 42, ROI-disjoint split, NFE=1.
Only output-head projection and an explicit component objective change.
"""
from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path

from mscloudr.cli import train as base
from mscloudr.cli import train_canonical_bridge as canonical
from mscloudr.component_training import component_training_step
from mscloudr.data import (
    build_reference_dataloaders, build_reference_datasets,
    discover_sen12mscr, load_ignored_sample_ids,
    reference_split_audit,
)
from mscloudr.models.component_bridge import (
    COMPONENT_MODEL_IDENTITY, CanonicalComponentBridgeNet,
)
from mscloudr.reproducibility import (
    make_torch_generator, seed_everything,
)
from mscloudr.runner import fit
from mscloudr.training import MR_INVERSE_CONDITIONING


def build_parser() -> argparse.ArgumentParser:
    parser = canonical.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        schedule="canonical_alpha", beta_a=1.0,
        conditioning_coordinate=MR_INVERSE_CONDITIONING,
    )
    parser.add_argument("--component-weight", type=float, default=0.5)
    return parser


def _validate_variant(args: argparse.Namespace) -> None:
    base.validate_cli_args(args)
    if args.schedule != "canonical_alpha":
        raise ValueError("experiment requires canonical_alpha")
    if args.backbone != "cnn":
        raise ValueError("component model requires the original cnn backbone")
    if tuple(args.widths) != (32, 64, 128, 256):
        raise ValueError("component model requires widths (32,64,128,256)")
    if float(args.endpoint_probability) != 0.0:
        raise ValueError("component experiment requires endpoint_probability=0")
    if float(args.beta_a) != 1.0:
        raise ValueError("experiment requires uniform alpha (beta-a=1)")
    if args.conditioning_coordinate != MR_INVERSE_CONDITIONING:
        raise ValueError("experiment requires mr_inverse conditioning")
    if not math.isfinite(args.component_weight) or not (0 <= args.component_weight <= 10):
        raise ValueError("component weight must be in [0,10]")


def _resume_signature(m: dict) -> dict:
    keys = (
        "model_identity", "trainable_parameters", "component_weight",
        "architecture_widths", "schedule_name", "conditioning_mode",
        "beta_a", "total_steps", "batch_size", "num_workers",
        "learning_rate", "split_protocol", "split_audit",
    )
    identity = {key: m[key] for key in keys}
    identity["reproducibility"] = m["reproducibility"]["seeds"]
    identity["deterministic_algorithms"] = m["reproducibility"]["deterministic_algorithms"]
    return identity


def run(args):
    _validate_variant(args)
    run_dir = base._prepare_output_dir(args)
    device = base.resolve_device(args.device)
    seed_everything(
        args.train_seed,
        deterministic_algorithms=args.deterministic_algorithms,
    )

    ignored = load_ignored_sample_ids(args.ignore_file)
    discovery = discover_sen12mscr(
        args.data_root, ignored_sample_ids=ignored, strict_counterparts=True
    )
    datasets = build_reference_datasets(
        discovery.samples, include_sample_id=True, strict_channels=True,
        verify_frozen_membership=True,
    )
    audit = reference_split_audit({
        name: dataset.samples for name, dataset in datasets.as_dict().items()
    })
    loaders = build_reference_dataloaders(
        datasets, batch_size=args.batch_size,
        num_workers=args.num_workers, train_seed=args.train_seed,
        pin_memory=(device.type == "cuda"),
    )
    model = CanonicalComponentBridgeNet(total_steps=args.total_steps)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = canonical._run_metadata(
        args=args, dataset_root=discovery.dataset_root,
        split_audit=audit, trainable_parameters=count, beta_a=1.0,
        device=device,
    )
    metadata.update({
        "model_identity": COMPONENT_MODEL_IDENTITY,
        "model_family": "dual_stream_residual_unet_projected_heads",
        "component_decomposition": "orthogonal_2x2_block_low_high",
        "low_projection": "2x2_block_mean",
        "high_projection": "identity_minus_low_projection",
        "component_weight": float(args.component_weight),
        "component_loss": "0.5*(L1(P_L(head_L), P_L(x0)) + L1(P_H(head_H), P_H(x0)))",
        "objective": "direct_x0_L1 + component_weight * component_loss",
        "train_l1_field_semantics": "composite_training_objective_not_plain_L1",
        "prediction_target": "direct_x0",
        "backbone_variant": "cnn",
    })
    path = run_dir / "run_config.json"
    if args.resume is None:
        base._write_json_atomic(path, metadata)
    else:
        if not path.is_file():
            raise FileNotFoundError("resume requires existing run_config.json")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if _resume_signature(previous) != _resume_signature(metadata):
            raise ValueError("resume settings mismatch")
        base._write_json_atomic(path, metadata)
    print(json.dumps({
        "output": str(run_dir), "model_identity": COMPONENT_MODEL_IDENTITY,
        "parameters": count, "component_weight": args.component_weight,
        "train_samples": len(datasets.train), "val_samples": len(datasets.val),
        "checkpoint_selection": "val_endpoint_l1",
    }, indent=2), flush=True)
    return fit(
        model, train_loader=loaders.train, val_loader=loaders.val,
        schedule=canonical._beta_a1_schedule(1.0),
        schedule_name="canonical_alpha",
        total_steps=args.total_steps,
        sampler_generator=make_torch_generator(args.sampler_seed, device="cpu"),
        output_dir=run_dir, epochs=args.epochs, device=device,
        model_identity=COMPONENT_MODEL_IDENTITY,
        conditioning_mode=MR_INVERSE_CONDITIONING,
        train_step_fn=partial(component_training_step, component_weight=args.component_weight),
        lr=args.lr, run_metadata=metadata, resume_from=args.resume,
        epoch_callback=base._epoch_progress,
        batch_progress_callback=base._batch_progress if args.progress_every else None,
        progress_every=args.progress_every if args.progress_every else None,
        max_train_batches=args.smoke_train_batches if args.smoke_run else None,
        max_val_batches=args.smoke_val_batches if args.smoke_run else None,
    )


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result[-1].to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
