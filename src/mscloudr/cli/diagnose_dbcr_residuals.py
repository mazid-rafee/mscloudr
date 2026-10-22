"""Inspect actual NAFBlock residual updates from a saved DB-CR checkpoint.

This is entirely read-only with respect to the running training job.
Uses deterministic evenly spaced validation samples from the *frozen*
ROI-disjoint manifest; does not perform optimization or test-set evaluation.

Example:
  python -m mscloudr.cli.diagnose_dbcr_residuals \
      --checkpoint ../mscloudr/outputs/DBCR_SourceGuided_perblock_full_seed42/checkpoints/latest.pt \
      --device cuda:6 --num-samples 20 --batch-size 2 --num-workers 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from mscloudr.checkpointing import CHECKPOINT_FORMAT_VERSION
from mscloudr.cli import eval as eval_base
from mscloudr.cli.eval_source_guided_dbcr import _checkpoint_config
from mscloudr.data import (
    build_reference_datasets,
    discover_sen12mscr,
    load_ignored_sample_ids,
    reference_split_audit,
)
from mscloudr.models.source_guided_dbcr import SourceGuidedDBCRNet
from mscloudr.residual_diagnostics import ResidualContributionRecorder
from mscloudr.reproducibility import seed_everything


def evenly_spaced_indices(dataset_size: int, count: int) -> list[int]:
    """Deterministic, unique indexes spread across validation membership."""
    if dataset_size <= 0 or count <= 0 or count > dataset_size:
        raise ValueError("require 1 <= count <= dataset_size")
    if count == 1:
        return [dataset_size // 2]
    return [index * (dataset_size - 1) // (count - 1) for index in range(count)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--device", required=True,
        help="Explicit device, e.g. cuda:6; avoid the GPU running training.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--ignore-file", default="manifests/known_invalid_samples.txt"
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.batch_size <= 0 or args.num_workers < 0 or args.num_samples <= 0:
        raise ValueError("batch size/samples must be positive; workers >= 0")
    device = eval_base.resolve_device(args.device)
    payload = eval_base._torch_load_payload(args.checkpoint)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if not isinstance(payload.get("run_metadata"), dict):
        raise ValueError("checkpoint missing run_metadata")
    if "model_state" not in payload:
        raise ValueError("checkpoint missing model_state")
    metadata = payload["run_metadata"]
    cfg = _checkpoint_config(metadata)
    seed = eval_base._train_seed_from_metadata(metadata)
    seed_everything(seed, deterministic_algorithms=False)

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
        name: dataset.samples
        for name, dataset in datasets.as_dict().items()
    })
    if metadata.get("split_audit") != split_audit:
        raise ValueError("validation dataset does not match checkpoint fingerprint")

    indices = evenly_spaced_indices(len(datasets.val), args.num_samples)
    selected_ids = [datasets.val.samples[i].sample_id for i in indices]
    loader = DataLoader(
        Subset(datasets.val, indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = SourceGuidedDBCRNet(cfg)
    count = model.architecture_audit()["trainable_parameters"]
    if count != int(metadata.get("trainable_parameters", -1)):
        raise ValueError("checkpoint/model parameter count differs")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    recorder = ResidualContributionRecorder(model)

    with torch.inference_mode(), recorder.capture():
        for batch in loader:
            cloudy = batch["cloudy"].to(device=device, non_blocking=True)
            sar = batch["sar"].to(device=device, non_blocking=True)
            t = torch.full(
                (cloudy.shape[0],), cfg.total_steps,
                device=device, dtype=torch.float32,
            )
            prediction = model(cloudy, t, sar)
            if prediction.shape != cloudy.shape or not bool(
                torch.isfinite(prediction).all()
            ):
                raise RuntimeError("nonfinite or incorrect model output")
    summary = recorder.summary()
    if summary["num_samples"] != len(selected_ids):
        raise RuntimeError("recorded sample count does not match selection")
    if summary["num_blocks"] != model.architecture_audit()[
        "num_time_conditioned_nafblocks"
    ]:
        raise RuntimeError("not all patent NAFBlocks were recorded")
    if summary["num_fusion_blocks"] != len(model.fusion_blocks):
        raise RuntimeError("not all SFBlocks were recorded")

    result = {
        "diagnostic": "source_guided_dbcr_feature_magnitudes_v2",
        "inference": "endpoint_xT_equals_cloudy_t_equals_T",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "model_identity": metadata["model_identity"],
        "model_parameters": count,
        "device": str(device),
        "split": "val",
        "split_audit": split_audit,
        "selection": "deterministic_evenly_spaced_validation_indices",
        "selection_indices": indices,
        "selected_sample_ids": selected_ids,
        **summary,
    }

    output = args.output
    if output is None:
        checkpoint_dir = args.checkpoint.parent
        run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
        output = (
            run_dir / "diagnostics" /
            f"feature_magnitudes_epoch{payload['epoch']:03d}.json"
        )
    eval_base._write_json_atomic(output, result, overwrite=args.overwrite)
    print(f"Checkpoint epoch: {payload['epoch']}")
    print(f"Validation samples: {summary['num_samples']} (fixed, spread across split)")
    print(f"Saved diagnostic: {output}")
    print("\nNAFBlocks: relative residual and absolute input/change L2")
    print(
        f"{'Stage':<18} {'N':>3} {'MB rel':>9} {'FFN rel':>9} "
        f"{'Total rel':>10} {'Input L2':>11} {'Change L2':>11}"
    )
    for stage, stats in sorted(summary["stages"].items()):
        print(
            f"{stage:<18} {stats['num_blocks']:>3} "
            f"{stats['mbconv_relative_l2_mean']:>9.5f} "
            f"{stats['ffn_relative_l2_mean']:>9.5f} "
            f"{stats['total_relative_l2_mean']:>10.5f} "
            f"{stats['input_l2_mean']:>11.4f} "
            f"{stats['total_update_l2_mean']:>11.4f}"
        )
    print("\nNAFBlocks: per-element RMS (comparable across resolutions)")
    print(
        f"{'Stage':<18} {'Input':>11} {'MB update':>11} "
        f"{'FFN update':>11} {'Total update':>13}"
    )
    for stage, stats in sorted(summary["stages"].items()):
        print(
            f"{stage:<18} {stats['input_rms_mean']:>11.6f} "
            f"{stats['mbconv_update_rms_mean']:>11.6f} "
            f"{stats['ffn_update_rms_mean']:>11.6f} "
            f"{stats['total_update_rms_mean']:>13.6f}"
        )
    print("\nSFBlocks: fusion output relative to optical input")
    print(
        f"{'Block':<18} {'Opt RMS':>11} {'SAR RMS':>11} "
        f"{'Out RMS':>11} {'Change RMS':>11} {'Change/Opt':>11}"
    )
    for name, stats in sorted(summary["fusion_blocks"].items()):
        print(
            f"{name:<18} {stats['optical_input_rms_mean']:>11.6f} "
            f"{stats['sar_input_rms_mean']:>11.6f} "
            f"{stats['output_rms_mean']:>11.6f} "
            f"{stats['output_minus_optical_rms_mean']:>11.6f} "
            f"{stats['output_minus_optical_relative_l2_mean']:>11.6f}"
        )
    print("\nNote: SFBlock output change is not isolated SAR influence.")
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
