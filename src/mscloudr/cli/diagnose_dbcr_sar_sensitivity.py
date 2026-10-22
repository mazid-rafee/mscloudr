"""Read-only source-guided DB-CR SFBlock SAR sensitivity experiment.

Checks the same validation optical inputs and clean targets under:
NORMAL: aligned normalized SAR; SHUFFLED: global cyclic permutation of the
selected validation SAR pool (no self-pairs); ZERO: normalized SAR = 0.

All runs use a saved checkpoint, x_T=cloudy, t=T, and no gradients.
Nothing in training/model/checkpoint files is changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from mscloudr.checkpointing import CHECKPOINT_FORMAT_VERSION
from mscloudr.cli import eval as eval_base
from mscloudr.cli.diagnose_dbcr_residuals import evenly_spaced_indices
from mscloudr.cli.eval_source_guided_dbcr import _checkpoint_config
from mscloudr.data import (
    build_reference_datasets, discover_sen12mscr,
    load_ignored_sample_ids, reference_split_audit,
)
from mscloudr.data.sen12mscr_splits import sample_group_key
from mscloudr.models.source_guided_dbcr import SourceGuidedDBCRNet
from mscloudr.reproducibility import seed_everything
from mscloudr.sar_sensitivity import (
    SFBlockSnapshot, extend_sample_measurements,
    paired_reconstruction_statistics, per_sample_l1, rotate_other_sample_indices,
    summarize_samples, tensor_change,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", required=True, help="Use an unused GPU, not GPU 7.")
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


def _empty_condition(*, normal: bool) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {"reconstruction_l1": []}
    if not normal:
        values.update({
            "prediction_delta_l1": [],
            "prediction_delta_rms": [],
            "prediction_delta_relative_l2": [],
            "reconstruction_l1_delta_vs_normal": [],
        })
    return values


def _empty_fusion() -> dict[str, list[float]]:
    return {
        "output_delta_l1": [],
        "output_delta_rms": [],
        "output_delta_relative_l2": [],
        "normal_output_rms": [],
    }


def run(args: argparse.Namespace) -> dict:
    if args.num_samples < 2:
        raise ValueError("--num-samples must be >= 2 for strict global SAR derangement")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be > 0 and num-workers >= 0")

    device = eval_base.resolve_device(args.device)
    payload = eval_base._torch_load_payload(args.checkpoint)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict) or "model_state" not in payload:
        raise ValueError("checkpoint missing metadata or model weights")
    cfg = _checkpoint_config(metadata)
    seed_everything(
        eval_base._train_seed_from_metadata(metadata),
        deterministic_algorithms=eval_base._deterministic_algorithms_from_metadata(
            metadata
        ),
    )

    ignored = load_ignored_sample_ids(args.ignore_file)
    discovery = discover_sen12mscr(
        args.data_root, ignored_sample_ids=ignored, strict_counterparts=True
    )
    datasets = build_reference_datasets(
        discovery.samples, include_sample_id=True,
        strict_channels=True, verify_frozen_membership=True,
    )
    split_audit = reference_split_audit({
        name: data.samples for name, data in datasets.as_dict().items()
    })
    if metadata.get("split_audit") != split_audit:
        raise ValueError("validation split fingerprint differs from checkpoint")
    indices = evenly_spaced_indices(len(datasets.val), args.num_samples)
    selected_ids = [datasets.val.samples[i].sample_id for i in indices]
    roi_groups = [sample_group_key(datasets.val.samples[i]) for i in indices]
    permutation = rotate_other_sample_indices(args.num_samples)
    shuffled_source_ids = [selected_ids[j] for j in permutation]
    shuffled_roi_groups = [roi_groups[j] for j in permutation]
    # The permutation is across all selected samples, not within minibatches.
    # Preload only the selected SAR images (as normalized tensors) on CPU.
    sar_pool = torch.stack([datasets.val[i]["sar"] for i in indices], dim=0)
    loader = DataLoader(
        Subset(datasets.val, indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False, drop_last=False,
        pin_memory=device.type == "cuda",
    )

    model = SourceGuidedDBCRNet(cfg)
    param_count = model.architecture_audit()["trainable_parameters"]
    if param_count != int(metadata.get("trainable_parameters", -1)):
        raise ValueError("checkpoint parameter count does not match model")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device=device).eval()
    snapshots = SFBlockSnapshot(model)
    if len(snapshots.blocks) != 4:
        raise ValueError("expected exactly 4 SFBlocks")

    conditions = {
        "NORMAL": _empty_condition(normal=True),
        "SHUFFLED": _empty_condition(normal=False),
        "ZERO": _empty_condition(normal=False),
    }
    fusion = {
        name: {
            stage: _empty_fusion() for stage in snapshots.blocks
        } for name in ("SHUFFLED", "ZERO")
    }
    per_sample = []
    seen = 0

    with torch.inference_mode():
        for batch in loader:
            cloudy = batch["cloudy"].to(device=device, non_blocking=True)
            sar = batch["sar"].to(device=device, non_blocking=True)
            target = batch["target"].to(device=device, non_blocking=True)
            size = cloudy.shape[0]
            if tuple(batch["sample_id"]) != tuple(selected_ids[seen:seen + size]):
                raise RuntimeError("validation order changed unexpectedly")
            shuffled = sar_pool[permutation[seen:seen + size]].to(
                device=device, non_blocking=True
            )
            zero = torch.zeros_like(sar)
            t = torch.full(
                (size,), cfg.total_steps, dtype=torch.float32, device=device
            )
            reference_pred, reference_maps = snapshots.run(model, cloudy, t, sar)
            reference_l1 = per_sample_l1(reference_pred, target)
            extend_sample_measurements(
                conditions["NORMAL"]["reconstruction_l1"], reference_l1
            )
            batch_sample_rows = [
                {
                    "sample_id": selected_ids[seen + i],
                    "roi_group": roi_groups[seen + i],
                    "shuffled_sar_source_id": shuffled_source_ids[seen + i],
                    "shuffled_sar_source_roi_group": shuffled_roi_groups[seen + i],
                    "normal_l1": float(reference_l1[i].item()),
                } for i in range(size)
            ]

            for name, replacement in (("SHUFFLED", shuffled), ("ZERO", zero)):
                pred, maps = snapshots.run(model, cloudy, t, replacement)
                l1 = per_sample_l1(pred, target)
                delta = tensor_change(reference_pred, pred)
                values = conditions[name]
                extend_sample_measurements(values["reconstruction_l1"], l1)
                extend_sample_measurements(
                    values["reconstruction_l1_delta_vs_normal"],
                    l1 - reference_l1,
                )
                for key in ("l1", "rms", "relative_l2"):
                    extend_sample_measurements(
                        values["prediction_delta_" + key], delta[key]
                    )
                for stage in snapshots.blocks:
                    measurements = tensor_change(
                        reference_maps[stage], maps[stage]
                    )
                    for key, metric_key in (
                        ("output_delta_l1", "l1"),
                        ("output_delta_rms", "rms"),
                        ("output_delta_relative_l2", "relative_l2"),
                        ("normal_output_rms", "baseline_rms"),
                    ):
                        extend_sample_measurements(
                            fusion[name][stage][key], measurements[metric_key]
                        )
                for i, row in enumerate(batch_sample_rows):
                    row[name.lower() + "_l1"] = float(l1[i].item())
                    row[name.lower() + "_prediction_delta_l1"] = float(
                        delta["l1"][i].item()
                    )
                    row[name.lower() + "_minus_normal_l1"] = (
                        float(l1[i].item()) - row["normal_l1"]
                    )
            per_sample.extend(batch_sample_rows)
            seen += size
            print(f"processed={seen}/{args.num_samples}", flush=True)

    if seen != args.num_samples:
        raise RuntimeError("processed sample count mismatch")
    paired = paired_reconstruction_statistics(
        conditions["NORMAL"]["reconstruction_l1"],
        conditions["SHUFFLED"]["reconstruction_l1"],
        roi_groups,
    )
    paired["shuffled_sar_source_same_roi_fraction"] = (
        sum(source == target for source, target in zip(
            shuffled_roi_groups, roi_groups
        )) / seen
    )

    result = {
        "diagnostic": "dbcr_sfblock_sar_sensitivity_paired_v2",
        "inference": "endpoint_direct_x0_t_equals_T",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "model_identity": metadata["model_identity"],
        "model_parameters": param_count,
        "device": str(device),
        "split": "val",
        "split_audit": split_audit,
        "sample_count": seen,
        "selection": "deterministic_evenly_spaced_validation_indices",
        "selection_indices": indices,
        "selected_sample_ids": selected_ids,
        "selected_roi_groups": roi_groups,
        "shuffled_sar_source_ids": shuffled_source_ids,
        "shuffled_sar_source_roi_groups": shuffled_roi_groups,
        "intervention_definitions": {
            "NORMAL": "correctly paired normalized SAR",
            "SHUFFLED": "cyclic permutation of selected validation SAR pool; no self-pairs",
            "ZERO": "normalized SAR values replaced by zeros",
            "fixed": "cloudy optical input, clean target, t=T, checkpoint weights",
        },
        "conditions": {
            name: summarize_samples(values)
            for name, values in conditions.items()
        },
        "sfblock_changes_vs_normal": {
            name: {
                stage: summarize_samples(values)
                for stage, values in by_stage.items()
            }
            for name, by_stage in fusion.items()
        },
        "per_sample": per_sample,
        "paired_normal_vs_shuffled": paired,
        "interpretation": (
            "SAR interventions are out-of-distribution perturbations; output "
            "differences measure sensitivity, not necessarily beneficial or "
            "causal use of correctly paired SAR."
        ),
    }
    output = args.output
    if output is None:
        checkpoint_dir = args.checkpoint.parent
        run_dir = (
            checkpoint_dir.parent
            if checkpoint_dir.name == "checkpoints"
            else checkpoint_dir
        )
        output = (
            run_dir / "diagnostics" /
            f"sar_sensitivity_paired_epoch{payload['epoch']:03d}_n{seen:04d}.json"
        )
    eval_base._write_json_atomic(output, result, overwrite=args.overwrite)

    print(f"\nCheckpoint epoch: {payload['epoch']}")
    print(f"Validation samples: {seen}; saved: {output}")
    print(
        f"{'Condition':<12} {'Recon L1':>12} "
        f"{'Delta L1':>12} {'Pred rel-L2':>14}"
    )
    for name, values in result["conditions"].items():
        print(
            f"{name:<12} {values['reconstruction_l1_mean']:>12.6f} "
            f"{values.get('reconstruction_l1_delta_vs_normal_mean', 0.0):>12.6f} "
            f"{values.get('prediction_delta_relative_l2_mean', 0.0):>14.6f}"
        )
    print("\nPaired SHUFFLED - NORMAL reconstruction L1 (positive favors NORMAL)")
    print(
        f"Mean delta: {paired['mean_delta_l1']:+.6f}; "
        f"median: {paired['median_delta_l1']:+.6f}; "
        f"relative increase: {paired['relative_increase_pct']:.2f}%"
    )
    print(
        f"NORMAL better: {paired['normal_better_count']}/{seen}; "
        f"SHUFFLED better: {paired['shuffled_better_count']}/{seen}; "
        f"ties: {paired['tie_count']}; "
        f"ROI groups represented: {paired['num_roi_groups']}"
    )
    print(
        f"95% patch bootstrap interval: {paired['patch_bootstrap_95ci']}; "
        f"95% ROI-cluster interval: {paired['roi_cluster_bootstrap_95ci']}"
    )
    print(
        "SHUFFLED SAR from same ROI as target: "
        f"{100.0 * paired['shuffled_sar_source_same_roi_fraction']:.1f}%"
    )
    print(
        "Caution: ROI-cluster intervals with few groups and one shuffled "
        "permutation are descriptive, not definitive significance tests."
    )
    for name, stages in result["sfblock_changes_vs_normal"].items():
        print(f"\n{name}: SFBlock output change relative to NORMAL")
        print(
            f"{'Stage':<20} {'Delta L1':>11} "
            f"{'Delta RMS':>11} {'Delta rel-L2':>14}"
        )
        for stage, values in sorted(stages.items()):
            print(
                f"{stage:<20} {values['output_delta_l1_mean']:>11.6f} "
                f"{values['output_delta_rms_mean']:>11.6f} "
                f"{values['output_delta_relative_l2_mean']:>14.6f}"
            )
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
