"""Read-only activation + gradient diagnostic for source-guided DB-CR.

Forward hooks inspect modulation, SimpleGate, SCA and FFN activations.
One endpoint validation L1 backward per batch probes gradient flow.
NO optimizer is constructed and NO parameter update or checkpoint write occurs.
Run on an unused GPU, never on the training GPU.
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
from mscloudr.internal_path_diagnostics import InternalPathRecorder
from mscloudr.models.source_guided_dbcr import SourceGuidedDBCRNet
from mscloudr.reproducibility import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", required=True, help="Use a free GPU (not GPU 7).")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ignore-file", default="manifests/known_invalid_samples.txt")
    parser.add_argument("--num-samples", default=4, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.num_samples < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("samples/batch-size must be positive; workers >= 0")
    device = eval_base.resolve_device(args.device)
    payload = eval_base._torch_load_payload(args.checkpoint)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict) or "model_state" not in payload:
        raise ValueError("checkpoint missing run_metadata or model_state")
    cfg = _checkpoint_config(metadata)
    if cfg.time_sharing != "per_block":
        raise ValueError("time-path diagnostic requires per-block time MLPs")
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
    loader = DataLoader(
        Subset(datasets.val, indices),
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False, pin_memory=device.type == "cuda"
    )

    model = SourceGuidedDBCRNet(cfg)
    count = model.architecture_audit()["trainable_parameters"]
    if count != int(metadata.get("trainable_parameters", -1)):
        raise ValueError("checkpoint/model parameter count differs")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device=device).eval()
    recorder = InternalPathRecorder(model)

    loss_values: list[float] = []
    seen = 0
    with torch.enable_grad(), recorder.capture():
        for batch in loader:
            model.zero_grad(set_to_none=True)
            cloudy = batch["cloudy"].to(device=device, non_blocking=True)
            sar = batch["sar"].to(device=device, non_blocking=True)
            target = batch["target"].to(device=device, non_blocking=True)
            bs = cloudy.shape[0]
            t = torch.full(
                (bs,), cfg.total_steps, device=device, dtype=torch.float32
            )
            prediction = model(cloudy, t, sar)
            if prediction.shape != target.shape:
                raise RuntimeError("incorrect model output shape")
            loss = (prediction - target).abs().mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("nonfinite diagnostic loss")
            loss_values.extend([float(v) for v in
                (prediction.detach() - target).abs().flatten(1).mean(1).cpu().tolist()
            ])
            loss.backward()
            recorder.record_parameter_grads()
            seen += bs
            print(f"processed={seen}/{args.num_samples} l1={loss.item():.6f}", flush=True)
            model.zero_grad(set_to_none=True)

    if seen != len(selected_ids):
        raise RuntimeError("mismatched sample count")
    summary = recorder.summary()
    if summary["num_blocks"] != model.architecture_audit()[
        "num_time_conditioned_nafblocks"
    ]:
        raise RuntimeError("missing time-conditioned blocks")
    result = {
        "diagnostic": "source_guided_dbcr_internal_paths_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "model_identity": metadata["model_identity"],
        "device": str(device),
        "inference": "endpoint_xT_equals_cloudy_t_equals_T",
        "gradient_loss": "per-batch mean abs(prediction - clean optical)",
        "gradient_optimizer_steps": 0,
        "split": "val",
        "split_audit": split_audit,
        "sample_count": seen,
        "batch_size": args.batch_size,
        "selection_indices": indices,
        "selected_sample_ids": selected_ids,
        "mean_reconstruction_l1": sum(loss_values) / seen,
        **summary,
    }
    output = args.output
    if output is None:
        folder = args.checkpoint.parent
        run_folder = folder.parent if folder.name == "checkpoints" else folder
        output = run_folder / "diagnostics" / (
            f"internal_paths_epoch{payload['epoch']:03d}_n{seen:03d}.json"
        )
    eval_base._write_json_atomic(output, result, overwrite=args.overwrite)
    print(f"\nCheckpoint epoch {payload['epoch']}; samples {seen}; L1 "
          f"{result['mean_reconstruction_l1']:.6f}")
    print(f"Saved: {output}")
    print("\nTime modulation and SCA diagnostics (stage mean over blocks)")
    print(
        f"{'Stage':<18} {'scale C':>8} {'scale F':>8} "
        f"{'mod C':>9} {'gate':>10} {'SCA RMS':>10} {'beta*SCA':>10}"
    )
    for name, row in sorted(summary["stages"].items()):
        print(
            f"{name:<18} {row['scale_conv_abs_mean_mean']:>8.5f} "
            f"{row['scale_ffn_abs_mean_mean']:>8.5f} "
            f"{row['mod_conv_rms_mean']:>9.5f} "
            f"{row['sca_gate_abs_mean_mean']:>10.6f} "
            f"{row['after_sca_rms_mean']:>10.6f} "
            f"{row['beta_scaled_sca_rms_mean']:>10.6f}"
        )
    print("\nGradient flow (RMS per parameter tensor, averaged over batches)")
    print(
        f"{'Stage':<18} {'input':>11} {'beta':>11} {'gamma':>11} "
        f"{'conv1':>11} {'SCA w':>11} {'time FC3':>11}"
    )
    for name, row in sorted(summary["stages"].items()):
        def val(key):
            value = row.get(key)
            return f"{value:.3e}" if value is not None else "NONE"
        print(
            f"{name:<18} {val('input_grad_rms_mean'):>11} "
            f"{val('beta_grad_rms_mean'):>11} "
            f"{val('gamma_grad_rms_mean'):>11} "
            f"{val('conv1.weight_grad_rms_mean'):>11} "
            f"{val('sca.weight_grad_rms_mean'):>11} "
            f"{val('time_embedding.linear3.weight_grad_rms_mean'):>11}"
        )
    print("\nNote: measurements are read-only; no optimizer steps taken.")
    print("Full per-block and FFN/shift diagnostics are in the JSON report.")
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
