"""Held-out ROI/season fixed-alpha risk profile for sampler design.

This diagnostic evaluates the CanonicalAlpha pilot checkpoint on the pilot
VALIDATION split only. The validation ROIs are geographically disjoint from
training under the frozen UnCRtainTS split, while the final test split remains
untouched.

For each fixed physical corruption level alpha it records per-sample L1 and
aggregates it by season-scoped ROI group. The primary robust summaries are:

  * sample_mean_l1: ordinary validation-set mean;
  * equal_group_mean_l1: mean of ROI-group means, giving each ROI equal weight;
  * worst_group_l1: maximum ROI-group mean;
  * cvar_group_l1: mean of the worst fraction of ROI-group means.

An optional training difficulty profile from ``mscloudr.cli.difficulty_profile``
can be supplied to record validation-minus-training gaps without rerunning the
training split. This keeps the final test set out of sampler design.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from mscloudr.bridge import make_bridge_state
from mscloudr.cli import eval as eval_base
from mscloudr.cli import fixed_alpha_sweep
from mscloudr.data import (
    build_reference_dataloaders,
    discover_sen12mscr,
    load_ignored_sample_ids,
)
from mscloudr.data.pilot import PILOT_PROTOCOL
from mscloudr.data.sen12mscr_splits import sample_group_key
from mscloudr.models import (
    LEGACY_DBCR_PARAMETER_COUNT,
    LegacyDBCRNet,
    count_legacy_parameters,
)
from mscloudr.reproducibility import seed_everything


DEFAULT_ALPHAS = tuple(i / 10.0 for i in range(11))
DEFAULT_CVAR_FRACTION = 0.20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure geographically held-out ROI/season risk across fixed "
            "physical alpha values using the pilot validation split."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--ignore-file",
        default="manifests/known_invalid_samples.txt",
    )
    parser.add_argument(
        "--alphas",
        default=",".join(f"{value:.1f}" for value in DEFAULT_ALPHAS),
        help="Comma-separated physical alpha values in [0,1].",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N batches for each alpha; 0 disables.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional truncation for smoke checks.",
    )
    parser.add_argument(
        "--cvar-fraction",
        type=float,
        default=DEFAULT_CVAR_FRACTION,
        help=(
            "Fraction of highest-risk ROI groups averaged for group CVaR; "
            "must lie in (0,1]."
        ),
    )
    parser.add_argument(
        "--train-profile",
        default=None,
        help=(
            "Optional difficulty_profile_train.json. When supplied, the "
            "diagnostic also records validation-minus-training L1 gaps."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_alphas(text: str) -> list[float]:
    return fixed_alpha_sweep.parse_alphas(text)


def _default_output_path(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    filename = "generalization_profile_val_groups.json"
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent / filename
    return checkpoint.with_name(filename)


def _validate_checkpoint(payload: dict) -> dict:
    metadata = payload.get("run_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing run_metadata")
    if metadata.get("split_protocol") != PILOT_PROTOCOL:
        raise ValueError("generalization profile requires a pilot10 checkpoint")
    if metadata.get("schedule_name") != "canonical_alpha":
        raise ValueError("generalization profile requires canonical_alpha")

    # For CanonicalAlpha, historical raw_t and physical_alpha conditioning are
    # numerically the same coordinate: t=T*alpha. fixed_alpha_sweep resolves
    # both correctly, so no stronger conditioning restriction is required.
    return metadata


def _load_train_profile(path: str | Path) -> dict[float, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("train profile is missing results")

    mapping: dict[float, float] = {}
    for entry in results:
        if not isinstance(entry, dict):
            raise ValueError("train profile result entries must be objects")
        alpha = float(entry["requested_alpha"])
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict) or "L1" not in metrics:
            raise ValueError("train profile result is missing metrics.L1")
        if alpha in mapping:
            raise ValueError(f"duplicate alpha in train profile: {alpha}")
        mapping[alpha] = float(metrics["L1"])
    return mapping


def _lookup_train_l1(mapping: dict[float, float], alpha: float) -> float | None:
    for key, value in mapping.items():
        if math.isclose(float(key), float(alpha), rel_tol=0.0, abs_tol=1e-9):
            return float(value)
    return None


def summarize_group_losses(
    *,
    sample_losses: list[float],
    group_losses: dict[str, list[float]],
    group_seasons: dict[str, str],
    cvar_fraction: float,
) -> dict:
    if not sample_losses:
        raise ValueError("sample_losses must not be empty")
    if not group_losses:
        raise ValueError("group_losses must not be empty")
    if not 0.0 < float(cvar_fraction) <= 1.0:
        raise ValueError("cvar-fraction must lie in (0,1]")

    group_rows = []
    for group in sorted(group_losses):
        values = group_losses[group]
        if not values:
            continue
        group_rows.append(
            {
                "group": group,
                "season": group_seasons[group],
                "num_samples": len(values),
                "mean_l1": float(sum(values) / len(values)),
            }
        )
    if not group_rows:
        raise ValueError("no non-empty groups were observed")

    group_means = [float(row["mean_l1"]) for row in group_rows]
    sorted_worst = sorted(group_rows, key=lambda row: row["mean_l1"], reverse=True)
    cvar_group_count = max(1, int(math.ceil(float(cvar_fraction) * len(group_rows))))
    cvar_rows = sorted_worst[:cvar_group_count]

    season_values: dict[str, list[float]] = defaultdict(list)
    for row in group_rows:
        season_values[str(row["season"])].append(float(row["mean_l1"]))
    season_equal_group_mean_l1 = {
        season: float(sum(values) / len(values))
        for season, values in sorted(season_values.items())
    }

    return {
        "num_samples": len(sample_losses),
        "num_groups": len(group_rows),
        "sample_mean_l1": float(sum(sample_losses) / len(sample_losses)),
        "equal_group_mean_l1": float(sum(group_means) / len(group_means)),
        "group_std_l1": (
            float(statistics.pstdev(group_means)) if len(group_means) > 1 else 0.0
        ),
        "worst_group_l1": float(sorted_worst[0]["mean_l1"]),
        "worst_group": sorted_worst[0]["group"],
        "cvar_fraction": float(cvar_fraction),
        "cvar_group_count": cvar_group_count,
        "cvar_group_l1": float(
            sum(float(row["mean_l1"]) for row in cvar_rows) / len(cvar_rows)
        ),
        "cvar_groups": [row["group"] for row in cvar_rows],
        "season_equal_group_mean_l1": season_equal_group_mean_l1,
        "groups": group_rows,
    }


def run(args: argparse.Namespace) -> dict:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.progress_every < 0:
        raise ValueError("progress-every must be non-negative")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("max-batches must be positive when provided")
    if not 0.0 < float(args.cvar_fraction) <= 1.0:
        raise ValueError("cvar-fraction must lie in (0,1]")

    alphas = parse_alphas(args.alphas)
    checkpoint_path = Path(args.checkpoint)
    payload = eval_base._torch_load_payload(checkpoint_path)
    metadata = _validate_checkpoint(payload)

    ignored = load_ignored_sample_ids(args.ignore_file)
    discovery = discover_sen12mscr(
        args.data_root,
        ignored_sample_ids=ignored,
        strict_counterparts=True,
    )
    metadata_validated, datasets, split_audit, membership = (
        fixed_alpha_sweep._validate_and_build_datasets(payload, discovery)
    )
    if metadata_validated != metadata:
        metadata = metadata_validated

    device = eval_base.resolve_device(args.device)
    train_seed = eval_base._train_seed_from_metadata(metadata)
    seed_everything(
        train_seed,
        deterministic_algorithms=eval_base._deterministic_algorithms_from_metadata(
            metadata
        ),
    )
    loaders = build_reference_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=train_seed,
        pin_memory=(device.type == "cuda"),
    )
    loader = loaders.val

    sample_meta = {
        sample.sample_id: {
            "group": sample_group_key(sample),
            "season": sample.season_prefix,
        }
        for sample in datasets.val.samples
    }

    model = LegacyDBCRNet()
    if count_legacy_parameters(model) != LEGACY_DBCR_PARAMETER_COUNT:
        raise RuntimeError("legacy_dbcr parameter count drifted")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()

    schedule_name = str(metadata["schedule_name"])
    conditioning_mode = fixed_alpha_sweep.conditioning_mode_from_metadata(metadata)
    total_steps = int(metadata["total_steps"])
    total_batches = len(loader)
    if args.max_batches is not None:
        total_batches = min(total_batches, int(args.max_batches))

    train_profile = (
        _load_train_profile(args.train_profile)
        if args.train_profile is not None
        else None
    )

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(payload.get("epoch", 0)),
                "schedule_name": schedule_name,
                "conditioning_mode": conditioning_mode,
                "split": "val",
                "split_samples": len(datasets.val),
                "num_roi_groups": split_audit["splits"]["val"]["num_roi_groups"],
                "alphas": alphas,
                "cvar_fraction": float(args.cvar_fraction),
                "train_profile": args.train_profile,
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    results = []
    with torch.inference_mode():
        for alpha in alphas:
            conditioning = fixed_alpha_sweep.conditioning_for_alpha(
                alpha,
                schedule_name=schedule_name,
                total_steps=total_steps,
                conditioning_mode=conditioning_mode,
            )
            model_conditioning_value = float(
                conditioning["model_conditioning_value"]
            )
            sample_losses: list[float] = []
            group_losses: dict[str, list[float]] = defaultdict(list)
            group_seasons: dict[str, str] = {}
            num_batches = 0

            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break

                sample_ids = list(batch["sample_id"])
                batch = fixed_alpha_sweep._move_batch(batch, device)
                target = batch["target"]
                cloudy = batch["cloudy"]
                sar = batch["sar"]

                bridge_state = make_bridge_state(
                    target,
                    cloudy,
                    torch.tensor(alpha, device=device, dtype=target.dtype),
                )
                model_conditioning = torch.full(
                    (target.shape[0],),
                    model_conditioning_value,
                    device=device,
                    dtype=torch.float32,
                )
                prediction = model(bridge_state, model_conditioning, sar)
                per_sample = (
                    torch.abs(prediction - target)
                    .mean(dim=(1, 2, 3))
                    .detach()
                    .cpu()
                    .tolist()
                )

                for sample_id, loss in zip(sample_ids, per_sample):
                    info = sample_meta[sample_id]
                    group = str(info["group"])
                    season = str(info["season"])
                    value = float(loss)
                    sample_losses.append(value)
                    group_losses[group].append(value)
                    group_seasons[group] = season

                num_batches += 1
                batch_number = batch_index + 1
                if (
                    args.progress_every > 0
                    and (
                        batch_number % args.progress_every == 0
                        or batch_number == total_batches
                    )
                ):
                    print(
                        "alpha={alpha:.2f} batch={batch}/{total} samples={samples} "
                        "val_L1={l1:.6f} groups_seen={groups}".format(
                            alpha=alpha,
                            batch=batch_number,
                            total=total_batches,
                            samples=len(sample_losses),
                            l1=sum(sample_losses) / len(sample_losses),
                            groups=len(group_losses),
                        ),
                        flush=True,
                    )

            summary = summarize_group_losses(
                sample_losses=sample_losses,
                group_losses=dict(group_losses),
                group_seasons=group_seasons,
                cvar_fraction=float(args.cvar_fraction),
            )
            result = {
                "requested_alpha": float(alpha),
                **conditioning,
                "num_batches": num_batches,
                **summary,
            }

            if train_profile is not None:
                train_l1 = _lookup_train_l1(train_profile, alpha)
                if train_l1 is None:
                    raise ValueError(
                        f"train profile does not contain requested alpha={alpha}"
                    )
                result["train_mean_l1"] = train_l1
                result["generalization_gaps"] = {
                    "sample_val_minus_train_l1": (
                        float(summary["sample_mean_l1"]) - train_l1
                    ),
                    "equal_group_val_minus_train_l1": (
                        float(summary["equal_group_mean_l1"]) - train_l1
                    ),
                    "worst_group_val_minus_train_l1": (
                        float(summary["worst_group_l1"]) - train_l1
                    ),
                    "cvar_group_val_minus_train_l1": (
                        float(summary["cvar_group_l1"]) - train_l1
                    ),
                }

            results.append(result)

    output_payload = {
        "format_version": 1,
        "diagnostic": "generalization_aware_fixed_alpha_group_risk",
        "analysis_role": "validation_group_risk_for_sampler_design",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "model_identity": metadata["model_identity"],
        "schedule_name": schedule_name,
        "conditioning_mode": conditioning_mode,
        "total_steps": total_steps,
        "split": "val",
        "split_protocol": metadata["split_protocol"],
        "split_audit": split_audit,
        "dataset_membership": membership,
        "batch_size": int(args.batch_size),
        "max_batches": args.max_batches,
        "cvar_fraction": float(args.cvar_fraction),
        "bridge_state_definition": "x_alpha=(1-alpha)*target+alpha*cloudy",
        "final_test_used_for_sampler_design": False,
        "train_profile": args.train_profile,
        "candidate_sampler_signal": (
            "inspect equal_group_mean_l1, worst_group_l1, cvar_group_l1, and "
            "optional validation-minus-training gaps before choosing q(alpha)"
        ),
        "paper_grade": False,
        "results": results,
    }

    output_path = (
        Path(args.output)
        if args.output is not None
        else _default_output_path(checkpoint_path)
    )
    eval_base._write_json_atomic(
        output_path,
        output_payload,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output": str(output_path)}, indent=2), flush=True)
    return output_payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
