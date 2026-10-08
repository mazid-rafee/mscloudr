"""SAR intervention evaluation for pilot CanonicalBridgeNet checkpoints.

This diagnostic keeps the optical input, target, checkpoint, endpoint
conditioning, and test sample order fixed while intervening only on SAR.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mscloudr.checkpointing import CHECKPOINT_FORMAT_VERSION
from mscloudr.cli import eval as base
from mscloudr.data import (
    SEN12MSCRDataset,
    SEN12MSCRSample,
    discover_sen12mscr,
    load_ignored_sample_ids,
    normalize_sar,
    validate_frozen_reference_split,
)
from mscloudr.data.pilot import PILOT_PROTOCOL, pilot_split_audit
from mscloudr.data.sen12mscr_splits import partition_samples_by_uncrtaints_split
from mscloudr.metrics import (
    REFERENCE_COMMIT,
    REFERENCE_REPO,
    ReferenceMetricAccumulator,
)
from mscloudr.models import (
    CANONICAL_BRIDGE_MODEL_IDENTITY,
    CanonicalBridgeNet,
    count_canonical_bridge_parameters,
)
from mscloudr.reproducibility import seed_dataloader_worker, seed_everything
from mscloudr.training import PHYSICAL_ALPHA_CONDITIONING

INTERVENTION_SEED = 42
CONDITIONS = ("NORMAL", "SHUFFLED", "ZERO", "NOISE")


@dataclass(frozen=True)
class PredictionChange:
    count: int
    total: float
    total_sq: float

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def sample_std(self) -> float:
        if self.count < 2:
            return 0.0
        variance = (self.total_sq - (self.total * self.total) / self.count) / (
            self.count - 1
        )
        return math.sqrt(max(0.0, variance))


class _ChangeAccumulator:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, normal: torch.Tensor, altered: torch.Tensor) -> None:
        if normal.shape != altered.shape:
            raise ValueError("prediction shapes must match")
        per_sample = torch.mean(
            torch.abs(altered - normal),
            dim=tuple(range(1, normal.ndim)),
        )
        self.values.extend(float(v) for v in per_sample.detach().cpu())

    def compute(self) -> PredictionChange:
        if not self.values:
            raise ValueError("no prediction changes were accumulated")
        return PredictionChange(
            count=len(self.values),
            total=sum(self.values),
            total_sq=sum(v * v for v in self.values),
        )


def build_parser():
    parser = base.build_parser()
    parser.description = (
        "Evaluate CanonicalBridgeNet SAR influence on the exact fixed pilot10 "
        "test subset using NORMAL, SHUFFLED, ZERO, and NOISE SAR."
    )
    parser.add_argument(
        "--pilot-manifest",
        default=None,
        help=(
            "Exact pilot_subset_manifest.json. Defaults to the run directory "
            "containing the checkpoint."
        ),
    )
    parser.add_argument(
        "--intervention-seed",
        type=int,
        default=INTERVENTION_SEED,
        help="Seed for the dataset-level SAR derangement and deterministic noise.",
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
        raise ValueError("expected model_identity=canonical_bridge_net")
    if metadata.get("split_protocol") != PILOT_PROTOCOL:
        raise ValueError("checkpoint does not use the fixed pilot10 split")
    if metadata.get("schedule_name") != "canonical_alpha":
        raise ValueError("SAR intervention expects canonical_alpha")
    if metadata.get("conditioning_mode") != PHYSICAL_ALPHA_CONDITIONING:
        raise ValueError(
            "pilot SAR intervention currently expects physical_alpha conditioning"
        )
    if metadata.get("bridge_geometry") != "straight_linear":
        raise ValueError("SAR intervention expects straight bridge geometry")
    if int(metadata.get("total_steps", 0)) <= 0:
        raise ValueError("checkpoint metadata has invalid total_steps")
    if int(metadata.get("trainable_parameters", 0)) <= 0:
        raise ValueError("checkpoint metadata has invalid trainable_parameters")
    return metadata


def make_derangement(n: int, seed: int = INTERVENTION_SEED) -> list[int]:
    """Return a deterministic Sattolo derangement over range(n)."""

    n = int(n)
    seed = int(seed)
    if n < 2:
        raise ValueError("derangement requires at least two samples")
    if seed < 0:
        raise ValueError("intervention seed must be non-negative")

    permutation = list(range(n))
    rng = random.Random(seed)
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        permutation[i], permutation[j] = permutation[j], permutation[i]

    if any(index == donor for index, donor in enumerate(permutation)):
        raise RuntimeError("Sattolo derangement unexpectedly contains fixed points")
    return permutation


def derangement_fixed_points(permutation: Sequence[int]) -> int:
    return sum(i == int(j) for i, j in enumerate(permutation))


def _stable_noise_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def deterministic_noise_sar(
    reference: torch.Tensor,
    sample_ids: Sequence[str],
    *,
    seed: int = INTERVENTION_SEED,
) -> torch.Tensor:
    """Generate deterministic i.i.d. U[0,1] noise in normalized SAR space.

    A stable SHA-256-derived seed is used for each sample, making the noise
    independent of batch size, worker count, and evaluation batching.
    """

    if reference.ndim != 4:
        raise ValueError("reference SAR must have shape [B,C,H,W]")
    if len(sample_ids) != reference.shape[0]:
        raise ValueError("sample_ids length must match SAR batch size")

    samples: list[torch.Tensor] = []
    sample_shape = tuple(reference.shape[1:])
    for sample_id in sample_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_noise_seed(seed, str(sample_id)))
        samples.append(
            torch.rand(
                sample_shape,
                generator=generator,
                dtype=reference.dtype,
                device="cpu",
            )
        )
    return torch.stack(samples, dim=0)


def intervention_sar(
    condition: str,
    normal_sar: torch.Tensor,
    *,
    shuffled_sar: torch.Tensor | None = None,
    sample_ids: Sequence[str] | None = None,
    seed: int = INTERVENTION_SEED,
) -> torch.Tensor:
    condition = str(condition).upper()
    if condition == "NORMAL":
        return normal_sar
    if condition == "SHUFFLED":
        if shuffled_sar is None:
            raise ValueError("SHUFFLED requires shuffled_sar")
        if shuffled_sar.shape != normal_sar.shape:
            raise ValueError("shuffled SAR shape must match normal SAR")
        return shuffled_sar
    if condition == "ZERO":
        return torch.zeros_like(normal_sar)
    if condition == "NOISE":
        if sample_ids is None:
            raise ValueError("NOISE requires sample_ids")
        noise = deterministic_noise_sar(normal_sar.cpu(), sample_ids, seed=seed)
        return noise.to(device=normal_sar.device, dtype=normal_sar.dtype)
    raise ValueError(f"unknown SAR intervention condition: {condition!r}")


def _read_normalized_sar(path: Path) -> torch.Tensor:
    import rasterio

    with rasterio.open(path) as dataset:
        array = dataset.read()
    return normalize_sar(torch.from_numpy(np.asarray(array)).float())


class _SARInterventionDataset(Dataset):
    """Base pilot item plus SAR from a deterministic donor sample."""

    def __init__(
        self,
        samples: Sequence[SEN12MSCRSample],
        permutation: Sequence[int],
    ) -> None:
        self.samples = tuple(samples)
        self.permutation = tuple(int(v) for v in permutation)
        if len(self.samples) != len(self.permutation):
            raise ValueError("permutation length must equal dataset length")
        if sorted(self.permutation) != list(range(len(self.samples))):
            raise ValueError("permutation must contain every dataset index exactly once")
        if derangement_fixed_points(self.permutation) != 0:
            raise ValueError("shuffled SAR permutation must have zero fixed points")
        self.base = SEN12MSCRDataset(
            self.samples,
            include_sample_id=True,
            strict_channels=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        donor = self.samples[self.permutation[index]]
        item["sar_shuffled"] = _read_normalized_sar(donor.sar_path)
        item["sar_donor_sample_id"] = donor.sample_id
        return item


def _fingerprint_ids(ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_manifest_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "pilot_subset_manifest.json"
    return checkpoint_path.with_name("pilot_subset_manifest.json")


def _default_output_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "sar_intervention_metrics.json"
    return checkpoint_path.with_name(
        checkpoint_path.stem + "_sar_intervention_metrics.json"
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"pilot manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("pilot manifest must contain a JSON object")
    if manifest.get("protocol") != PILOT_PROTOCOL:
        raise ValueError("pilot manifest protocol mismatch")

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("pilot manifest is missing splits")

    for split in ("train", "val", "test"):
        entry = splits.get(split)
        if not isinstance(entry, Mapping):
            raise ValueError(f"pilot manifest missing split {split!r}")
        ids = entry.get("sample_ids")
        if not isinstance(ids, list) or not all(isinstance(v, str) for v in ids):
            raise ValueError(f"pilot manifest split {split!r} is missing sample_ids")
        if len(ids) != len(set(ids)):
            raise ValueError(f"pilot manifest split {split!r} contains duplicate IDs")
        if _fingerprint_ids(ids) != entry.get("sample_ids_sha256"):
            raise ValueError(f"pilot manifest split {split!r} fingerprint mismatch")

    all_ids = [
        sample_id
        for split in ("train", "val", "test")
        for sample_id in splits[split]["sample_ids"]
    ]
    if _fingerprint_ids(all_ids) != manifest.get("sample_ids_sha256"):
        raise ValueError("pilot manifest global sample-ID fingerprint mismatch")
    return manifest


def _exact_test_samples(
    discovered: Sequence[SEN12MSCRSample],
    manifest: Mapping[str, Any],
) -> list[SEN12MSCRSample]:
    partitions = partition_samples_by_uncrtaints_split(discovered)
    validate_frozen_reference_split(partitions)

    test_ids = manifest["splits"]["test"]["sample_ids"]
    by_id = {sample.sample_id: sample for sample in partitions["test"]}
    missing = [sample_id for sample_id in test_ids if sample_id not in by_id]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"{len(missing)} manifest test samples are missing from the fixed "
            f"test split; first examples: {preview}"
        )

    samples = [by_id[sample_id] for sample_id in test_ids]
    if _fingerprint_ids([sample.sample_id for sample in samples]) != manifest[
        "splits"
    ]["test"]["sample_ids_sha256"]:
        raise ValueError("selected test sample fingerprint mismatch")
    return samples


def _move_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, non_blocking=True)


def _metrics_with_l1(
    accumulator: ReferenceMetricAccumulator,
) -> tuple[dict[str, float], dict[str, int]]:
    metrics = accumulator.compute()
    counts = accumulator.metric_counts()
    if "MAE" in metrics:
        metrics["L1"] = metrics["MAE"]
        counts["L1"] = counts.get("MAE", 0)
    return metrics, counts


def _metric_deltas(
    metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    normal = metrics["NORMAL"]
    return {
        condition: {
            name: float(value) - float(normal[name])
            for name, value in condition_metrics.items()
            if name in normal
        }
        for condition, condition_metrics in metrics.items()
        if condition != "NORMAL"
    }


def _assert_output_schema(payload: Mapping[str, Any]) -> None:
    required = {
        "checkpoint",
        "checkpoint_epoch",
        "model_identity",
        "trainable_parameters",
        "num_samples",
        "intervention_seed",
        "pilot_manifest_sample_ids_sha256",
        "intervention_definitions",
        "metrics",
        "deltas_vs_normal",
        "prediction_changes_vs_normal",
        "shuffled_permutation_fixed_points",
        "shuffled_is_derangement",
        "conditioning_mode",
        "endpoint_conditioning_value",
        "nfe",
        "inference",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "SAR intervention output missing fields: " + ", ".join(missing)
        )
    if set(payload["metrics"]) != set(CONDITIONS):
        raise ValueError("metrics must contain all four SAR conditions")


def run(args) -> dict[str, Any]:
    base.validate_cli_args(args)
    if args.intervention_seed < 0:
        raise ValueError("intervention-seed must be non-negative")

    checkpoint_path = Path(args.checkpoint)
    payload = base._torch_load_payload(checkpoint_path)
    metadata = _validate_checkpoint(payload)

    manifest_path = (
        Path(args.pilot_manifest)
        if args.pilot_manifest is not None
        else _default_manifest_path(checkpoint_path)
    )
    manifest = _load_manifest(manifest_path)
    split_audit = pilot_split_audit(manifest)
    if metadata.get("split_audit") != split_audit:
        raise ValueError("exact pilot manifest audit does not match checkpoint")
    if metadata.get("pilot_manifest_sample_ids_sha256") != manifest.get(
        "sample_ids_sha256"
    ):
        raise ValueError(
            "pilot manifest sample-ID fingerprint does not match checkpoint"
        )

    device = base.resolve_device(args.device)
    train_seed = base._train_seed_from_metadata(metadata)
    seed_everything(
        train_seed,
        deterministic_algorithms=base._deterministic_algorithms_from_metadata(
            metadata
        ),
    )

    ignored = load_ignored_sample_ids(args.ignore_file)
    data_root = (
        args.data_root
        if args.data_root is not None
        else metadata.get("data_root")
    )
    discovery = discover_sen12mscr(
        data_root,
        ignored_sample_ids=ignored,
        strict_counterparts=True,
    )
    test_samples = _exact_test_samples(discovery.samples, manifest)

    permutation = make_derangement(len(test_samples), args.intervention_seed)
    fixed_points = derangement_fixed_points(permutation)
    dataset = _SARInterventionDataset(test_samples, permutation)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=seed_dataloader_worker,
        persistent_workers=(args.num_workers > 0),
    )

    model = CanonicalBridgeNet(total_steps=int(metadata["total_steps"]))
    trainable_parameters = count_canonical_bridge_parameters(model)
    if trainable_parameters != int(metadata["trainable_parameters"]):
        raise RuntimeError(
            "canonical bridge parameter count does not match checkpoint metadata"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()

    accumulators = {
        condition: ReferenceMetricAccumulator()
        for condition in CONDITIONS
    }
    change_accumulators = {
        condition: _ChangeAccumulator()
        for condition in CONDITIONS
        if condition != "NORMAL"
    }

    max_batches = args.max_batches
    total_batches = (
        len(loader)
        if max_batches is None
        else min(len(loader), max_batches)
    )
    num_samples = 0
    num_batches = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            sample_ids = [str(value) for value in batch["sample_id"]]
            cloudy = _move_tensor(batch["cloudy"], device)
            target = _move_tensor(batch["target"], device)
            normal_sar = _move_tensor(batch["sar"], device)
            shuffled_sar = _move_tensor(batch["sar_shuffled"], device)
            conditioning = torch.full(
                (target.shape[0],),
                float(metadata["total_steps"]),
                device=device,
                dtype=target.dtype,
            )

            sar_inputs = {
                condition: intervention_sar(
                    condition,
                    normal_sar,
                    shuffled_sar=shuffled_sar,
                    sample_ids=sample_ids,
                    seed=args.intervention_seed,
                )
                for condition in CONDITIONS
            }
            predictions = {
                condition: model(
                    cloudy,
                    conditioning,
                    sar_inputs[condition],
                )
                for condition in CONDITIONS
            }

            normal_prediction = predictions["NORMAL"]
            for condition in CONDITIONS:
                prediction = predictions[condition]
                if prediction.shape != target.shape:
                    raise ValueError(
                        f"{condition} prediction shape does not match target"
                    )
                accumulators[condition].update_batch(target, prediction)
                if condition != "NORMAL":
                    change_accumulators[condition].update(
                        normal_prediction,
                        prediction,
                    )

            num_samples += int(target.shape[0])
            num_batches += 1

            batch_number = batch_index + 1
            if args.progress_every > 0 and (
                batch_number % args.progress_every == 0
                or batch_number == total_batches
            ):
                running = {
                    condition: _metrics_with_l1(
                        accumulators[condition]
                    )[0].get("L1")
                    for condition in CONDITIONS
                }
                print(
                    "sar-eval batch={}/{} samples={} "
                    "NORMAL_L1={:.6f} SHUFFLED_L1={:.6f} "
                    "ZERO_L1={:.6f} NOISE_L1={:.6f}".format(
                        batch_number,
                        total_batches,
                        num_samples,
                        running["NORMAL"],
                        running["SHUFFLED"],
                        running["ZERO"],
                        running["NOISE"],
                    ),
                    flush=True,
                )

    if num_samples == 0:
        raise ValueError("SAR intervention loader produced no samples")

    metrics: dict[str, dict[str, float]] = {}
    metric_counts: dict[str, dict[str, int]] = {}
    for condition in CONDITIONS:
        condition_metrics, condition_counts = _metrics_with_l1(
            accumulators[condition]
        )
        metrics[condition] = condition_metrics
        metric_counts[condition] = condition_counts

    prediction_changes: dict[str, dict[str, float | int]] = {}
    for condition, accumulator in change_accumulators.items():
        change = accumulator.compute()
        prediction_changes[condition] = {
            "mean_absolute_prediction_change": change.mean,
            "per_sample_mean_absolute_change_mean": change.mean,
            "per_sample_mean_absolute_change_std": change.sample_std,
            "num_samples": change.count,
        }

    output_payload: dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "checkpoint_metrics": payload.get("metrics", {}),
        "checkpoint_selection_metric": metadata.get(
            "checkpoint_selection_metric"
        ),
        "model_identity": CANONICAL_BRIDGE_MODEL_IDENTITY,
        "model_family": metadata.get("model_family"),
        "trainable_parameters": trainable_parameters,
        "conditioning_mode": metadata["conditioning_mode"],
        "conditioning_identity": metadata.get("conditioning_identity"),
        "endpoint_conditioning_value": int(metadata["total_steps"]),
        "total_steps": int(metadata["total_steps"]),
        "nfe": 1,
        "inference": "endpoint_direct_x0",
        "schedule_used_during_inference": False,
        "bridge_geometry": metadata.get("bridge_geometry"),
        "split": "test",
        "split_protocol": PILOT_PROTOCOL,
        "split_audit": split_audit,
        "pilot_manifest": str(manifest_path),
        "pilot_manifest_sample_ids_sha256": manifest[
            "sample_ids_sha256"
        ],
        "num_samples": num_samples,
        "num_batches": num_batches,
        "batch_size": int(args.batch_size),
        "max_batches": args.max_batches,
        "paper_grade": False,
        "intervention_seed": int(args.intervention_seed),
        "intervention_definitions": {
            "NORMAL": "correct paired normalized Sentinel-1 SAR",
            "SHUFFLED": (
                "real normalized SAR from a different pilot-test sample using "
                "one deterministic dataset-level Sattolo derangement"
            ),
            "ZERO": "torch.zeros_like(normalized SAR)",
            "NOISE": (
                "i.i.d. Uniform[0,1] in normalized SAR space; each sample uses "
                "a stable SHA-256-derived seed from intervention_seed and sample_id"
            ),
        },
        "shuffled_permutation_fixed_points": fixed_points,
        "shuffled_is_derangement": fixed_points == 0,
        "metrics": metrics,
        "metric_counts": metric_counts,
        "deltas_vs_normal": _metric_deltas(metrics),
        "prediction_changes_vs_normal": prediction_changes,
        "metric_reference_repo": REFERENCE_REPO,
        "metric_reference_commit": REFERENCE_COMMIT,
        "prediction_clipping": False,
    }
    _assert_output_schema(output_payload)

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
                "num_samples": num_samples,
                "shuffled_fixed_points": fixed_points,
                "metrics": metrics,
                "prediction_changes_vs_normal": prediction_changes,
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
