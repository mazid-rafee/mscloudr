"""SEN12MS-CR dataset discovery and preprocessing.

The dataset is intentionally kept outside the Git repository.  This module
stores only stable relative sample identities and resolves them against a
runtime data root.

Expected dataset root layout::

    SEN12MS-CR/
      ROIs2017_winter_s2_cloudy/
      ROIs2017_winter_s2/
      ROIs2017_winter_s1/
      ...
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

SEASON_PREFIXES = (
    "ROIs2017_winter",
    "ROIs1158_spring",
    "ROIs1868_summer",
    "ROIs1970_fall",
)

SEASON_ALIASES = {
    "winter": "ROIs2017_winter",
    "spring": "ROIs1158_spring",
    "summer": "ROIs1868_summer",
    "fall": "ROIs1970_fall",
    "autumn": "ROIs1970_fall",
}

OPTICAL_MIN = 0.0
OPTICAL_MAX = 10000.0
SAR_VV_MIN = -25.0
SAR_VH_MIN = -32.5
SAR_MAX = 0.0

_DATA_ENV = "MSCLOUDR_DATA_ROOT"
_VALID_EXTENSIONS = {".tif", ".tiff"}


@dataclass(frozen=True)
class SEN12MSCRSample:
    """Stable identity plus runtime paths for one aligned triplet."""

    sample_id: str
    season_prefix: str
    roi_id: str
    patch_id: str
    cloudy_path: Path
    sar_path: Path
    target_path: Path

    @property
    def group_id(self) -> str:
        """Season-scoped ROI grouping key.

        Cross-season ROI numbers are not assumed to denote the same geography.
        """

        return f"{self.season_prefix}:{self.roi_id}"


@dataclass(frozen=True)
class DiscoveryReport:
    dataset_root: Path
    samples: tuple[SEN12MSCRSample, ...]
    missing_counterparts: tuple[str, ...]
    ignored_sample_ids: tuple[str, ...]

    def counts_by_season(self) -> dict[str, int]:
        counts = {season: 0 for season in SEASON_PREFIXES}
        for sample in self.samples:
            counts[sample.season_prefix] = counts.get(sample.season_prefix, 0) + 1
        return counts


def resolve_dataset_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the SEN12MS-CR root without copying data into the repository.

    Resolution order:
    1. explicit ``path``;
    2. ``MSCLOUDR_DATA_ROOT`` environment variable.

    Both a direct SEN12MS-CR root and its parent directory are accepted.
    """

    value = path if path is not None else os.environ.get(_DATA_ENV)
    if value is None:
        raise ValueError(
            "SEN12MS-CR root not configured. Pass data_root or set "
            f"{_DATA_ENV}."
        )

    candidate = Path(value).expanduser().resolve()
    direct_marker = candidate / f"{SEASON_PREFIXES[0]}_s2_cloudy"
    nested = candidate / "SEN12MS-CR"
    nested_marker = nested / f"{SEASON_PREFIXES[0]}_s2_cloudy"

    if direct_marker.is_dir():
        return candidate
    if nested_marker.is_dir():
        return nested

    raise FileNotFoundError(
        "Could not find SEN12MS-CR season directories under either "
        f"{candidate} or {nested}."
    )


def resolve_seasons(seasons: Sequence[str] | str | None) -> tuple[str, ...]:
    if seasons is None:
        return SEASON_PREFIXES

    if isinstance(seasons, str):
        tokens = [t for t in re.split(r"[\s,]+", seasons.strip()) if t]
    else:
        tokens = [str(t).strip() for t in seasons if str(t).strip()]

    resolved: list[str] = []
    for token in tokens:
        season = SEASON_ALIASES.get(token.lower(), token)
        if season not in SEASON_PREFIXES:
            raise ValueError(f"Unknown season: {token}")
        if season not in resolved:
            resolved.append(season)
    return tuple(resolved)


def normalize_optical(x: torch.Tensor) -> torch.Tensor:
    """Clip Sentinel-2 reflectance-like values to [0, 10000] and scale to [0, 1]."""

    return torch.clamp(x.float(), OPTICAL_MIN, OPTICAL_MAX) / OPTICAL_MAX


def normalize_sar(x: torch.Tensor) -> torch.Tensor:
    """Normalize the two SEN12MS-CR SAR channels to [0, 1].

    Channel 0 (VV): [-25, 0] dB
    Channel 1 (VH): [-32.5, 0] dB
    """

    if x.ndim < 1 or x.shape[0] != 2:
        raise ValueError(f"expected exactly 2 SAR channels, got shape={tuple(x.shape)}")

    vv = (torch.clamp(x[0].float(), SAR_VV_MIN, SAR_MAX) - SAR_VV_MIN) / (
        SAR_MAX - SAR_VV_MIN
    )
    vh = (torch.clamp(x[1].float(), SAR_VH_MIN, SAR_MAX) - SAR_VH_MIN) / (
        SAR_MAX - SAR_VH_MIN
    )
    return torch.stack((vv, vh), dim=0)


def load_ignored_sample_ids(path: str | os.PathLike[str] | None) -> set[str]:
    if path is None:
        return set()

    ignore_path = Path(path)
    if not ignore_path.exists():
        return set()

    ids: set[str] = set()
    for raw_line in ignore_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


def _counterpart_relative_path(
    cloudy_relative_path: Path,
    *,
    modality: str,
) -> Path:
    """Map a cloudy relative path to S1 or cloud-free S2 relative path."""

    if modality not in {"s1", "s2"}:
        raise ValueError("modality must be 's1' or 's2'")

    parts = list(cloudy_relative_path.parts)
    if not parts:
        raise ValueError("empty cloudy relative path")

    # Typical first path component is s2_cloudy_<ROI>.
    parts[0] = parts[0].replace("s2_cloudy_", f"{modality}_", 1)
    parts[-1] = parts[-1].replace("s2_cloudy_", f"{modality}_", 1)
    return Path(*parts)


def _parse_roi_patch(relative_path: Path) -> tuple[str, str]:
    roi_dir = relative_path.parts[0] if relative_path.parts else ""
    roi_match = re.fullmatch(r"s2_cloudy_(\d+)", roi_dir)
    roi_id = roi_match.group(1) if roi_match else roi_dir

    match = re.search(r"_(p\d+)\.(?:tif|tiff)$", relative_path.name, re.IGNORECASE)
    patch_id = match.group(1) if match else relative_path.stem
    return str(roi_id), str(patch_id)


def discover_sen12mscr(
    data_root: str | os.PathLike[str] | None = None,
    *,
    seasons: Sequence[str] | str | None = None,
    ignored_sample_ids: Iterable[str] = (),
    strict_counterparts: bool = True,
) -> DiscoveryReport:
    """Discover aligned cloudy/SAR/clean triplets with stable ordering.

    Sample identity is the cloudy TIFF path relative to the dataset root,
    represented with POSIX separators.  Absolute machine-specific paths never
    enter split manifests.
    """

    root = resolve_dataset_root(data_root)
    season_prefixes = resolve_seasons(seasons)
    ignored = set(ignored_sample_ids)

    samples: list[SEN12MSCRSample] = []
    missing: list[str] = []
    ignored_seen: list[str] = []

    for season in season_prefixes:
        cloudy_dir = root / f"{season}_s2_cloudy"
        sar_dir = root / f"{season}_s1"
        target_dir = root / f"{season}_s2"

        required = (cloudy_dir, sar_dir, target_dir)
        absent = [str(path) for path in required if not path.is_dir()]
        if absent:
            raise FileNotFoundError(
                f"missing modality directories for {season}: {absent}"
            )

        cloudy_paths: list[Path] = []
        for current_root, dirnames, filenames in os.walk(cloudy_dir):
            dirnames.sort()
            for filename in sorted(filenames):
                path = Path(current_root) / filename
                if path.suffix.lower() in _VALID_EXTENSIONS:
                    cloudy_paths.append(path)

        for cloudy_path in cloudy_paths:
            rel_within_cloudy = cloudy_path.relative_to(cloudy_dir)
            sample_id = cloudy_path.relative_to(root).as_posix()

            if sample_id in ignored:
                ignored_seen.append(sample_id)
                continue

            sar_rel = _counterpart_relative_path(rel_within_cloudy, modality="s1")
            target_rel = _counterpart_relative_path(rel_within_cloudy, modality="s2")
            sar_path = sar_dir / sar_rel
            target_path = target_dir / target_rel

            if not sar_path.is_file() or not target_path.is_file():
                missing.append(sample_id)
                continue

            roi_id, patch_id = _parse_roi_patch(rel_within_cloudy)
            samples.append(
                SEN12MSCRSample(
                    sample_id=sample_id,
                    season_prefix=season,
                    roi_id=roi_id,
                    patch_id=patch_id,
                    cloudy_path=cloudy_path,
                    sar_path=sar_path,
                    target_path=target_path,
                )
            )

    samples.sort(key=lambda sample: sample.sample_id)
    missing.sort()
    ignored_seen.sort()

    if strict_counterparts and missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} cloudy samples are missing S1/S2 counterparts; "
            f"first examples: {preview}"
        )

    return DiscoveryReport(
        dataset_root=root,
        samples=tuple(samples),
        missing_counterparts=tuple(missing),
        ignored_sample_ids=tuple(ignored_seen),
    )


def _read_tiff(path: Path) -> torch.Tensor:
    # Delayed import keeps manifest/discovery utilities usable on machines that
    # do not have rasterio installed.
    import rasterio

    with rasterio.open(path) as dataset:
        array = dataset.read()
    return torch.from_numpy(np.asarray(array)).float()


class SEN12MSCRDataset(Dataset):
    """PyTorch dataset backed by externally stored SEN12MS-CR TIFFs."""

    def __init__(
        self,
        samples: Sequence[SEN12MSCRSample],
        *,
        include_sample_id: bool = True,
        strict_channels: bool = True,
    ) -> None:
        self.samples = tuple(samples)
        self.include_sample_id = include_sample_id
        self.strict_channels = strict_channels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]

        cloudy = _read_tiff(sample.cloudy_path)
        sar = _read_tiff(sample.sar_path)
        target = _read_tiff(sample.target_path)

        if self.strict_channels:
            if cloudy.shape[0] != 13 or target.shape[0] != 13:
                raise ValueError(
                    f"{sample.sample_id}: expected 13 optical channels, "
                    f"got cloudy={cloudy.shape[0]}, target={target.shape[0]}"
                )
            if sar.shape[0] != 2:
                raise ValueError(
                    f"{sample.sample_id}: expected 2 SAR channels, got {sar.shape[0]}"
                )

        item: dict[str, torch.Tensor | str] = {
            "cloudy": normalize_optical(cloudy),
            "sar": normalize_sar(sar),
            "target": normalize_optical(target),
        }
        if self.include_sample_id:
            item["sample_id"] = sample.sample_id
        return item
