"""Dataset utilities."""

from __future__ import annotations

import os

# Default dataset location on the research server.  An explicit --data-root
# argument still takes precedence, and MSCLOUDR_DATA_ROOT can override this
# default when running on another machine.
DEFAULT_LOCAL_DATA_ROOT = (
    "/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Removal/"
    "data/SEN12MS-CR"
)
os.environ.setdefault("MSCLOUDR_DATA_ROOT", DEFAULT_LOCAL_DATA_ROOT)

from .sen12mscr import (
    DiscoveryReport,
    SEN12MSCRDataset,
    SEN12MSCRSample,
    discover_sen12mscr,
    load_ignored_sample_ids,
    normalize_optical,
    normalize_sar,
    resolve_dataset_root,
)

__all__ = [
    "DEFAULT_LOCAL_DATA_ROOT",
    "DiscoveryReport",
    "SEN12MSCRDataset",
    "SEN12MSCRSample",
    "discover_sen12mscr",
    "load_ignored_sample_ids",
    "normalize_optical",
    "normalize_sar",
    "resolve_dataset_root",
]

from .loaders import (
    FROZEN_DATASET_SAMPLE_COUNT,
    FROZEN_DATASET_SAMPLE_IDS_SHA256,
    FROZEN_SPLIT_AUDIT,
    REFERENCE_PROTOCOL,
    ReferenceDataLoaders,
    ReferenceDatasets,
    build_reference_dataloaders,
    build_reference_datasets,
    reference_split_audit,
    validate_frozen_reference_split,
)

__all__ += [
    "FROZEN_DATASET_SAMPLE_COUNT",
    "FROZEN_DATASET_SAMPLE_IDS_SHA256",
    "FROZEN_SPLIT_AUDIT",
    "REFERENCE_PROTOCOL",
    "ReferenceDataLoaders",
    "ReferenceDatasets",
    "build_reference_dataloaders",
    "build_reference_datasets",
    "reference_split_audit",
    "validate_frozen_reference_split",
]
