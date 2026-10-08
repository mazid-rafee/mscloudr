"""Model components for mscloudr."""

from .blocks import (
    LayerNorm2d,
    NAFBlock,
    SFBlock,
    SimpleGate,
    count_trainable_parameters,
)
from .canonical_bridge import (
    CANONICAL_BRIDGE_MODEL_IDENTITY,
    CANONICAL_BRIDGE_OPTICAL_ONLY_MODEL_IDENTITY,
    AlphaFiLM,
    CanonicalBridgeNet,
    CanonicalBridgeOpticalOnlyNet,
    GatedSARFusion,
    PhysicalAlphaEmbedding,
    ResidualConvBlock,
    count_canonical_bridge_optical_only_parameters,
    count_canonical_bridge_parameters,
)
from .dbcr import (
    AuditedDBCRNet,
    DBCRArchitectureConfig,
    HISTORICAL_MSCLOUDR_PARAMETER_COUNT,
    PUBLISHED_PARAMETER_COUNT,
    SinusoidalTimeEmbedding,
)
from .legacy_dbcr import (
    DBCRNet,
    LEGACY_DBCR_PARAMETER_COUNT,
    LegacyDBCRNet,
    LegacyNAFBlock,
    LegacySFBlock,
    count_legacy_parameters,
)
from .patent_blocks import (
    PatentFeatureMapLayerNorm,
    PatentSimplifiedChannelAttention,
    PatentTimeEmbeddedNAFBlock,
    PatentTimeEmbedding,
    PatentTimeEmbeddingConfig,
    sine_bridge_alpha,
)

__all__ = [
    "AlphaFiLM",
    "AuditedDBCRNet",
    "CANONICAL_BRIDGE_MODEL_IDENTITY",
    "CANONICAL_BRIDGE_OPTICAL_ONLY_MODEL_IDENTITY",
    "CanonicalBridgeNet",
    "CanonicalBridgeOpticalOnlyNet",
    "DBCRArchitectureConfig",
    "DBCRNet",
    "GatedSARFusion",
    "HISTORICAL_MSCLOUDR_PARAMETER_COUNT",
    "LEGACY_DBCR_PARAMETER_COUNT",
    "LayerNorm2d",
    "LegacyDBCRNet",
    "LegacyNAFBlock",
    "LegacySFBlock",
    "NAFBlock",
    "PUBLISHED_PARAMETER_COUNT",
    "PatentFeatureMapLayerNorm",
    "PatentSimplifiedChannelAttention",
    "PatentTimeEmbeddedNAFBlock",
    "PatentTimeEmbedding",
    "PatentTimeEmbeddingConfig",
    "PhysicalAlphaEmbedding",
    "ResidualConvBlock",
    "SFBlock",
    "SimpleGate",
    "SinusoidalTimeEmbedding",
    "count_canonical_bridge_optical_only_parameters",
    "count_canonical_bridge_parameters",
    "count_legacy_parameters",
    "count_trainable_parameters",
    "sine_bridge_alpha",
]
