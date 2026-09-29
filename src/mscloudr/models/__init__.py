"""Model components for mscloudr."""

from .blocks import (
    LayerNorm2d,
    NAFBlock,
    SFBlock,
    SimpleGate,
    count_trainable_parameters,
)
from .dbcr import (
    AuditedDBCRNet,
    DBCRArchitectureConfig,
    HISTORICAL_MSCLOUDR_PARAMETER_COUNT,
    PUBLISHED_PARAMETER_COUNT,
    SinusoidalTimeEmbedding,
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
    "AuditedDBCRNet",
    "DBCRArchitectureConfig",
    "HISTORICAL_MSCLOUDR_PARAMETER_COUNT",
    "LayerNorm2d",
    "NAFBlock",
    "PUBLISHED_PARAMETER_COUNT",
    "PatentFeatureMapLayerNorm",
    "PatentSimplifiedChannelAttention",
    "PatentTimeEmbeddedNAFBlock",
    "PatentTimeEmbedding",
    "PatentTimeEmbeddingConfig",
    "SFBlock",
    "SimpleGate",
    "SinusoidalTimeEmbedding",
    "count_trainable_parameters",
    "sine_bridge_alpha",
]
