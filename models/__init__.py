"""
AlphaEarth Foundations - Enhanced Global Multimodal Foundation Model

This package contains the core model implementations for the AlphaEarth Foundations model,
including multi-modal encoders, fusion modules, and task-specific heads.
"""

from .encoders import (
    OpticalEncoder,
    RadarEncoder,
    MeteorologicalEncoder,
    TextEncoder,
    MultiModalEncoder
)

from .fusion import (
    CrossAttentionFusion,
    SpatialTemporalPrecision,
    MultiModalFusion
)

from .heads import (
    ClassificationHead,
    RegressionHead,
    SegmentationHead
)

from .alphaearth import AlphaEarthFoundations

__all__ = [
    "OpticalEncoder",
    "RadarEncoder", 
    "MeteorologicalEncoder",
    "TextEncoder",
    "MultiModalEncoder",
    "CrossAttentionFusion",
    "SpatialTemporalPrecision",
    "MultiModalFusion",
    "ClassificationHead",
    "RegressionHead",
    "SegmentationHead",
    "AlphaEarthFoundations"
]