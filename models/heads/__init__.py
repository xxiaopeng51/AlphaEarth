"""
Task-specific heads for AlphaEarth Foundations model.
"""

from .classification_head import ClassificationHead
from .regression_head import RegressionHead
from .segmentation_head import SegmentationHead

__all__ = [
    "ClassificationHead",
    "RegressionHead",
    "SegmentationHead"
]