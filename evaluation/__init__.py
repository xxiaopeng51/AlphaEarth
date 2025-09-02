"""
Evaluation framework for AlphaEarth Foundations model.
"""

from .evaluator import Evaluator
from .metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    SegmentationMetrics,
    MultiModalMetrics
)

__all__ = [
    "Evaluator",
    "ClassificationMetrics",
    "RegressionMetrics",
    "SegmentationMetrics",
    "MultiModalMetrics"
]