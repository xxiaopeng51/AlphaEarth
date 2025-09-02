"""
Loss functions for AlphaEarth Foundations model.
"""

from .multimodal_loss import MultiModalLoss
from .contrastive_loss import ContrastiveLoss
from .spatial_temporal_loss import SpatialTemporalLoss

__all__ = [
    "MultiModalLoss",
    "ContrastiveLoss",
    "SpatialTemporalLoss"
]